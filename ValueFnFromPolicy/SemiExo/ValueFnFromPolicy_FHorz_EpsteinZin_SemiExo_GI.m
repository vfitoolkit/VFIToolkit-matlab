function V=ValueFnFromPolicy_FHorz_EpsteinZin_SemiExo_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Compute V from a given Policy when the model uses Epstein-Zin preferences
% AND has semi-exogenous shocks (n_semiz>0) AND uses the grid interpolation
% layer (vfoptions.gridinterplayer==1). Mirrors ValueFnFromPolicy_FHorz_SemiExo_GI
% with the Epstein-Zin transform chain: V' is transformed by ^ezc5 BEFORE the
% expectations (one joint certainty-equivalent over (semizprime,zprime,eprime),
% with the semiz transition at the policy's own d_semiz choice), the L2
% interpolation weights are applied to the TRANSFORMED per-d_semiz continuation
% (the same object the EZ GI solver interpolates), and the certainty-equivalent
% power ^ezc6 is applied pointwise AFTER the interpolated per-state lookup.
% The warm-glow fn is evaluated exactly on the fine grid and indexed at the
% policy's fine point. The ezc1-ezc8/sj/warmglow preamble is done by the caller
% (ValueFnFromPolicy_FHorz_EpsteinZin) and passed through.
%
% Policy under GI has shape (l_d+l_aprime+1, n_a, n_semiz, [n_z,] [n_e,] N_j),
% where the last component along dim 1 is the L2 (layer-2) fine-grid index.
% Only the first aprime component (a1) is interpolated; remaining aprime components stay on the standard grid.

%% Setup
% Need semiz gridvals and pi_semiz
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end

% z gridvals
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);

if isfield(vfoptions,'l_dsemiz')
    l_dsemiz=vfoptions.l_dsemiz;
else
    l_dsemiz=1;
end
n_dsemiz=n_d(end-l_dsemiz+1:end);
N_dsemiz=prod(n_dsemiz);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_EpsteinZin_SemiExo_GI: SemiExo requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);
l_aprime=l_a;

% Grid interpolation parameters
n2short=vfoptions.ngridinterp; % evenly spaced points between each pair of a_grid points (a1)
n2aprime=n_a(1)+(n_a(1)-1)*n2short; % number of points in the fine a1 grid
aprime_grid=interp1(1:1:n_a(1),a_grid(1:n_a(1)),linspace(1,n_a(1),n2aprime)); % the fine a1 grid (used for warm-glow)

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

%% PolicyValues (PolicyInd2Val handles GI internally; returns interpolated aprime values)
if N_e==0
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks, l_d+l_aprime, N_j]
else
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks*N_e, l_d+l_aprime, N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
end
l_daprime=size(PolicyValues,1);

%% Extract per-state indices from Policy: d_semiz_idx, aprime1_midpoint, aprime_other_idx, L2_idx
% Strip trailing L2flag channel if present (Policy may carry it; we only need l_d+l_aprime+1 channels)
if size(Policy,1) > (l_d+l_aprime+1)
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:l_d+l_aprime+1,:), [(l_d+l_aprime+1), tempsize(2:end)]);
end

% Reshape Policy to Kron form
if N_e==0
    Policy_k=reshape(Policy,[l_d+l_aprime+1, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_aprime+1, N_a, N_shocks, N_e, N_j]);
end

% d_semiz_idx: last l_dsemiz components of d
if N_e==0
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end

% aprime: position l_d+1 is a1 midpoint; positions l_d+2..l_d+l_aprime are other aprime; position l_d+l_aprime+1 is L2_idx
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:,:),1); % [N_a, N_shocks, N_j] or [N_a, N_shocks, N_e, N_j]
L2_idx=shiftdim(Policy_k(l_d+l_aprime+1,:,:,:,:),1);

% Build a1 fine index: a1_fine = (n2short+1)*(a1_mid-1) + L2_idx
a1_fine_idx=(n2short+1)*(a1_mid-1)+L2_idx;

% Convert a1_fine_idx to fractional position in original a1 grid: frac = 1 + (a1_fine_idx-1)/(n2short+1)
a1_frac=1+(a1_fine_idx-1)/(n2short+1);
a1_lower=floor(a1_frac);
a1_weight=a1_frac-a1_lower; % weight on upper point
% Clamp upper to n_a(1)
a1_upper=min(a1_lower+1, n_a(1));
% When at the top exactly (a1_frac == n_a(1)), weight should be 0 and upper=lower
a1_upper(a1_lower>=n_a(1))=n_a(1);
a1_lower(a1_lower<1)=1;

% Build full kron-style lower and upper aprime index over n_a.
% Only a1 is interpolated; remaining aprime components stay on the standard grid
% and contribute the same offset to both lower and upper.
% aprime_fine_idx is the policy's fine-grid index over the full aprime (used for warm-glow).
if l_a==1
    %% GI1
    aprime_lower_idx=a1_lower;
    aprime_upper_idx=a1_upper;
    aprime_fine_idx=a1_fine_idx;
    n_a_fine=n2aprime;
    a_grid_fine=aprime_grid(:);
elseif l_a==2
    %% GI2A: a1 is interpolated, a2 is on the standard grid
    a2_idx=shiftdim(Policy_k(l_d+2,:,:,:,:),1);
    a2_off=n_a(1)*(a2_idx-1);
    aprime_lower_idx=a1_lower+a2_off;
    aprime_upper_idx=a1_upper+a2_off;
    aprime_fine_idx=a1_fine_idx+n2aprime*(a2_idx-1);
    n_a_fine=[n2aprime,n_a(2)];
    a_grid_fine=[aprime_grid(:); a_grid(n_a(1)+1:end)];
else
    error('ValueFnFromPolicy_FHorz_EpsteinZin_SemiExo_GI: only l_a==1 (GI1) and l_a==2 (GI2A) are supported')
end

%% Joint shock gridvals for ReturnFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J;
else
    joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
    for jj=1:N_j
        joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
    end
end

%% Backward iteration
if N_e==0
    V=zeros(N_a, N_shocks, N_j, 'gpuArray');
else
    V=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
end

[~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz);
if N_z>0
    [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);
end

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Evaluate ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        F_jj=reshape(F_jj,[N_a, N_shocks]);
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
    end

    DiscountFactorParamsVec=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
    if vfoptions.EZoneminusbeta==1
        ezc1=1-DiscountFactorParamsVec;
    elseif vfoptions.EZoneminusbeta==2
        ezc1=1-sj(jj)*DiscountFactorParamsVec;
    end

    % If there is a warm-glow, evaluate the warmglowfn on the fine grid, then at the policy's fine point
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixfineraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a_fine, a_grid_fine, WGParamsVec);
        WGmatrixfine=WGmatrixfineraw;
        WGmatrixfine(isfinite(WGmatrixfineraw))=(ezc4*WGmatrixfineraw(isfinite(WGmatrixfineraw))).^ezc5(jj);
        WGmatrixfine(WGmatrixfineraw==0)=0; % otherwise zero to negative power is set to infinity
        if N_e==0
            WGofPolicy=reshape(WGmatrixfine(aprime_fine_idx(:,:,jj)),[N_a,N_shocks]);
        else
            WGofPolicy=reshape(WGmatrixfine(aprime_fine_idx(:,:,:,jj)),[N_a,N_shocks,N_e]);
        end
    end

    if jj==N_j
        % Modify the Return Function appropriately for Epstein-Zin Preferences
        Vjj=F_jj;
        becareful=logical(isfinite(F_jj).*(F_jj~=0)); % finite but not zero
        Vjj(becareful)=(ezc1*F_jj(becareful).^ezc2(N_j)).^ezc7(N_j);
        Vjj(F_jj==0)=-Inf;
        if warmglow==1
            becareful2=(WGofPolicy==0);
            WGofPolicy(isfinite(WGofPolicy))=ezc3*DiscountFactorParamsVec*(((1-sj(N_j))*WGofPolicy(isfinite(WGofPolicy)).^ezc8(N_j)).^ezc6(N_j));
            WGofPolicy(becareful2)=0;
            Vjj=Vjj+WGofPolicy;
        end
        if N_e==0
            V(:,:,jj)=Vjj;
        else
            V(:,:,:,jj)=Vjj;
        end
    else
        % Part of Epstein-Zin is before taking expectation (transform V' by ^ezc5 BEFORE the
        % e/z/semiz sums: one joint certainty-equivalent over (semizprime,zprime,eprime))
        if N_e==0
            V_nextpre=V(:,:,jj+1); % [N_a, N_shocks]
        else
            V_nextpre=V(:,:,:,jj+1); % [N_a, N_shocks, N_e]
        end
        temp=V_nextpre;
        temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
        temp(V_nextpre==0)=0;

        if N_e==0
            V_next=temp; % [N_a, N_shocks]
        else
            % Integrate over e' using iid pi_e_J(:,jj+1) (the distribution of the e realized in period jj+1)
            V_next=sum(temp .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_shocks, 1]
            V_next(isnan(V_next))=0; % multiplications of -Inf with 0 give NaN, replace with zeros (the zeros come from the probabilities)
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Integrate over z' and per d_semiz over semiz' (on the TRANSFORMED object)
        if N_z==0
            V_next_r=V_next; % [N_a, N_semiz]
            % Step 1: integrate over z' is trivial (no z)
            EV_after_z=V_next_r; % [N_a, N_semiz_to]
            % Step 2: for each d_semiz, integrate over semiz'
            EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from]
                EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2); % [N_a, 1, N_semiz_from]
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
            end
        else
            V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
            % Step 1: integrate over z' (does not depend on d_semiz)
            EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3); % [N_a, N_semiz_to, 1, N_z_from]
            EV_after_z(isnan(EV_after_z))=0;
            EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]); % [anext, semiz_to, z_from]
            % Step 2: for each d_semiz, integrate over semiz'
            EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from]
                pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]); % [1, N_semiz_to, 1, N_semiz_from]
                EVd2c=sum(EV_after_z .* pi_reshape, 2); % [N_a, 1, N_z, N_semiz_from]
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
            end
        end

        % Per-state INTERPOLATED lookup on aprime: the L2 weights are applied to the
        % TRANSFORMED per-d_semiz continuation (BEFORE the certainty-equivalent power ^ezc6)
        if N_e==0
            aprime_lo_jj=aprime_lower_idx(:,:,jj);
            aprime_up_jj=aprime_upper_idx(:,:,jj);
            w_jj=a1_weight(:,:,jj);
            d2_jj=d_semiz_idx(:,:,jj);
            if N_z==0
                aprime_lo_r=reshape(aprime_lo_jj, [N_a, N_semiz]);
                aprime_up_r=reshape(aprime_up_jj, [N_a, N_semiz]);
                w_r=reshape(w_jj, [N_a, N_semiz]);
                d2_r=reshape(d2_jj, [N_a, N_semiz]);
                base_off=N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_r(:)-1);
                lo_idx=aprime_lo_r(:)+base_off;
                up_idx=aprime_up_r(:)+base_off;
                EVnextOfPolicy=reshape((1-w_r(:)).*EVnext_byd2(lo_idx)+w_r(:).*EVnext_byd2(up_idx), [N_a, N_shocks]);
            else
                aprime_lo_r=reshape(aprime_lo_jj, [N_a, N_semiz, N_z]);
                aprime_up_r=reshape(aprime_up_jj, [N_a, N_semiz, N_z]);
                w_r=reshape(w_jj, [N_a, N_semiz, N_z]);
                d2_r=reshape(d2_jj, [N_a, N_semiz, N_z]);
                base_off=N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1);
                lo_idx=aprime_lo_r(:)+base_off;
                up_idx=aprime_up_r(:)+base_off;
                EVnextOfPolicy=reshape((1-w_r(:)).*EVnext_byd2(lo_idx)+w_r(:).*EVnext_byd2(up_idx), [N_a, N_shocks]);
            end
        else
            if N_z==0
                EVnextOfPolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
                for e_c=1:N_e
                    aprime_lo_e=reshape(aprime_lower_idx(:,:,e_c,jj), [N_a, N_semiz]);
                    aprime_up_e=reshape(aprime_upper_idx(:,:,e_c,jj), [N_a, N_semiz]);
                    w_e=reshape(a1_weight(:,:,e_c,jj), [N_a, N_semiz]);
                    d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz]);
                    base_off=N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1);
                    lo_idx=aprime_lo_e(:)+base_off;
                    up_idx=aprime_up_e(:)+base_off;
                    EVnextOfPolicy(:,:,e_c)=reshape((1-w_e(:)).*EVnext_byd2(lo_idx)+w_e(:).*EVnext_byd2(up_idx), [N_a, N_semiz]);
                end
            else
                EVnextOfPolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                for e_c=1:N_e
                    aprime_lo_e=reshape(aprime_lower_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                    aprime_up_e=reshape(aprime_upper_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                    w_e=reshape(a1_weight(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                    d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                    base_off=N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1);
                    lo_idx=aprime_lo_e(:)+base_off;
                    up_idx=aprime_up_e(:)+base_off;
                    EVnextOfPolicy(:,:,:,e_c)=reshape((1-w_e(:)).*EVnext_byd2(lo_idx)+w_e(:).*EVnext_byd2(up_idx), [N_a, N_semiz, N_z]);
                end
            end
            EVnextOfPolicy=reshape(EVnextOfPolicy, [N_a, N_shocks, N_e]);
        end
        EVnextOfPolicy(isnan(EVnextOfPolicy))=0; % the interpolation weights are probabilities: 0*(-Inf) gives NaN, replace with zeros (goes beyond the vNM _SemiExo_GI, which has no clear here)

        % Certainty-equivalent (and mortality-risk/warm-glow) transform, pointwise at the policy
        temp4=EVnextOfPolicy;
        if warmglow==1
            becareful=logical(isfinite(temp4).*isfinite(WGofPolicy)); % both are finite
            temp4(becareful)=(sj(jj)*temp4(becareful).^ezc8(jj)+(1-sj(jj))*WGofPolicy(becareful).^ezc8(jj)).^ezc6(jj);
            temp4((EVnextOfPolicy==0)&(WGofPolicy==0))=0; % Is actually zero
        else % not using warmglow
            temp4(isfinite(temp4))=(sj(jj)*temp4(isfinite(temp4)).^ezc8(jj)).^ezc6(jj);
            temp4(EVnextOfPolicy==0)=0;
        end

        % Modify the Return Function appropriately for Epstein-Zin Preferences
        becareful=logical(isfinite(F_jj).*(F_jj~=0)); % finite but not zero
        temp2=F_jj;
        temp2(becareful)=F_jj(becareful).^ezc2(jj);
        temp2(F_jj==0)=-Inf;

        Vjj=ezc1*temp2+ezc3*DiscountFactorParamsVec*temp4;

        temp5=logical(isfinite(Vjj).*(Vjj~=0));
        Vjj(temp5)=Vjj(temp5).^ezc7(jj);  % matlab otherwise puts 0 to negative power to infinity
        Vjj(Vjj==0)=-Inf;

        if N_e==0
            V(:,:,jj)=Vjj;
        else
            V(:,:,:,jj)=Vjj;
        end
    end
end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, n_semiz, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, n_semiz, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_semiz, n_z, N_j]);
else
    V=reshape(V, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end


end
