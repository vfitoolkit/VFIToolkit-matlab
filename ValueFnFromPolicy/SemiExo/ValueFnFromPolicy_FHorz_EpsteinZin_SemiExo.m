function V=ValueFnFromPolicy_FHorz_EpsteinZin_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Compute V from a given Policy when the model uses Epstein-Zin preferences
% AND has semi-exogenous shocks (n_semiz>0). Mirrors ValueFnFromPolicy_FHorz_SemiExo
% with the Epstein-Zin transform chain: V' is transformed by ^ezc5 BEFORE the
% expectations (one joint certainty-equivalent over (semizprime,zprime,eprime),
% with the semiz transition at the policy's own d_semiz choice), and the
% certainty-equivalent power ^ezc6 is applied pointwise after the per-state
% lookup. The ezc1-ezc8/sj/warmglow preamble is done by the caller
% (ValueFnFromPolicy_FHorz_EpsteinZin) and passed through.

%% GI variant is step 3 of the EZ-SemiExo build
if vfoptions.gridinterplayer==1
    error('ValueFnFromPolicy_FHorz_EpsteinZin_SemiExo: not yet implemented for gridinterplayer (Epstein-Zin SemiExo with grid interpolation layer is itself not yet implemented)')
end

%% Setup
% Need semiz gridvals and pi_semiz: SemiExogShockSetup_FHorz populates vfoptions.semiz_gridvals_J and vfoptions.pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end

% z gridvals (parent already called ExogShockSetup_FHorz, but it returned z_gridvals_J/pi_z_J as locals; re-run here)
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);

% l_dsemiz: number of d variables that affect the semi-exogenous state (last l_dsemiz of d)
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
    error('ValueFnFromPolicy_FHorz_EpsteinZin_SemiExo: SemiExo requires at least one decision variable')
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=l_a;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

% Treat (semiz, z) as joint shock for PolicyInd2Val and ReturnFn evaluation
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

%% PolicyValues for ReturnFn evaluation
if N_e==0
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
    % PolicyValues shape: [l_d+l_aprime, N_a, N_shocks, N_j]
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks, l_d+l_aprime, N_j]
else
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1); % PolicyInd2Val auto-adds vfoptions.n_semiz and vfoptions.n_e
    % PolicyValues shape: [l_d+l_aprime, N_a, N_shocks*N_e, N_j]
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_shocks*N_e, l_d+l_aprime, N_j] — keep shock dim combined for EvalFnOnAgentDist_Grid
end
l_daprime=size(PolicyValues,1);

%% Extract per-state indices: aprime_idx, d_semiz_idx
% Reshape Policy to Kron form
if N_e==0
    Policy_k=reshape(Policy,[l_d+l_aprime, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_aprime, N_a, N_shocks, N_e, N_j]);
end

% d_semiz components are positions (l_d-l_dsemiz+1) through l_d
if N_e==0
    d_semiz_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1); % drop leading singleton
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end

% aprime components are positions (l_d+1) through (l_d+l_aprime)
if N_e==0
    aprime_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    aprime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_a=[1, cumprod(n_a(1:end-1))];
for ii=1:l_aprime
    comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
    aprime_idx=aprime_idx+cumprods_a(ii)*(comp-1);
end

%% Joint shock gridvals for ReturnFn
if N_z==0
    joint_gridvals_J=semiz_gridvals_J; % [N_semiz, length(n_semiz), N_j]
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

[~, SZ_grid_noz]=ndgrid(1:N_a, 1:N_semiz); % for N_z==0 lookup
if N_z>0
    [~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z); % for N_z>0 lookup
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

    % If there is a warm-glow, evaluate the warmglowfn (over aprime), then at the policy
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a, a_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        if N_e==0
            WGofPolicy=reshape(WGmatrix(aprime_idx(:,:,jj)),[N_a,N_shocks]);
        else
            WGofPolicy=reshape(WGmatrix(aprime_idx(:,:,:,jj)),[N_a,N_shocks,N_e]);
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
            V_next(isnan(V_next))=0;
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Reshape V_next as [N_a, N_semiz, N_z (or 1)]
        if N_z==0
            V_next_r=V_next; % [N_a, N_semiz]
            % Step 1: integrate over z' is trivial (no z)
            EV_after_z=V_next_r; % [N_a, N_semiz_to]
            % Step 2: for each d_semiz, integrate over semiz'
            EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)'; % transpose: pi_semiz_J is [N_semiz_from, N_semiz_to]; we want [N_semiz_to, N_semiz_from] so the broadcast contracts semiz_to with V's semiz_to
                EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2); % [N_a, 1, N_semiz_from]
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
            end
        else
            V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
            % Step 1: integrate over z' (does not depend on d_semiz)
            % EV_after_z[anext, semiz_to, z_from] = sum_{z_to} pi_z_J(z_from, z_to, jj) * V_next[anext, semiz_to, z_to]
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

        % Step 3: per-state lookup using aprime_idx and d_semiz_idx
        if N_e==0
            % aprime_idx, d_semiz_idx shape at jj: [N_a, N_shocks]
            aprime_jj=aprime_idx(:,:,jj);
            d2_jj=d_semiz_idx(:,:,jj);
            if N_z==0
                % EVnext_byd2: [N_a, N_semiz, N_dsemiz]; index = aprime + N_a*(sz-1) + N_a*N_semiz*(d2-1)
                aprime_jj_r=reshape(aprime_jj, [N_a, N_semiz]);
                d2_jj_r=reshape(d2_jj, [N_a, N_semiz]);
                linear_idx=aprime_jj_r(:)+N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_jj_r(:)-1);
                EVnextOfPolicy=reshape(EVnext_byd2(linear_idx), [N_a, N_shocks]);
            else
                % EVnext_byd2: [N_a, N_semiz, N_z, N_dsemiz]; index = aprime + N_a*(sz-1) + N_a*N_semiz*(z-1) + N_a*N_semiz*N_z*(d2-1)
                aprime_jj_r=reshape(aprime_jj, [N_a, N_semiz, N_z]);
                d2_jj_r=reshape(d2_jj, [N_a, N_semiz, N_z]);
                linear_idx=aprime_jj_r(:)+N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_jj_r(:)-1);
                EVnextOfPolicy=reshape(EVnext_byd2(linear_idx), [N_a, N_shocks]);
            end
        else
            % With e, lookup is per-e
            if N_z==0
                EVnextOfPolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
                for e_c=1:N_e
                    aprime_e=reshape(aprime_idx(:,:,e_c,jj), [N_a, N_semiz]);
                    d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz]);
                    linear_idx=aprime_e(:)+N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1);
                    EVnextOfPolicy(:,:,e_c)=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz]);
                end
            else
                EVnextOfPolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                for e_c=1:N_e
                    aprime_e=reshape(aprime_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                    d2_e=reshape(d_semiz_idx(:,:,e_c,jj), [N_a, N_semiz, N_z]);
                    linear_idx=aprime_e(:)+N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1);
                    EVnextOfPolicy(:,:,:,e_c)=reshape(EVnext_byd2(linear_idx), [N_a, N_semiz, N_z]);
                end
            end
            EVnextOfPolicy=reshape(EVnextOfPolicy, [N_a, N_shocks, N_e]);
        end

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
