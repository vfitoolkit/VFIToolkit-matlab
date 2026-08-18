function V=ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_SemiExo_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8)
% Compute V from a given Policy when the model uses Epstein-Zin preferences
% AND has riskyasset (vfoptions.riskyasset==1) AND semi-exogenous shocks
% (n_semiz>0) AND the grid interpolation layer (vfoptions.gridinterplayer==1).
% Mirrors ValueFnFromPolicy_FHorz_RiskyAsset_SemiExo_GI (the vNM version) with
% the Epstein-Zin transform chain of ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_SemiExo:
% V' is transformed by ^ezc5 BEFORE all the expectations (one joint
% certainty-equivalent over (u,semizprime,zprime,eprime); e' is summed first,
% then z' is collapsed, then the d_semiz-dependent semiz' transition is
% collapsed per d_semiz choice, and finally the u-lottery over a2prime is
% applied via the per-state lookup on the collapsed object), with the
% certainty-equivalent power ^ezc6 applied pointwise after the per-state
% lookup. Under GI, Policy carries an L2 fine-grid index for a1prime, so the
% per-state lookup is a 2x2 corner interpolation in (a1prime,a2prime) applied
% to the TRANSFORMED EV (the established EZ GI convention, as in
% ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_GI), then summed over u with
% pi_u; every probability-weighted sum gets an isnan clear (0*(-Inf) gives NaN).
% The warm-glow of bequests is over the risky asset a2prime (as in the EZ
% riskyasset VFI raws), so it is unaffected by the GI layer (which refines
% a1prime only): it is transformed by ^ezc5 and averaged over the policy's
% a2prime u-lottery before being folded into the sj/ezc8 structure.
% The ezc1-ezc8/sj/warmglow preamble is done by the caller
% (ValueFnFromPolicy_FHorz_EpsteinZin) and passed through.
%
% riskyasset: a2prime = aprimeFn(d2,d3, u) -- depends on iid (between-period) shock u, NOT on current a2.
% u does NOT enter Policy; pi_u integrates u out when computing E[V'|policy].
% The d split is set by vfoptions.refine_d ([d1,d2,d3,d4]): d1 ReturnFn only,
% d2 aprimeFn only, d3 both, d4 the semiz decision.

%% Setup
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
% z gridvals
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use riskyasset you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;
if ~isfield(vfoptions,'n_u'),    error('To use riskyasset you must define vfoptions.n_u'),    end
if ~isfield(vfoptions,'u_grid'), error('To use riskyasset you must define vfoptions.u_grid'), end
if ~isfield(vfoptions,'pi_u'),   error('To use riskyasset you must define vfoptions.pi_u'),   end
n_u=vfoptions.n_u;
u_grid=gpuArray(vfoptions.u_grid);
pi_u=gpuArray(vfoptions.pi_u);
N_u=prod(n_u);
l_u=length(n_u);

n_semiz=vfoptions.n_semiz;
N_semiz=prod(n_semiz);

if ~isfield(vfoptions,'refine_d')
    error('ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_SemiExo_GI requires vfoptions.refine_d (with a 4th entry for the semiz decision)')
end
refine_d=vfoptions.refine_d;
l_dsemiz=refine_d(4); % the semiz decision (d4) is the last refine_d(4) components of d
n_dsemiz=n_d(end-l_dsemiz+1:end);
N_dsemiz=prod(n_dsemiz);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_SemiExo_GI: riskyasset+semiz requires at least one decision variable')
end
l_d=length(n_d);

% noa1 case (n_a is scalar -- risky asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to the non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    V=ValueFnFromPolicy_FHorz_EpsteinZin_RiskyAsset_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8);
    return
end

% Split a into a1 (standard) and a2 (risky asset).
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
l_a1=length(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);

% Which d feed the aprimeFn (d2,d3), under the 'refine' split:
whichisdforriskyasset=(refine_d(1)+1):sum(refine_d(1:3));
l_drisky=length(whichisdforriskyasset);
% Which d feed the ReturnFn (d1,d3,d4) -- exclude d2 (aprimeFn-only). Order matches the VFI's
% ReturnMatrix ([n_d1,n_d3,n_d4] and a1).
d1rows=1:refine_d(1);
d3rows=(refine_d(1)+refine_d(2)+1):sum(refine_d(1:3));
d4rows=(sum(refine_d(1:3))+1):sum(refine_d(1:4));

% aprimeFnParamNames: riskyasset aprimeFn is (d2,d3, u) -- NO current a2
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_drisky+l_u)
    aprimeFnParamNames={temp{l_drisky+l_u+1:end}};
else
    aprimeFnParamNames={};
end

% Treat (semiz, z) as joint shock for PolicyInd2Val and ReturnFn evaluation
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*max(N_z,1);

n2short=vfoptions.ngridinterp; % number of interior fine-grid points between coarse points

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles riskyasset+GI internally -- drops a2prime, auto-adds n_semiz and n_e)
% ReturnFn takes (d1,d3,d4,a1prime,...): d2 (aprimeFn-only) is NOT an input, so keep only those rows.
returnrows=[d1rows, d3rows, d4rows, (l_d+1):(l_d+l_a1)];
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
PolicyValues=PolicyValues(returnrows,:,:,:,:); % drop d2 (not in the ReturnFn)
l_daprime=size(PolicyValues,1); % = n_d1 + n_d3 + n_d4 + l_a1
% PolicyValues shape:
% - N_e==0: [l_daprime, N_a, N_shocks, N_j]
% - N_e>0:  [l_daprime, N_a, N_shocks*N_e, N_j]
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);

%% Strip trailing L2flag channel if present, then reshape Policy to canonical Kron form
% Under GI, Policy rows are [d..., a1prime_low (l_a1 rows), L2] (+ optional trailing L2flag).
size_first=l_d+l_a1+1;
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end
if N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_e, N_j]);
end

%% d_semiz_idx: last l_dsemiz components of d
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

%% a1prime lower fine-grid index and GI weights (positions l_d+1 .. l_d+l_a1, plus L2 at l_d+l_a1+1)
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:,:),1);            % lower index in first a1 dimension
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:,:),1);          % L2 fine-grid index
w_a1_upper=(L2-1)/(n2short+1);
w_a1_lower=1-w_a1_upper;
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_lower=a1_mid;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_mid>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);
% shapes: [N_a, N_shocks, N_j] (N_e==0) or [N_a, N_shocks, N_e, N_j] (N_e>0)

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

    % Step 1: a2primeIndex, a2primeProbs at this age -- helper adds the u dim
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj); % [size_first, N_a, N_shocks]
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyRiskyAsset(Policy_slice, aprimeFn, whichisdforriskyasset, n_d, n_a1, n_a2, N_shocks*max(N_e,1), n_u, d_grid, a2_grid, u_grid, aprimeFnParamsVec);
    % Shapes (helper passes N_z arg=N_shocks*max(N_e,1)>0 so the N_z>0 branch is always taken):
    %   N_e==0: [N_a, N_shocks,    N_u]
    %   N_e>0:  [N_a, N_shocks*N_e, N_u]

    % Step 2: ReturnFn at policy (u does not enter Return)
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

    % If there is a warm-glow, evaluate the warmglowfn over a2prime (the bequest is the risky
    % asset, mirrors the EZ riskyasset VFI raws; a2prime has no GI fine grid, so this block is
    % identical to the non-GI version), then the policy's a2prime u-lottery
    if warmglow==1
        WGParamsVec=CreateVectorFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames,jj);
        WGmatrixraw=CreateWarmGlowFnMatrix_Case1_Disc_Par2(vfoptions.WarmGlowBequestsFn, n_a2, a2_grid, WGParamsVec);
        WGmatrix=WGmatrixraw;
        WGmatrix(isfinite(WGmatrixraw))=(ezc4*WGmatrixraw(isfinite(WGmatrixraw))).^ezc5(jj);
        WGmatrix(WGmatrixraw==0)=0; % otherwise zero to negative power is set to infinity
        % u-lottery of the transformed warm-glow at the policy's a2prime
        WGlower=WGmatrix(a2primeIndex);   % same shape as a2primeIndex
        WGupper=WGmatrix(a2primeIndex+1);
        wgPrb=a2primeProbs;
        wgPrb(WGlower==WGupper)=0; % skipinterp
        WGlott=wgPrb.*WGlower+(1-wgPrb).*WGupper;
        WGofPolicy=sum(WGlott .* shiftdim(pi_u,-2), 3); % sum over u -> [N_a, N_shocks] or [N_a, N_shocks*N_e]
        WGofPolicy(isnan(WGofPolicy))=0;
        if N_e>0
            WGofPolicy=reshape(WGofPolicy,[N_a,N_shocks,N_e]);
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
        % Step 3 (EZ): transform V' by ^ezc5 BEFORE all the sums (one joint certainty-equivalent
        % over (u,semizprime,zprime,eprime)), then collapse e' (iid), then z' (markov), then the
        % d_semiz-dependent semiz' transition per d_semiz choice -- every expectation acts on the
        % transformed object; each probability-weighted sum gets an isnan clear (0*(-Inf) gives NaN).
        if N_e==0
            V_nextpre=V(:,:,jj+1); % [N_a, N_shocks]
        else
            V_nextpre=V(:,:,:,jj+1); % [N_a, N_shocks, N_e]
        end
        temp=V_nextpre;
        temp(isfinite(V_nextpre))=(ezc4*V_nextpre(isfinite(V_nextpre))).^ezc5(jj);
        temp(V_nextpre==0)=0;

        % Step 3a: integrate the transformed V' over e' (if any)
        if N_e==0
            V_next=temp;
        else
            V_next=sum(temp .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
            V_next(isnan(V_next))=0; % -Inf times zero e'-probability
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Step 3b: integrate over z' (markov, does not depend on d_semiz)
        if N_z==0
            EV_after_z=V_next;
        else
            V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
            EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3); % [N_a, N_semiz_to, 1, N_z_from]
            EV_after_z(isnan(EV_after_z))=0;
            EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);
        end

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2
        if N_z==0
            EVnext_byd2=zeros(N_a, N_semiz, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
                EVd2c=sum(EV_after_z .* shiftdim(pi_d2c, -1), 2);
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,d2_c)=reshape(EVd2c, [N_a, N_semiz]);
            end
        else
            EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
            for d2_c=1:N_dsemiz
                pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
                pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
                EVd2c=sum(EV_after_z .* pi_reshape, 2);
                EVd2c(isnan(EVd2c))=0;
                EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
            end
        end

        % Step 4: per-state lookup on the transformed-and-collapsed object. EVnext_byd2 is
        % already pre-collapsed over (z', semiz') per d_semiz choice. Under GI, a1prime has two
        % fine-grid corners (a1_lower/a1_upper with weights w_a1_lower/w_a1_upper) and a2prime
        % has two corners (a2primeIndex/+1 with a2primeProbs), so the lookup is a 2x2 corner
        % interpolation in (a1,a2) applied to the TRANSFORMED EV (the established EZ GI
        % convention), then summed over u with pi_u, with an isnan clear after the u-sum
        % (zero corner weights times -Inf next-states). The ^ezc6 comes after, via temp4.
        if N_e==0
            d2_jj=d_semiz_idx(:,:,jj); % [N_a, N_shocks]
            if N_z==0
                d2_r =reshape(d2_jj,[N_a, N_semiz]);
                a1l =reshape(a1_lower(:,:,jj),[N_a, N_semiz]);
                a1u =reshape(a1_upper(:,:,jj),[N_a, N_semiz]);
                wa1l=reshape(w_a1_lower(:,:,jj),[N_a, N_semiz]);
                wa1u=reshape(w_a1_upper(:,:,jj),[N_a, N_semiz]);
                a2l=a2primeIndex; a2u=a2primeIndex+1;   % [N_a, N_semiz, N_u]
                wa2l=a2primeProbs; wa2u=1-a2primeProbs;
                base_off=reshape(N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_r(:)-1), [N_a, N_semiz]);
                lin_LL=a1l+N_a1*(a2l-1)+base_off;
                lin_LU=a1l+N_a1*(a2u-1)+base_off;
                lin_UL=a1u+N_a1*(a2l-1)+base_off;
                lin_UU=a1u+N_a1*(a2u-1)+base_off;
                EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_u]);
                EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_u]);
                EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_u]);
                EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_u]);
                per_u=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
                EVnextOfPolicy=sum(per_u .* shiftdim(pi_u,-2), 3); % [N_a, N_semiz]
                EVnextOfPolicy(isnan(EVnextOfPolicy))=0;
            else
                d2_r =reshape(d2_jj,[N_a, N_semiz, N_z]);
                a1l =reshape(a1_lower(:,:,jj),[N_a, N_semiz, N_z]);
                a1u =reshape(a1_upper(:,:,jj),[N_a, N_semiz, N_z]);
                wa1l=reshape(w_a1_lower(:,:,jj),[N_a, N_semiz, N_z]);
                wa1u=reshape(w_a1_upper(:,:,jj),[N_a, N_semiz, N_z]);
                a2l=reshape(a2primeIndex,[N_a, N_semiz, N_z, N_u]); a2u=a2l+1;
                wa2l=reshape(a2primeProbs,[N_a, N_semiz, N_z, N_u]); wa2u=1-wa2l;
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1), [N_a, N_semiz, N_z]);
                lin_LL=a1l+N_a1*(a2l-1)+base_off;
                lin_LU=a1l+N_a1*(a2u-1)+base_off;
                lin_UL=a1u+N_a1*(a2l-1)+base_off;
                lin_UU=a1u+N_a1*(a2u-1)+base_off;
                EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_z, N_u]);
                EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_z, N_u]);
                EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_z, N_u]);
                EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_z, N_u]);
                per_u=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
                EVnextOfPolicy=sum(per_u .* shiftdim(pi_u,-3), 4); % [N_a, N_semiz, N_z]
                EVnextOfPolicy(isnan(EVnextOfPolicy))=0;
            end
            EVnextOfPolicy=reshape(EVnextOfPolicy, [N_a, N_shocks]);
        else
            if N_z==0
                EVnextOfPolicy=zeros(N_a, N_semiz, N_e, 'gpuArray');
                for e_c=1:N_e
                    block=(e_c-1)*N_shocks + (1:N_shocks);
                    d2_e =reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz]);
                    a1l_e =reshape(a1_lower(:,:,e_c,jj),[N_a, N_semiz]);
                    a1u_e =reshape(a1_upper(:,:,e_c,jj),[N_a, N_semiz]);
                    wa1l_e=reshape(w_a1_lower(:,:,e_c,jj),[N_a, N_semiz]);
                    wa1u_e=reshape(w_a1_upper(:,:,e_c,jj),[N_a, N_semiz]);
                    a2l_e=reshape(a2primeIndex(:,block,:),[N_a, N_semiz, N_u]); a2u_e=a2l_e+1;
                    wa2l_e=reshape(a2primeProbs(:,block,:),[N_a, N_semiz, N_u]); wa2u_e=1-wa2l_e;
                    base_off=reshape(N_a*(SZ_grid_noz(:)-1)+N_a*N_semiz*(d2_e(:)-1), [N_a, N_semiz]);
                    lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                    lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                    lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                    lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                    EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_u]);
                    EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_u]);
                    EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_u]);
                    EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_u]);
                    per_u=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
                    EV_summed=sum(per_u .* shiftdim(pi_u,-2), 3);
                    EV_summed(isnan(EV_summed))=0;
                    EVnextOfPolicy(:,:,e_c)=EV_summed;
                end
            else
                EVnextOfPolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                for e_c=1:N_e
                    block=(e_c-1)*N_shocks + (1:N_shocks);
                    d2_e =reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a1l_e =reshape(a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a1u_e =reshape(a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    wa1l_e=reshape(w_a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    wa1u_e=reshape(w_a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a2l_e=reshape(a2primeIndex(:,block,:),[N_a, N_semiz, N_z, N_u]); a2u_e=a2l_e+1;
                    wa2l_e=reshape(a2primeProbs(:,block,:),[N_a, N_semiz, N_z, N_u]); wa2u_e=1-wa2l_e;
                    base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
                    lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                    lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                    lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                    lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                    EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_z, N_u]);
                    EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_z, N_u]);
                    EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_z, N_u]);
                    EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_z, N_u]);
                    per_u=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
                    EV_summed=sum(per_u .* shiftdim(pi_u,-3), 4);
                    EV_summed(isnan(EV_summed))=0;
                    EVnextOfPolicy(:,:,:,e_c)=EV_summed;
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
