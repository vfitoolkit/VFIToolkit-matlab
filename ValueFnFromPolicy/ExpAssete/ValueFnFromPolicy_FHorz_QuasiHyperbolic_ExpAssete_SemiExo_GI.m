function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Quasi-hyperbolic ValueFnFromPolicy for experienceassete (a2prime=aprimeFn(d,a2,e)) WITH a
% semi-exogenous state (semiz) AND the grid interpolation layer (vfoptions.gridinterplayer==1).
% Combines:
%   - the e aprime machinery + semiz reconstruction + GI 2x2-corner lookup
%     (cf ValueFnFromPolicy_FHorz_ExpAsseteSemiExo_GI, the structural base)
%   - the QH dual-value (Naive/Sophisticated) idiom with d_semiz indirection
%     (cf ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo and _QuasiHyperbolic_ExpAssetze_SemiExo_GI).
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is always built from the recursion-driver value (Vdrive).
%
% semiz combines with the Markov z into bothz=[semiz,z] (semiz fastest); e is a separate iid
% start-of-period shock and aprimeFn depends on e ONLY (not on z, not on semiz). Unlike the ze
% variant, z is OPTIONAL here: with no z the joint shock is just semiz.
% Under GI, Policy carries an L2 fine-grid index for a1prime; the per-state lookup is a 2x2 corner
% interpolation (a1 low/up x a2 low/up) combined with the d_semiz-dependent bothz transition and
% e'-integration. For Naive, Vtilde looks up EVnext at Policy's GI indices/a2prime/d_semiz;
% Valt looks up EVnext at Policyalt's.
%
% Convention on d ordering with semiz: d = [...other d..., d_expasset, d_semiz]. d_semiz is the
% last l_dsemiz components; d_expasset is the l_d2 components immediately before them.

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAsseteSemiExo_GI)
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
% z gridvals
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassete you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

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
if N_e==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo_GI: experienceassete requires N_e>0 (aprimeFn depends on e)')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo_GI: experienceassete+semiz requires at least one decision variable')
end
l_d=length(n_d);
l_e=length(vfoptions.n_e);

% z is optional for experienceassete; N_zeff=1 collapses the z dimension without special-casing every reshape
N_zeff=max(N_z,1);

% noa1 case (n_a is scalar -- experience asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete_SemiExo(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);

if isfield(vfoptions,'l_dexperienceassete')
    l_d2=vfoptions.l_dexperienceassete;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);

% aprimeFnParamNames: leading inputs are (d_expasset..., a2, e...)   -- no z, unlike the ze variant
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_e)
    aprimeFnParamNames={temp{l_d2+l_a2+l_e+1:end}};
else
    aprimeFnParamNames={};
end

% Joint Markov-like shock = [semiz, z] (semiz fastest); e separate. With no z it is just semiz.
if N_z==0
    n_shocks=n_semiz;
else
    n_shocks=[n_semiz,n_z];
end
N_shocks=N_semiz*N_zeff;

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassete+GI internally); Policy, and Policyalt if Naive
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
end

%% Strip trailing L2flag channel from Policy (and Policyalt) if present
size_first=l_d+l_a1+1; % under GI: d, a1mid, L2 -> l_d + l_a1 + 1 channels
if size(Policy,1) > size_first
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1), prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:size_first,:), [size_first, tempsize(2:end)]);
end
if isNaive
    if size(Policyalt,1) > size_first
        tempsize=size(Policyalt);
        Policyalt=reshape(Policyalt,[tempsize(1), prod(tempsize)/tempsize(1)]);
        Policyalt=reshape(Policyalt(1:size_first,:), [size_first, tempsize(2:end)]);
    end
end

%% Reshape Policy (and Policyalt) to canonical Kron form
Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_e, N_j]);
if isNaive
    Policyalt_k=reshape(Policyalt,[size_first, N_a, N_shocks, N_e, N_j]);
end

%% d_semiz_idx: last l_dsemiz components of d (for Policy, and Policyalt if Naive)
d_semiz_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
cumprods_dsemiz=[1, cumprod(n_dsemiz(1:end-1))];
for ii=1:l_dsemiz
    comp=shiftdim(Policy_k(l_d-l_dsemiz+ii, :, :, :, :),1);
    d_semiz_idx=d_semiz_idx+cumprods_dsemiz(ii)*(comp-1);
end
if isNaive
    d_semiz_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    for ii=1:l_dsemiz
        comp=shiftdim(Policyalt_k(l_d-l_dsemiz+ii, :, :, :, :),1);
        d_semiz_idx_alt=d_semiz_idx_alt+cumprods_dsemiz(ii)*(comp-1);
    end
end

%% a1prime GI indices: midpoint (position l_d+1) + L2 (last). Other a1 components at l_d+2..l_d+l_a1.
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:,:),1);
L2    =shiftdim(Policy_k(l_d+l_a1+1,:,:,:,:),1);
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

%% Same GI-index extraction for Policyalt when Naive
if isNaive
    a1_mid_alt=shiftdim(Policyalt_k(l_d+1,:,:,:,:),1);
    L2_alt    =shiftdim(Policyalt_k(l_d+l_a1+1,:,:,:,:),1);
    w_a1_upper_alt=(L2_alt-1)/(n2short+1);
    w_a1_lower_alt=1-w_a1_upper_alt;
    a1_lower_alt=a1_mid_alt;
    for ii=2:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii,:,:,:,:),1);
        a1_lower_alt=a1_lower_alt+cumprods_a1(ii)*(comp-1);
    end
    a1_upper_alt=a1_lower_alt+1;
    a1_top_clamp_alt=(a1_mid_alt>=n_a1(1));
    a1_upper_alt(a1_top_clamp_alt)=a1_lower_alt(a1_top_clamp_alt);
end

%% Joint Markov-like shock gridvals for ReturnFn (semiz, z)
if N_z==0
    joint_gridvals_J=semiz_gridvals_J;
else
    joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
    for jj=1:N_j
        joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
    end
end

%% The two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
Vdrive=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray'); % Naive: Valt (at Policyalt).  Soph: Vunderbar (at Policy).
Vrep  =zeros(N_a, N_shocks, N_e, N_j, 'gpuArray'); % Naive: Vtilde (at Policy).   Soph: Vhat (at Policy).

[~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_zeff);

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % a2prime interpolation (aprimeFn depends on e only), at Policy
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    Policy_slice=reshape(Policy_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssete(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, vfoptions.n_e, N_semiz, N_z, N_e, d_grid, a2_grid, vfoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % shape [N_a, N_semiz*N_zeff*N_e]

    % ReturnFn at Policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);

    if isNaive
        Policyalt_slice=reshape(Policyalt_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssete(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, vfoptions.n_e, N_semiz, N_z, N_e, d_grid, a2_grid, vfoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
        F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
    end

    if jj==N_j
        if isNaive
            Vdrive(:,:,:,jj)=F_alt_jj; % Valt
            Vrep(:,:,:,jj)  =F_jj;     % Vtilde
        else
            Vdrive(:,:,:,jj)=F_jj; % Vunderbar
            Vrep(:,:,:,jj)  =F_jj; % Vhat
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % EVnext from the recursion-driver value (Vdrive):
        % Step a: integrate next-period value over e' (iid)
        V_next=Vdrive(:,:,:,jj+1);
        V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
        V_next=reshape(V_next, [N_a, N_shocks]);

        % Step b: integrate over z' (markov, does not depend on d_semiz). With no z this is a no-op.
        V_next_r=reshape(V_next, [N_a, N_semiz, N_zeff]);
        if N_z==0
            EV_after_z=V_next_r;
        else
            EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3);
            EV_after_z(isnan(EV_after_z))=0;
            EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_zeff]);
        end

        % Step c: for each d_semiz, integrate over semiz' -> EVnext_byd2(a, semiz_from, z_from, d_semiz)
        EVnext_byd2=zeros(N_a, N_semiz, N_zeff, N_dsemiz, 'gpuArray');
        for d2_c=1:N_dsemiz
            pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
            pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
            EVd2c=sum(EV_after_z .* pi_reshape, 2);
            EVd2c(isnan(EVd2c))=0;
            EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_zeff]);
        end

        % Step d: per-state 2x2 corner interpolation at Policy (a1 low/up x a2 low/up) + d_semiz indirection, per-e
        EVnext_atP=zeros(N_a, N_semiz, N_zeff, N_e, 'gpuArray');
        for e_c=1:N_e
            block=(e_c-1)*N_shocks + (1:N_shocks);
            a1l_e=reshape(a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
            a1u_e=reshape(a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
            wa1l_e=reshape(w_a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
            wa1u_e=reshape(w_a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
            a2l_e=reshape(a2primeIndex(:,block),[N_a, N_semiz, N_zeff]); a2u_e=a2l_e+1;
            wa2l_e=reshape(a2primeProbs(:,block),[N_a, N_semiz, N_zeff]); wa2u_e=1-wa2l_e;
            d2_e=reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
            base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_zeff*(d2_e(:)-1), [N_a, N_semiz, N_zeff]);
            lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
            lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
            lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
            lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
            EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_zeff]);
            EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_zeff]);
            EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_zeff]);
            EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_zeff]);
            EVnext_atP(:,:,:,e_c)=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
        end

        if isNaive
            % Naive: Valt looks up EVnext at Policyalt's GI indices/a2prime/d_semiz
            EVnext_atPa=zeros(N_a, N_semiz, N_zeff, N_e, 'gpuArray');
            for e_c=1:N_e
                block=(e_c-1)*N_shocks + (1:N_shocks);
                a1l_e=reshape(a1_lower_alt(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
                a1u_e=reshape(a1_upper_alt(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
                wa1l_e=reshape(w_a1_lower_alt(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
                wa1u_e=reshape(w_a1_upper_alt(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
                a2l_e=reshape(a2primeIndex_alt(:,block),[N_a, N_semiz, N_zeff]); a2u_e=a2l_e+1;
                wa2l_e=reshape(a2primeProbs_alt(:,block),[N_a, N_semiz, N_zeff]); wa2u_e=1-wa2l_e;
                d2_e=reshape(d_semiz_idx_alt(:,:,e_c,jj),[N_a, N_semiz, N_zeff]);
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_zeff*(d2_e(:)-1), [N_a, N_semiz, N_zeff]);
                lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_zeff]);
                EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_zeff]);
                EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_zeff]);
                EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_zeff]);
                EVnext_atPa(:,:,:,e_c)=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
            end
            Vdrive(:,:,:,jj)=F_alt_jj + beta    *reshape(EVnext_atPa,[N_a,N_shocks,N_e]); % Valt (drives recursion)
            Vrep(:,:,:,jj)  =F_jj     + beta0beta*reshape(EVnext_atP, [N_a,N_shocks,N_e]); % Vtilde (reported)
        else
            Vdrive(:,:,:,jj)=F_jj + beta    *reshape(EVnext_atP,[N_a,N_shocks,N_e]); % Vunderbar (drives recursion)
            Vrep(:,:,:,jj)  =F_jj + beta0beta*reshape(EVnext_atP,[N_a,N_shocks,N_e]); % Vhat (reported)
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
if N_z==0
    V   =reshape(Vrep,   [n_a, n_semiz, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, vfoptions.n_e, N_j]);
else
    V   =reshape(Vrep,   [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end

end
