function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetz (vfoptions.experienceassetz>=1),
% semi-exogenous shocks (n_semiz>0), quasi-hyperbolic discounting, AND the grid interpolation layer.
% Combines:
%   - the z aprime machinery + semiz reconstruction + GI 2x2-corner lookup
%     (cf ValueFnFromPolicy_FHorz_ExpAssetz_SemiExo_GI, the structural base)
%   - the quasi-hyperbolic two-value bookkeeping
%     (cf ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo and _QuasiHyperbolic_ExpAssete_SemiExo_GI).
%
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is always built from the recursion-driver value (Vdrive).
%
% Under GI, Policy carries an L2 fine-grid index for a1prime; the per-state lookup is a 2x2 corner
% interpolation (a1 low/up x a2 low/up) combined with the d_semiz-dependent bothz transition. For
% Naive, Vtilde looks up EVnext at Policy's GI indices/a2prime/d_semiz; Valt looks up EVnext at Policyalt's.
%
% aprimeFn = aprimeFn(d_expasset, a2, z, ...) -- depends on the standard Markov z (REQUIRED).

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetz_SemiExo_GI)
% Semiz gridvals + pi_semiz_J
if ~isfield(vfoptions,'pi_semiz_J')
    vfoptions=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,vfoptions,3);
end
% z gridvals
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetz you must define vfoptions.aprimeFn')
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
if N_z==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo_GI: experienceassetz requires N_z>0 (aprimeFn depends on z)')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo_GI: experienceassetz+semiz requires at least one decision variable')
end
l_d=length(n_d);
l_z=length(n_z);

% noa1 case (n_a is scalar -- experience asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to non-GI SemiExo version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);

if isfield(vfoptions,'l_dexperienceassetz')
    l_d2=vfoptions.l_dexperienceassetz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+1:end}};
else
    aprimeFnParamNames={};
end

% Joint shock = [semiz, z] (semiz fastest), matches ValueFnIter convention
n_shocks=[n_semiz,n_z];
N_shocks=N_semiz*N_z;

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetz+GI internally); Policy, and Policyalt if Naive
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
if N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_j]);
    if isNaive
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_shocks, N_j]);
    end
else
    Policy_k=reshape(Policy,[size_first, N_a, N_shocks, N_e, N_j]);
    if isNaive
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_shocks, N_e, N_j]);
    end
end

%% d_semiz_idx: last l_dsemiz components of d (for Policy, and Policyalt if Naive)
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
if isNaive
    if N_e==0
        d_semiz_idx_alt=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        d_semiz_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    for ii=1:l_dsemiz
        comp=shiftdim(Policyalt_k(l_d-l_dsemiz+ii, :, :, :, :),1);
        d_semiz_idx_alt=d_semiz_idx_alt+cumprods_dsemiz(ii)*(comp-1);
    end
end

%% a1prime: midpoint (position l_d+1) + L2 (last). Other a1 components at l_d+2..l_d+l_a1.
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

%% Joint shock gridvals for ReturnFn
joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
for jj=1:N_j
    joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
end

%% The two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
if N_e==0
    Vdrive=zeros(N_a, N_shocks, N_j, 'gpuArray'); % Naive: Valt (at Policyalt).  Soph: Vunderbar (at Policy).
    Vrep  =zeros(N_a, N_shocks, N_j, 'gpuArray'); % Naive: Vtilde (at Policy).   Soph: Vhat (at Policy).
else
    Vdrive=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
    Vrep  =zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
end

[~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj); % [size_first, N_a, N_shocks]
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetz(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, N_semiz, N_z, N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);

    % Step 2: ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
    end

    % Same two objects at Policyalt when Naive
    if isNaive
        if N_e==0
            Policyalt_slice=Policyalt_k(:,:,:,jj);
        else
            Policyalt_slice=reshape(Policyalt_k(:,:,:,:,jj), [size_first, N_a, N_shocks*N_e]);
        end
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssetz(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, N_semiz, N_z, N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
        if N_e==0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        else
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        end
    end

    if jj==N_j
        % Terminal period: no continuation, so both values are just the return at the relevant policy
        if N_e==0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj; % Valt
            else
                Vdrive(:,:,jj)=F_jj;     % Vunderbar
            end
            Vrep(:,:,jj)=F_jj;           % Vtilde / Vhat
        else
            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj;
            else
                Vdrive(:,:,:,jj)=F_jj;
            end
            Vrep(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % EVnext is always built from the recursion-driver value (Vdrive)
        % Step 3a: integrate over e' (if any)
        if N_e==0
            V_next=Vdrive(:,:,jj+1);
        else
            V_next=Vdrive(:,:,:,jj+1);
            V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Step 3b: integrate over z' (markov)
        V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
        EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3);
        EV_after_z(isnan(EV_after_z))=0;
        EV_after_z=reshape(EV_after_z, [N_a, N_semiz, N_z]);

        % Step 3c: for each d_semiz, integrate over semiz' -> EVnext_byd2(a, semiz_from, z_from, d_semiz)
        EVnext_byd2=zeros(N_a, N_semiz, N_z, N_dsemiz, 'gpuArray');
        for d2_c=1:N_dsemiz
            pi_d2c=pi_semiz_J(:,:,d2_c,jj)';
            pi_reshape=reshape(pi_d2c, [1, N_semiz, 1, N_semiz]);
            EVd2c=sum(EV_after_z .* pi_reshape, 2);
            EVd2c(isnan(EVd2c))=0;
            EVnext_byd2(:,:,:,d2_c)=reshape(permute(EVd2c, [1,4,3,2]), [N_a, N_semiz, N_z]);
        end

        % Step 4: per-state 2x2 corner interpolation on EVnext_byd2 at the d_semiz-selected slice.
        % Corners: (a1_low, a2_low), (a1_low, a2_up), (a1_up, a2_low), (a1_up, a2_up)
        if N_e==0
            % --- at Policy ---
            a1l_r=reshape(a1_lower(:,:,jj),[N_a, N_semiz, N_z]);
            a1u_r=reshape(a1_upper(:,:,jj),[N_a, N_semiz, N_z]);
            wa1l_r=reshape(w_a1_lower(:,:,jj),[N_a, N_semiz, N_z]);
            wa1u_r=reshape(w_a1_upper(:,:,jj),[N_a, N_semiz, N_z]);
            a2l=reshape(a2primeIndex,[N_a, N_semiz, N_z]); a2u=a2l+1;
            wa2l=reshape(a2primeProbs,[N_a, N_semiz, N_z]); wa2u=1-wa2l;
            d2_r=reshape(d_semiz_idx(:,:,jj),[N_a, N_semiz, N_z]);
            base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1), [N_a, N_semiz, N_z]);
            lin_LL=a1l_r+N_a1*(a2l-1)+base_off;
            lin_LU=a1l_r+N_a1*(a2u-1)+base_off;
            lin_UL=a1u_r+N_a1*(a2l-1)+base_off;
            lin_UU=a1u_r+N_a1*(a2u-1)+base_off;
            EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_z]);
            EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_z]);
            EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_z]);
            EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_z]);
            EVnext_atP=wa1l_r.*wa2l.*EV_LL + wa1l_r.*wa2u.*EV_LU + wa1u_r.*wa2l.*EV_UL + wa1u_r.*wa2u.*EV_UU;

            if isNaive
                % --- at Policyalt ---
                a1l_a=reshape(a1_lower_alt(:,:,jj),[N_a, N_semiz, N_z]);
                a1u_a=reshape(a1_upper_alt(:,:,jj),[N_a, N_semiz, N_z]);
                wa1l_a=reshape(w_a1_lower_alt(:,:,jj),[N_a, N_semiz, N_z]);
                wa1u_a=reshape(w_a1_upper_alt(:,:,jj),[N_a, N_semiz, N_z]);
                a2l_a=reshape(a2primeIndex_alt,[N_a, N_semiz, N_z]); a2u_a=a2l_a+1;
                wa2l_a=reshape(a2primeProbs_alt,[N_a, N_semiz, N_z]); wa2u_a=1-wa2l_a;
                d2_a=reshape(d_semiz_idx_alt(:,:,jj),[N_a, N_semiz, N_z]);
                base_off_a=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_a(:)-1), [N_a, N_semiz, N_z]);
                lin_LL_a=a1l_a+N_a1*(a2l_a-1)+base_off_a;
                lin_LU_a=a1l_a+N_a1*(a2u_a-1)+base_off_a;
                lin_UL_a=a1u_a+N_a1*(a2l_a-1)+base_off_a;
                lin_UU_a=a1u_a+N_a1*(a2u_a-1)+base_off_a;
                EV_LL_a=reshape(EVnext_byd2(lin_LL_a(:)),[N_a, N_semiz, N_z]);
                EV_LU_a=reshape(EVnext_byd2(lin_LU_a(:)),[N_a, N_semiz, N_z]);
                EV_UL_a=reshape(EVnext_byd2(lin_UL_a(:)),[N_a, N_semiz, N_z]);
                EV_UU_a=reshape(EVnext_byd2(lin_UU_a(:)),[N_a, N_semiz, N_z]);
                EVnext_atPa=wa1l_a.*wa2l_a.*EV_LL_a + wa1l_a.*wa2u_a.*EV_LU_a + wa1u_a.*wa2l_a.*EV_UL_a + wa1u_a.*wa2u_a.*EV_UU_a;
                Vdrive(:,:,jj)=F_alt_jj + beta*reshape(EVnext_atPa,[N_a,N_shocks]); % Valt
            else
                Vdrive(:,:,jj)=F_jj + beta*reshape(EVnext_atP,[N_a,N_shocks]);      % Vunderbar
            end
            Vrep(:,:,jj)=F_jj + beta0beta*reshape(EVnext_atP,[N_a,N_shocks]);       % Vtilde / Vhat
        else
            % --- at Policy ---
            EVnext_atP=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
            for e_c=1:N_e
                block=(e_c-1)*N_shocks + (1:N_shocks);
                a1l_e=reshape(a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                a1u_e=reshape(a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                wa1l_e=reshape(w_a1_lower(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                wa1u_e=reshape(w_a1_upper(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                a2l_e=reshape(a2primeIndex(:,block),[N_a, N_semiz, N_z]); a2u_e=a2l_e+1;
                wa2l_e=reshape(a2primeProbs(:,block),[N_a, N_semiz, N_z]); wa2u_e=1-wa2l_e;
                d2_e=reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
                lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_z]);
                EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_z]);
                EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_z]);
                EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_z]);
                EVnext_atP(:,:,:,e_c)=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
            end

            if isNaive
                % --- at Policyalt ---
                EVnext_atPa=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
                for e_c=1:N_e
                    block=(e_c-1)*N_shocks + (1:N_shocks);
                    a1l_e=reshape(a1_lower_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a1u_e=reshape(a1_upper_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    wa1l_e=reshape(w_a1_lower_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    wa1u_e=reshape(w_a1_upper_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a2l_e=reshape(a2primeIndex_alt(:,block),[N_a, N_semiz, N_z]); a2u_e=a2l_e+1;
                    wa2l_e=reshape(a2primeProbs_alt(:,block),[N_a, N_semiz, N_z]); wa2u_e=1-wa2l_e;
                    d2_e=reshape(d_semiz_idx_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
                    lin_LL=a1l_e+N_a1*(a2l_e-1)+base_off;
                    lin_LU=a1l_e+N_a1*(a2u_e-1)+base_off;
                    lin_UL=a1u_e+N_a1*(a2l_e-1)+base_off;
                    lin_UU=a1u_e+N_a1*(a2u_e-1)+base_off;
                    EV_LL=reshape(EVnext_byd2(lin_LL(:)),[N_a, N_semiz, N_z]);
                    EV_LU=reshape(EVnext_byd2(lin_LU(:)),[N_a, N_semiz, N_z]);
                    EV_UL=reshape(EVnext_byd2(lin_UL(:)),[N_a, N_semiz, N_z]);
                    EV_UU=reshape(EVnext_byd2(lin_UU(:)),[N_a, N_semiz, N_z]);
                    EVnext_atPa(:,:,:,e_c)=wa1l_e.*wa2l_e.*EV_LL + wa1l_e.*wa2u_e.*EV_LU + wa1u_e.*wa2l_e.*EV_UL + wa1u_e.*wa2u_e.*EV_UU;
                end
                Vdrive(:,:,:,jj)=F_alt_jj + beta*reshape(EVnext_atPa,[N_a,N_shocks,N_e]); % Valt
            else
                Vdrive(:,:,:,jj)=F_jj + beta*reshape(EVnext_atP,[N_a,N_shocks,N_e]);      % Vunderbar
            end
            Vrep(:,:,:,jj)=F_jj + beta0beta*reshape(EVnext_atP,[N_a,N_shocks,N_e]);       % Vtilde / Vhat
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
if N_e==0
    V   =reshape(Vrep,   [n_a, n_semiz, n_z, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, n_z, N_j]);
else
    V   =reshape(Vrep,   [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_semiz, n_z, vfoptions.n_e, N_j]);
end


end
