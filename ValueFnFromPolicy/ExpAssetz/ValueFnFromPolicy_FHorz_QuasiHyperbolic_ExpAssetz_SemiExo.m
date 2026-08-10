function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Quasi-hyperbolic ValueFnFromPolicy for experienceassetz WITH a semi-exogenous state (semiz).
% Combines:
%   - ExpAssetz+SemiExo: a2 is the experience asset; a2prime=aprimeFn(d_expasset,a2,z,...) with z the
%     standard Markov shock (REQUIRED, aprimeFn does not depend on semiz). semiz has transition pi_semiz
%     that depends on the policy-chosen d_semiz (the last l_dsemiz components of d). semiz and z are joined
%     into bothz (N_bothz=N_semiz*N_z, semiz fastest). Base structure: ValueFnFromPolicy_FHorz_ExpAssetz_SemiExo.
%   - QH dual-value overlay (Naive/Sophisticated), cf ValueFnFromPolicy_FHorz_QuasiHyperbolic_SemiExo and
%     ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz:
%       Naive:         V=Vrep=Vtilde (beta0*beta at Policy);  Valt=Vdrive = exponential value
%                      (beta at Policyalt, drives recursion). Requires Policyalt.
%       Sophisticated: V=Vrep=Vhat   (beta0*beta at Policy);  Valt=Vdrive=Vunderbar
%                      (beta at Policy, drives recursion).
% The continuation (EVnext) is always built from the recursion-driver value Vdrive. For Naive, Vtilde looks
% up EVnext under Policy's d_semiz (and Policy's a2prime); Vdrive under Policyalt's d_semiz (and a2prime).
%
% Convention on d ordering with semiz: d = [...other d..., d_expasset, d_semiz]. d_semiz is the last
% l_dsemiz components; d_expasset is the l_d2 components immediately before them.

%% Scope guard
if vfoptions.gridinterplayer==1
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo: gridinterplayer not yet implemented for experienceassetz+SemiExo QH')
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetz_SemiExo)
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
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo: experienceassetz requires N_z>0 (aprimeFn depends on z)')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_SemiExo: experienceassetz+semiz requires at least one decision variable')
end
l_d=length(n_d);
l_z=length(n_z);

% Split a into a1 (standard) and a2 (experience asset)
if isscalar(n_a)
    % noa1: the experience asset is the only endogenous state
    n_a1=0;
    N_a1=1; % so aprime_low=a1p+N_a1*(a2pIdx-1) reduces to the a2primeIndex lookup (a1p stays 1)
    l_a1=0; % Policy contains only the d channels
else
    n_a1=n_a(1:end-1);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a2=length(n_a2);

% Which d drives the experience asset. With semiz, d ordering is [...other, d_expasset, d_semiz];
% the last l_dsemiz are for semiz, the l_d2 immediately before them drive the expasset.
if isfield(vfoptions,'l_dexperienceassetz')
    l_d2=vfoptions.l_dexperienceassetz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_dsemiz-l_d2+1):(l_d-l_dsemiz);

% aprimeFnParamNames: leading inputs are (d_expasset..., a2, z...)
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+1:end}};
else
    aprimeFnParamNames={};
end

% Joint shock = [semiz, z] (semiz fastest), matches ValueFnIter_FHorz_ExpAssetzSemiExo convention
n_shocks=[n_semiz,n_z];
N_shocks=N_semiz*N_z;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);
semiz_gridvals_J=vfoptions.semiz_gridvals_J;
pi_semiz_J=vfoptions.pi_semiz_J;

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetz internally; auto-adds n_semiz and n_e)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
% PolicyValues shape:
% - N_e==0: [l_daprime, N_a, N_shocks, N_j]
% - N_e>0:  [l_daprime, N_a, N_shocks*N_e, N_j]
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);

% Reshape Policy to canonical Kron form
if N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_shocks, N_e, N_j]);
end

% d_semiz_idx (Policy): joint index into n_dsemiz from the last l_dsemiz components of d
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

% a1prime_idx (Policy): joint index across N_a1 dims (positions l_d+1 .. l_d+l_a1)
if N_e==0
    a1prime_idx=ones(N_a,N_shocks,N_j,'gpuArray');
else
    a1prime_idx=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% PolicyaltValues + per-state indices (Naive only, recursion driver)
if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
    if N_e==0
        Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_shocks, N_j]);
        d_semiz_idx_alt=ones(N_a,N_shocks,N_j,'gpuArray');
        a1prime_idx_alt=ones(N_a,N_shocks,N_j,'gpuArray');
    else
        Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_shocks, N_e, N_j]);
        d_semiz_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
        a1prime_idx_alt=ones(N_a,N_shocks,N_e,N_j,'gpuArray');
    end
    for ii=1:l_dsemiz
        comp=shiftdim(Policyalt_k(l_d-l_dsemiz+ii, :, :, :, :),1);
        d_semiz_idx_alt=d_semiz_idx_alt+cumprods_dsemiz(ii)*(comp-1);
    end
    for ii=1:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii, :, :, :, :),1);
        a1prime_idx_alt=a1prime_idx_alt+cumprods_a1(ii)*(comp-1);
    end
end

%% Joint shock gridvals for ReturnFn
joint_gridvals_J=zeros(N_shocks, length(n_semiz)+length(n_z), N_j, 'gpuArray');
for jj=1:N_j
    joint_gridvals_J(:,:,jj)=[repmat(semiz_gridvals_J(:,:,jj),N_z,1), repelem(z_gridvals_J(:,:,jj),N_semiz,1)];
end

%% Two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
if N_e==0
    Vdrive=zeros(N_a, N_shocks, N_j, 'gpuArray');
    Vrep=zeros(N_a, N_shocks, N_j, 'gpuArray');
else
    Vdrive=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
    Vrep=zeros(N_a, N_shocks, N_e, N_j, 'gpuArray');
end

[~, SZ_grid, Z_grid]=ndgrid(1:N_a, 1:N_semiz, 1:N_z);

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age, at Policy (and Policyalt if Naive)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_e==0
        Policy_slice=Policy_k(:,:,:,jj); % [l_d+l_a1, N_a, N_shocks]
    else
        Policy_slice=reshape(Policy_k(:,:,:,:,jj), [l_d+l_a1, N_a, N_shocks*N_e]);
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetz(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, N_semiz, N_z, N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
    % Shapes: N_e==0: [N_a, N_shocks];  N_e>0: [N_a, N_shocks*N_e]
    if isNaive
        if N_e==0
            Policyalt_slice=Policyalt_k(:,:,:,jj);
        else
            Policyalt_slice=reshape(Policyalt_k(:,:,:,:,jj), [l_d+l_a1, N_a, N_shocks*N_e]);
        end
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssetz(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, N_semiz, N_z, N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
    end

    % Step 2: ReturnFn at Policy (and Policyalt if Naive)
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        if isNaive
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_shocks, a_gridvals, joint_gridvals_J(:,:,jj));
        end
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        if isNaive
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_shocks,vfoptions.n_e], a_gridvals, [repmat(joint_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_shocks,1)]), [N_a, N_shocks, N_e]);
        end
    end

    if jj==N_j
        if N_e==0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj; Vrep(:,:,jj)=F_jj;
            else
                Vdrive(:,:,jj)=F_jj; Vrep(:,:,jj)=F_jj;
            end
        else
            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj; Vrep(:,:,:,jj)=F_jj;
            else
                Vdrive(:,:,:,jj)=F_jj; Vrep(:,:,:,jj)=F_jj;
            end
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % Step 3a: recursion-driver next-period value, integrate out e' (iid, if present)
        if N_e==0
            V_next=Vdrive(:,:,jj+1);
        else
            V_next=Vdrive(:,:,:,jj+1);
            V_next=sum(V_next .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
            V_next=reshape(V_next, [N_a, N_shocks]);
        end

        % Step 3b: integrate over z' (markov, does not depend on d_semiz)
        V_next_r=reshape(V_next, [N_a, N_semiz, N_z]);
        EV_after_z=sum(V_next_r .* shiftdim(pi_z_J(:,:,jj)', -2), 3); % [N_a, N_semiz, 1, N_z_from]
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

        % Step 4: per-state lookup -- combine d_semiz indirection and a2 interpolation.
        % At Policy (reported, Vrep) -> EVnext_atpolicy; and (Naive) at Policyalt (driver) -> EVnext_atpolicyalt.
        if N_e==0
            a1p_r=reshape(a1prime_idx(:,:,jj),[N_a, N_semiz, N_z]);
            d2_r =reshape(d_semiz_idx(:,:,jj),[N_a, N_semiz, N_z]);
            a2pIdx=reshape(a2primeIndex,[N_a, N_semiz, N_z]);
            a2pPrb=reshape(a2primeProbs,[N_a, N_semiz, N_z]);
            aprime_low=a1p_r+N_a1*(a2pIdx-1);
            aprime_up =a1p_r+N_a1*(a2pIdx);
            base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r(:)-1), [N_a, N_semiz, N_z]);
            EV_lo=reshape(EVnext_byd2((aprime_low+base_off)),[N_a, N_semiz, N_z]);
            EV_up=reshape(EVnext_byd2((aprime_up +base_off)),[N_a, N_semiz, N_z]);
            EVnext_atpolicy=a2pPrb.*EV_lo+(1-a2pPrb).*EV_up;
            if isNaive
                a1p_r_alt=reshape(a1prime_idx_alt(:,:,jj),[N_a, N_semiz, N_z]);
                d2_r_alt =reshape(d_semiz_idx_alt(:,:,jj),[N_a, N_semiz, N_z]);
                a2pIdx_alt=reshape(a2primeIndex_alt,[N_a, N_semiz, N_z]);
                a2pPrb_alt=reshape(a2primeProbs_alt,[N_a, N_semiz, N_z]);
                aprime_low_alt=a1p_r_alt+N_a1*(a2pIdx_alt-1);
                aprime_up_alt =a1p_r_alt+N_a1*(a2pIdx_alt);
                base_off_alt=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_r_alt(:)-1), [N_a, N_semiz, N_z]);
                EV_lo_alt=reshape(EVnext_byd2((aprime_low_alt+base_off_alt)),[N_a, N_semiz, N_z]);
                EV_up_alt=reshape(EVnext_byd2((aprime_up_alt +base_off_alt)),[N_a, N_semiz, N_z]);
                EVnext_atpolicyalt=a2pPrb_alt.*EV_lo_alt+(1-a2pPrb_alt).*EV_up_alt;
            end

            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta    *reshape(EVnext_atpolicyalt, [N_a, N_shocks]);
                Vrep(:,:,jj)  =F_jj    +beta0beta*reshape(EVnext_atpolicy,    [N_a, N_shocks]);
            else
                Vdrive(:,:,jj)=F_jj+beta    *reshape(EVnext_atpolicy, [N_a, N_shocks]);
                Vrep(:,:,jj)  =F_jj+beta0beta*reshape(EVnext_atpolicy, [N_a, N_shocks]);
            end
        else
            EVnext_atpolicy=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
            if isNaive
                EVnext_atpolicyalt=zeros(N_a, N_semiz, N_z, N_e, 'gpuArray');
            end
            for e_c=1:N_e
                block=(e_c-1)*N_shocks + (1:N_shocks);
                a1p_e=reshape(a1prime_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                d2_e =reshape(d_semiz_idx(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                a2pIdx_e=reshape(a2primeIndex(:,block),[N_a, N_semiz, N_z]);
                a2pPrb_e=reshape(a2primeProbs(:,block),[N_a, N_semiz, N_z]);
                aprime_low_e=a1p_e+N_a1*(a2pIdx_e-1);
                aprime_up_e =a1p_e+N_a1*(a2pIdx_e);
                base_off=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e(:)-1), [N_a, N_semiz, N_z]);
                EV_lo=reshape(EVnext_byd2((aprime_low_e+base_off)),[N_a, N_semiz, N_z]);
                EV_up=reshape(EVnext_byd2((aprime_up_e +base_off)),[N_a, N_semiz, N_z]);
                EVnext_atpolicy(:,:,:,e_c)=a2pPrb_e.*EV_lo+(1-a2pPrb_e).*EV_up;
                if isNaive
                    a1p_e_alt=reshape(a1prime_idx_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    d2_e_alt =reshape(d_semiz_idx_alt(:,:,e_c,jj),[N_a, N_semiz, N_z]);
                    a2pIdx_e_alt=reshape(a2primeIndex_alt(:,block),[N_a, N_semiz, N_z]);
                    a2pPrb_e_alt=reshape(a2primeProbs_alt(:,block),[N_a, N_semiz, N_z]);
                    aprime_low_e_alt=a1p_e_alt+N_a1*(a2pIdx_e_alt-1);
                    aprime_up_e_alt =a1p_e_alt+N_a1*(a2pIdx_e_alt);
                    base_off_alt=reshape(N_a*(SZ_grid(:)-1)+N_a*N_semiz*(Z_grid(:)-1)+N_a*N_semiz*N_z*(d2_e_alt(:)-1), [N_a, N_semiz, N_z]);
                    EV_lo_alt=reshape(EVnext_byd2((aprime_low_e_alt+base_off_alt)),[N_a, N_semiz, N_z]);
                    EV_up_alt=reshape(EVnext_byd2((aprime_up_e_alt +base_off_alt)),[N_a, N_semiz, N_z]);
                    EVnext_atpolicyalt(:,:,:,e_c)=a2pPrb_e_alt.*EV_lo_alt+(1-a2pPrb_e_alt).*EV_up_alt;
                end
            end

            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj+beta    *reshape(EVnext_atpolicyalt, [N_a, N_shocks, N_e]);
                Vrep(:,:,:,jj)  =F_jj    +beta0beta*reshape(EVnext_atpolicy,    [N_a, N_shocks, N_e]);
            else
                Vdrive(:,:,:,jj)=F_jj+beta    *reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
                Vrep(:,:,:,jj)  =F_jj+beta0beta*reshape(EVnext_atpolicy, [N_a, N_shocks, N_e]);
            end
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
