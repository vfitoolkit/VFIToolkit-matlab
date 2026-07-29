function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Quasi-hyperbolic ValueFnFromPolicy for experienceassetz WITH grid interpolation layer (vfoptions.gridinterplayer==1).
% experienceassetz: a2prime = aprimeFn(d_expasset, a2, z) -- depends on Markov shock z.
% Under GI, Policy carries an L2 fine-grid index for a1prime; lookup is 2x2 (a1 low/up x a2 low/up).
% Combines the GI reconstruction machinery (cf ValueFnFromPolicy_FHorz_ExpAssetz_GI) with the QH dual-value
% (Naive/Sophisticated) reconstruction (cf ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz).
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is always built from the recursion-driver value.
% Requires N_z>0; may have e too. This file IS the GI variant (only reached via dispatch).

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetz_GI)
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetz you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_z==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_GI: experienceassetz requires N_z>0 (aprimeFn depends on z)')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_GI: experienceassetz requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

if isscalar(n_a)
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_GI: case with no a1 (experience asset as only asset) not yet implemented')
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1;

if isfield(vfoptions,'l_dexperienceassetz')
    l_d2=vfoptions.l_dexperienceassetz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+1:end}};
else
    aprimeFnParamNames={};
end

if N_e==0
    N_ze=N_z;
else
    N_ze=N_z*N_e;
end

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetz + GI internally); Policy, and Policyalt if Naive
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
end

%% Strip trailing L2flag channel if present (keep l_d+l_a1+1)
size_first=l_d+l_a1+1;
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

%% Reshape Policy to canonical Kron form
Policy_k=reshape(Policy,[size_first, N_a, N_ze, N_j]);
if isNaive
    Policyalt_k=reshape(Policyalt,[size_first, N_a, N_ze, N_j]);
end

%% Extract a1prime midpoint (lower) and L2 from Policy (2x2 GI indices)
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:),1);
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1); % weight on upper a1 grid point
w_a1_lower=1-w_a1_upper;
a1_lower=a1_mid;
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
a1_upper=a1_lower+1;
a1_top_clamp=(a1_mid>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp);

%% Same GI-index extraction for Policyalt when Naive
if isNaive
    a1_mid_alt=shiftdim(Policyalt_k(l_d+1,:,:,:),1);
    L2_alt=shiftdim(Policyalt_k(l_d+l_a1+1,:,:,:),1);
    w_a1_upper_alt=(L2_alt-1)/(n2short+1);
    w_a1_lower_alt=1-w_a1_upper_alt;
    a1_lower_alt=a1_mid_alt;
    for ii=2:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii,:,:,:),1);
        a1_lower_alt=a1_lower_alt+cumprods_a1(ii)*(comp-1);
    end
    a1_upper_alt=a1_lower_alt+1;
    a1_top_clamp_alt=(a1_mid_alt>=n_a1(1));
    a1_upper_alt(a1_top_clamp_alt)=a1_lower_alt(a1_top_clamp_alt);
end

%% Joint z+e gridvals for ReturnFn when both present
if N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% Two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
if N_e==0
    Vdrive=zeros(N_a,N_z,N_j,'gpuArray'); Vrep=zeros(N_a,N_z,N_j,'gpuArray');
    zidxoffset=N_a*gpuArray(0:N_z-1); % [1,N_z]
else
    Vdrive=zeros(N_a,N_z,N_e,N_j,'gpuArray'); Vrep=zeros(N_a,N_z,N_e,N_j,'gpuArray');
    zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]); % [1,N_z,1]
end

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);

    % a2prime interpolation + return, at Policy (and Policyalt if Naive)
    if N_e==0
        [a2primeIndex,a2primeProbs]=CreateaprimePolicyExperienceAssetz(Policy_k(:,:,:,jj), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        if isNaive
            [a2primeIndex_alt,a2primeProbs_alt]=CreateaprimePolicyExperienceAssetz(Policyalt_k(:,:,:,jj), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        end
    else
        Policy_zE=reshape(Policy_k(:,:,:,jj),[size_first, N_a, N_z, N_e]);
        a2primeIndex=zeros(N_a, N_z, N_e, 'gpuArray');
        a2primeProbs=zeros(N_a, N_z, N_e, 'gpuArray');
        for e_idx=1:N_e
            [a2pi, a2pp]=CreateaprimePolicyExperienceAssetz(Policy_zE(:,:,:,e_idx), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
            a2primeIndex(:,:,e_idx)=a2pi;
            a2primeProbs(:,:,e_idx)=a2pp;
        end
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
        if isNaive
            Policyalt_zE=reshape(Policyalt_k(:,:,:,jj),[size_first, N_a, N_z, N_e]);
            a2primeIndex_alt=zeros(N_a, N_z, N_e, 'gpuArray');
            a2primeProbs_alt=zeros(N_a, N_z, N_e, 'gpuArray');
            for e_idx=1:N_e
                [a2pi, a2pp]=CreateaprimePolicyExperienceAssetz(Policyalt_zE(:,:,:,e_idx), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
                a2primeIndex_alt(:,:,e_idx)=a2pi;
                a2primeProbs_alt(:,:,e_idx)=a2pp;
            end
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
        end
    end

    if jj==N_j
        if N_e==0
            if isNaive, Vdrive(:,:,jj)=F_alt_jj; Vrep(:,:,jj)=F_jj; else, Vdrive(:,:,jj)=F_jj; Vrep(:,:,jj)=F_jj; end
        else
            if isNaive, Vdrive(:,:,:,jj)=F_alt_jj; Vrep(:,:,:,jj)=F_jj; else, Vdrive(:,:,:,jj)=F_jj; Vrep(:,:,:,jj)=F_jj; end
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % EVnext from the recursion-driver value: integrate e' (iid, if present) then z' (markov) -> [N_a,N_z]
        if N_e==0
            EVnext=Vdrive(:,:,jj+1)*pi_z_J(:,:,jj)';
        else
            EVnext=sum(Vdrive(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3);
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)';
        end
        EVnext(isnan(EVnext))=0;

        % 2x2 corner interpolation at Policy's GI indices (a1 low/up x a2 low/up) -> reported value continuation
        if N_e==0
            a1l=a1_lower(:,:,jj); a1u=a1_upper(:,:,jj);
            wa1l=w_a1_lower(:,:,jj); wa1u=w_a1_upper(:,:,jj);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z]);
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
        else
            a1l=reshape(a1_lower(:,:,jj),[N_a,N_z,N_e]);  a1u=reshape(a1_upper(:,:,jj),[N_a,N_z,N_e]);
            wa1l=reshape(w_a1_lower(:,:,jj),[N_a,N_z,N_e]); wa1u=reshape(w_a1_upper(:,:,jj),[N_a,N_z,N_e]);
            a2l=a2primeIndex;     a2u=a2primeIndex+1;
            wa2l=a2primeProbs;    wa2u=1-a2primeProbs;
            lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
            lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
            EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z,N_e]);
            EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z,N_e]);
            EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z,N_e]);
            EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z,N_e]);
            EVnext_atpolicy=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
        end

        % 2x2 corner interpolation at Policyalt's GI indices (Naive only) -> recursion-driver continuation
        if isNaive
            if N_e==0
                a1l_alt=a1_lower_alt(:,:,jj); a1u_alt=a1_upper_alt(:,:,jj);
                wa1l_alt=w_a1_lower_alt(:,:,jj); wa1u_alt=w_a1_upper_alt(:,:,jj);
                a2l_alt=a2primeIndex_alt;     a2u_alt=a2primeIndex_alt+1;
                wa2l_alt=a2primeProbs_alt;    wa2u_alt=1-a2primeProbs_alt;
                lin_LL_alt=a1l_alt+N_a1*(a2l_alt-1)+zidxoffset; lin_LU_alt=a1l_alt+N_a1*(a2u_alt-1)+zidxoffset;
                lin_UL_alt=a1u_alt+N_a1*(a2l_alt-1)+zidxoffset; lin_UU_alt=a1u_alt+N_a1*(a2u_alt-1)+zidxoffset;
                EV_LL_alt=reshape(EVnext(lin_LL_alt(:)),[N_a,N_z]);
                EV_LU_alt=reshape(EVnext(lin_LU_alt(:)),[N_a,N_z]);
                EV_UL_alt=reshape(EVnext(lin_UL_alt(:)),[N_a,N_z]);
                EV_UU_alt=reshape(EVnext(lin_UU_alt(:)),[N_a,N_z]);
                EVnext_atpolicyalt=wa1l_alt.*wa2l_alt.*EV_LL_alt + wa1l_alt.*wa2u_alt.*EV_LU_alt + wa1u_alt.*wa2l_alt.*EV_UL_alt + wa1u_alt.*wa2u_alt.*EV_UU_alt;
            else
                a1l_alt=reshape(a1_lower_alt(:,:,jj),[N_a,N_z,N_e]);  a1u_alt=reshape(a1_upper_alt(:,:,jj),[N_a,N_z,N_e]);
                wa1l_alt=reshape(w_a1_lower_alt(:,:,jj),[N_a,N_z,N_e]); wa1u_alt=reshape(w_a1_upper_alt(:,:,jj),[N_a,N_z,N_e]);
                a2l_alt=a2primeIndex_alt;     a2u_alt=a2primeIndex_alt+1;
                wa2l_alt=a2primeProbs_alt;    wa2u_alt=1-a2primeProbs_alt;
                lin_LL_alt=a1l_alt+N_a1*(a2l_alt-1)+zidxoffset; lin_LU_alt=a1l_alt+N_a1*(a2u_alt-1)+zidxoffset;
                lin_UL_alt=a1u_alt+N_a1*(a2l_alt-1)+zidxoffset; lin_UU_alt=a1u_alt+N_a1*(a2u_alt-1)+zidxoffset;
                EV_LL_alt=reshape(EVnext(lin_LL_alt(:)),[N_a,N_z,N_e]);
                EV_LU_alt=reshape(EVnext(lin_LU_alt(:)),[N_a,N_z,N_e]);
                EV_UL_alt=reshape(EVnext(lin_UL_alt(:)),[N_a,N_z,N_e]);
                EV_UU_alt=reshape(EVnext(lin_UU_alt(:)),[N_a,N_z,N_e]);
                EVnext_atpolicyalt=wa1l_alt.*wa2l_alt.*EV_LL_alt + wa1l_alt.*wa2u_alt.*EV_LU_alt + wa1u_alt.*wa2l_alt.*EV_UL_alt + wa1u_alt.*wa2u_alt.*EV_UU_alt;
            end
        end

        if N_e==0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta    *EVnext_atpolicyalt;
                Vrep(:,:,jj)  =F_jj    +beta0beta*EVnext_atpolicy;
            else
                Vdrive(:,:,jj)=F_jj+beta    *EVnext_atpolicy;
                Vrep(:,:,jj)  =F_jj+beta0beta*EVnext_atpolicy;
            end
        else
            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj+beta    *EVnext_atpolicyalt;
                Vrep(:,:,:,jj)  =F_jj    +beta0beta*EVnext_atpolicy;
            else
                Vdrive(:,:,:,jj)=F_jj+beta    *EVnext_atpolicy;
                Vrep(:,:,:,jj)  =F_jj+beta0beta*EVnext_atpolicy;
            end
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
if N_e==0
    V   =reshape(Vrep,   [n_a,n_z,N_j]);
    Valt=reshape(Vdrive, [n_a,n_z,N_j]);
else
    V   =reshape(Vrep,   [n_a,n_z,vfoptions.n_e,N_j]);
    Valt=reshape(Vdrive, [n_a,n_z,vfoptions.n_e,N_j]);
end

end
