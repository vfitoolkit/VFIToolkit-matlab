function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Quasi-hyperbolic ValueFnFromPolicy for experienceassetz (a2prime=aprimeFn(d,a2,z); may also have iid e).
% Combines the z aprime machinery (cf ValueFnFromPolicy_FHorz_ExpAssetz) with the QH dual-value
% (Naive/Sophisticated) reconstruction (cf the plain-z/z+e blocks of ValueFnFromPolicy_FHorz_QuasiHyperbolic).
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is always built from the recursion-driver value.

%% Scope (validated-by-test: base method, {z} and {z,e} shocks, Naive & Sophisticated)
if prod(vfoptions.n_semiz)>0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: experienceassetz+SemiExo not yet implemented')
end
if vfoptions.gridinterplayer==1
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetz_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetz)
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetz you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_z==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic (experienceassetz): requires N_z>0')
end
l_d=length(n_d);
l_z=length(n_z);

if isscalar(n_a)
    % noa1: the experience asset is the only endogenous state
    n_a1=0;
    N_a1=1; % so EV_low=EVnext(a1p+N_a1*(a2pi-1)+...) reduces to the a2primeIndex lookup (a1p stays 1)
    l_a1=0; % Policy contains only the d channels
else
    n_a1=n_a(1:end-1);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a2=length(n_a2);

if isfield(vfoptions,'l_dexperienceassetz')
    l_d2=vfoptions.l_dexperienceassetz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;

% aprimeFnParamNames: first inputs are (d_expasset..., a2, z)
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

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues + a1prime index (Policy, and Policyalt if Naive)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_ze, N_j]);
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
Policy_a1prime=ones(N_a, N_ze, N_j, 'gpuArray');
for ii=1:l_a1
    Policy_a1prime=Policy_a1prime+cumprods_a1(ii)*(shiftdim(Policy_k(l_d+ii,:,:,:),1)-1);
end

if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
    Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_ze, N_j]);
    Policyalt_a1prime=ones(N_a, N_ze, N_j, 'gpuArray');
    for ii=1:l_a1
        Policyalt_a1prime=Policyalt_a1prime+cumprods_a1(ii)*(shiftdim(Policyalt_k(l_d+ii,:,:,:),1)-1);
    end
end

%% Joint z+e gridvals for ReturnFn when e present
if N_e>0
    ze_gridvals_J=zeros(N_z*N_e, l_z+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        ze_gridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
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
        [a2pi,a2pp]=CreateaprimePolicyExperienceAssetz(Policy_k(:,:,:,jj), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        if isNaive
            [a2pi_alt,a2pp_alt]=CreateaprimePolicyExperienceAssetz(Policyalt_k(:,:,:,jj), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        end
    else
        Policy_zE=reshape(Policy_k(:,:,:,jj),[l_d+l_a1, N_a, N_z, N_e]);
        a2pi=zeros(N_a,N_z,N_e,'gpuArray'); a2pp=zeros(N_a,N_z,N_e,'gpuArray');
        for e_idx=1:N_e
            [t1,t2]=CreateaprimePolicyExperienceAssetz(Policy_zE(:,:,:,e_idx), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
            a2pi(:,:,e_idx)=t1; a2pp(:,:,e_idx)=t2;
        end
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, ze_gridvals_J(:,:,jj)), [N_a,N_z,N_e]);
        if isNaive
            Policyalt_zE=reshape(Policyalt_k(:,:,:,jj),[l_d+l_a1, N_a, N_z, N_e]);
            a2pi_alt=zeros(N_a,N_z,N_e,'gpuArray'); a2pp_alt=zeros(N_a,N_z,N_e,'gpuArray');
            for e_idx=1:N_e
                [t1,t2]=CreateaprimePolicyExperienceAssetz(Policyalt_zE(:,:,:,e_idx), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
                a2pi_alt(:,:,e_idx)=t1; a2pp_alt(:,:,e_idx)=t2;
            end
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, ze_gridvals_J(:,:,jj)), [N_a,N_z,N_e]);
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
            EVnext=sum(Vdrive(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3);
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)';
        end
        EVnext(isnan(EVnext))=0;

        % Interpolated lookup at Policy's a2prime (reported value)
        a1p=reshape(Policy_a1prime(:,:,jj),size(a2pi));
        EV_low=reshape(EVnext((a1p+N_a1*(a2pi-1)+zidxoffset)),size(a2pi));
        EV_up =reshape(EVnext((a1p+N_a1*(a2pi)  +zidxoffset)),size(a2pi));
        EVnext_atpolicy=a2pp.*EV_low+(1-a2pp).*EV_up;

        if isNaive
            a1p_alt=reshape(Policyalt_a1prime(:,:,jj),size(a2pi_alt));
            EV_low_alt=reshape(EVnext((a1p_alt+N_a1*(a2pi_alt-1)+zidxoffset)),size(a2pi_alt));
            EV_up_alt =reshape(EVnext((a1p_alt+N_a1*(a2pi_alt)  +zidxoffset)),size(a2pi_alt));
            EVnext_atpolicyalt=a2pp_alt.*EV_low_alt+(1-a2pp_alt).*EV_up_alt;
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
