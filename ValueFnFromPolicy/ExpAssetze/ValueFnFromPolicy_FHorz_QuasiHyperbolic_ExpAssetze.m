function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetze(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Quasi-hyperbolic ValueFnFromPolicy for experienceassetze (a2prime=aprimeFn(d,a2,z,e)).
% Combines the ze aprime machinery (cf ValueFnFromPolicy_FHorz_ExpAssetze) with the QH dual-value
% (Naive/Sophisticated) reconstruction (cf the plain-z block of ValueFnFromPolicy_FHorz_QuasiHyperbolic).
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is always built from the recursion-driver value.

%% Scope (validated-by-test: base method, {z,e} shocks, Naive & Sophisticated)
if prod(vfoptions.n_semiz)>0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic: experienceassetze+SemiExo not yet implemented')
end
if vfoptions.gridinterplayer==1
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssetze_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAssetze)
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use experienceassetze you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_z==0 || N_e==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic (experienceassetze): requires both N_z>0 and N_e>0')
end
l_d=length(n_d);
l_z=length(n_z);
l_e=length(vfoptions.n_e);

% Split a into a1 (standard) and a2 (experience asset)
if isscalar(n_a)
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic (experienceassetze): case with no a1 not yet implemented')
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);

% Which d affects the experience asset (default: last d only)
if isfield(vfoptions,'l_dexperienceassetze')
    l_d2=vfoptions.l_dexperienceassetze;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;

% aprimeFnParamNames: first inputs are (d_expasset..., a2, z, e)
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+1:end}};
else
    aprimeFnParamNames={};
end

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);
a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues + a1prime index (Policy, and Policyalt if Naive)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_e, N_j]);

cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1prime_idx=ones(N_a, N_z, N_e, N_j, 'gpuArray');
for ii=1:l_a1
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(shiftdim(Policy_k(l_d+ii,:,:,:,:),1)-1);
end

if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
    Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_z, N_e, N_j]);
    a1prime_idx_alt=ones(N_a, N_z, N_e, N_j, 'gpuArray');
    for ii=1:l_a1
        a1prime_idx_alt=a1prime_idx_alt+cumprods_a1(ii)*(shiftdim(Policyalt_k(l_d+ii,:,:,:,:),1)-1);
    end
end

%% Joint z+e gridvals for ReturnFn
joint_zegridvals_J=zeros(N_z*N_e, l_z+l_e, N_j, 'gpuArray');
for jj=1:N_j
    joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
end

%% The two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
Vdrive=zeros(N_a,N_z,N_e,N_j,'gpuArray'); % Naive: Valt (at Policyalt).  Soph: Vunderbar (at Policy).
Vrep  =zeros(N_a,N_z,N_e,N_j,'gpuArray'); % Naive: Vtilde (at Policy).   Soph: Vhat (at Policy).
zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);

    % a2prime interpolation + return, at Policy
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetze(Policy_k(:,:,:,:,jj), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, vfoptions.n_e, 0,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
    a2primeIndex=reshape(a2primeIndex,[N_a,N_z,N_e]);
    a2primeProbs=reshape(a2primeProbs,[N_a,N_z,N_e]);
    F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);

    if isNaive
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAssetze(Policyalt_k(:,:,:,:,jj), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, vfoptions.n_e, 0,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), vfoptions.e_gridvals_J(:,:,jj), aprimeFnParamsVec);
        a2primeIndex_alt=reshape(a2primeIndex_alt,[N_a,N_z,N_e]);
        a2primeProbs_alt=reshape(a2primeProbs_alt,[N_a,N_z,N_e]);
        F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
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

        % EVnext from the recursion-driver value: integrate e' (iid) then z' (markov) -> [N_a,N_z]
        EVnext=sum(Vdrive(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3);
        EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)';
        EVnext(isnan(EVnext))=0;

        % Interpolated lookup at Policy's a2prime
        a1p=a1prime_idx(:,:,:,jj);
        EV_low=reshape(EVnext((a1p+N_a1*(a2primeIndex-1)+zidxoffset)),[N_a,N_z,N_e]);
        EV_up =reshape(EVnext((a1p+N_a1*(a2primeIndex)  +zidxoffset)),[N_a,N_z,N_e]);
        EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;

        if isNaive
            a1p_alt=a1prime_idx_alt(:,:,:,jj);
            EV_low_alt=reshape(EVnext((a1p_alt+N_a1*(a2primeIndex_alt-1)+zidxoffset)),[N_a,N_z,N_e]);
            EV_up_alt =reshape(EVnext((a1p_alt+N_a1*(a2primeIndex_alt)  +zidxoffset)),[N_a,N_z,N_e]);
            EVnext_atpolicyalt=a2primeProbs_alt.*EV_low_alt+(1-a2primeProbs_alt).*EV_up_alt;

            Vdrive(:,:,:,jj)=F_alt_jj+beta    *EVnext_atpolicyalt; % Valt
            Vrep(:,:,:,jj)  =F_jj    +beta0beta*EVnext_atpolicy;   % Vtilde
        else
            Vdrive(:,:,:,jj)=F_jj+beta    *EVnext_atpolicy; % Vunderbar
            Vrep(:,:,:,jj)  =F_jj+beta0beta*EVnext_atpolicy; % Vhat
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
V   =reshape(Vrep,   [n_a,n_z,vfoptions.n_e,N_j]);
Valt=reshape(Vdrive, [n_a,n_z,vfoptions.n_e,N_j]);

end
