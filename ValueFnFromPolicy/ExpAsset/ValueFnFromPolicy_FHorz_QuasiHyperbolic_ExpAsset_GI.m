function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has an experience asset (vfoptions.experienceasset>=1),
% uses the grid interpolation layer (vfoptions.gridinterplayer==1), and quasi-hyperbolic discounting.
%
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is ALWAYS built from the recursion-driver value (Vdrive).
%
% Structural base: ValueFnFromPolicy_FHorz_ExpAsset_GI (the 2x2 interpolated lookup is carried over
% untouched). QH bookkeeping mirrors ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset.
%
% Under GI, Policy stores an extra L2 (layer-2 fine-grid) index at the end, used to interpolate a1prime
% between two adjacent a1_grid points. a2prime continues to be interpolated via aprimeFn. Per-state
% EVnext lookup is therefore a 2x2 interpolation: lower/upper a1 x lower/upper a2.
%
% The interpolated lookup is written once and run over a pass loop ({Policy} or {Policy,Policyalt}),
% since the interpolation is identical for the two policies -- only the indices/weights differ.

%% Setup
[z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);

if ~isfield(vfoptions,'aprimeFn')
    error('To use an experience asset you must define vfoptions.aprimeFn')
end
aprimeFn=vfoptions.aprimeFn;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
if N_d==0
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset_GI: experienceasset requires at least one decision variable')
end
l_d=length(n_d);
l_a=length(n_a);

% noa1 case (n_a is scalar -- experience asset is the only endogenous state): GI refines a1, which
% doesn't apply when there's no a1. Fall back to the non-GI version (which handles noa1 correctly).
% Matches the upstream VFI convention (noa1 has no GI/DC/DC+GI raw files).
if isscalar(n_a)
    vfoptions.gridinterplayer=0;
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end
n_a1=n_a(1:end-1);
N_a1=prod(n_a1);
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1; % Policy stores a1prime only (plus L2 in GI)

if isfield(vfoptions,'l_dexperienceasset')
    l_d2=vfoptions.l_dexperienceasset;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);

% aprimeFnParamNames
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}};
else
    aprimeFnParamNames={};
end

if N_z==0 && N_e==0
    N_ze=0;
elseif N_z>0 && N_e==0
    N_ze=N_z;
elseif N_z==0 && N_e>0
    N_ze=N_e;
else
    N_ze=N_z*N_e;
end

n2short=vfoptions.ngridinterp;

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles experienceasset + GI internally: drops a2prime, combines a1prime midpoint + L2 into fine index)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); % [N_a, l_daprime, N_j]
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
end
if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    if N_z==0 && N_e==0
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,1,3]);
    else
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
    end
end

%% Strip trailing L2flag channel if present (Policy may carry it; we only need l_d+l_a1+1 channels)
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

%% Reshape Policy to canonical Kron form: [l_d+l_a1+1, N_a, N_ze, N_j] (or no shock dim when no shocks)
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[size_first, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[size_first, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[size_first, N_a, N_z*N_e, N_j]);
end
if isNaive
    if N_z==0 && N_e==0
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_j]);
    elseif N_z>0 && N_e==0
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_z, N_j]);
    elseif N_z==0 && N_e>0
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_e, N_j]);
    else
        Policyalt_k=reshape(Policyalt,[size_first, N_a, N_z*N_e, N_j]);
    end
end

%% Extract a1prime midpoint (lower) and L2 from Policy
% Position l_d+1 is a1prime midpoint (first a1 component); l_d+2 .. l_d+l_a1 are other a1prime indices; l_d+l_a1+1 is L2
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
a1_mid=shiftdim(Policy_k(l_d+1,:,:,:),1);
L2=shiftdim(Policy_k(l_d+l_a1+1,:,:,:),1);
w_a1_upper=(L2-1)/(n2short+1); % weight on upper a1 grid point
w_a1_lower=1-w_a1_upper;
a1_lower=a1_mid; % first dim contribution (1*(a1_mid-1)+1 = a1_mid)
for ii=2:l_a1
    comp=shiftdim(Policy_k(l_d+ii,:,:,:),1);
    a1_lower=a1_lower+cumprods_a1(ii)*(comp-1);
end
% upper a1 differs only in the first a1 component (clamp at top of grid)
a1_upper=a1_lower+1;
a1_top_clamp=(a1_mid>=n_a1(1));
a1_upper(a1_top_clamp)=a1_lower(a1_top_clamp); % no-op when at top
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

%% Joint zegridvals for ReturnFn (when both z and e present)
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% Two value functions (Vdrive uses beta and drives the recursion; Vrep uses beta0*beta and is reported as V)
if N_z==0 && N_e==0
    Vdrive=zeros(N_a, N_j, 'gpuArray');       Vrep=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    Vdrive=zeros(N_a, N_e, N_j, 'gpuArray');  Vrep=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    Vdrive=zeros(N_a, N_z, N_j, 'gpuArray');  Vrep=zeros(N_a, N_z, N_j, 'gpuArray');
else
    Vdrive=zeros(N_a, N_z, N_e, N_j, 'gpuArray'); Vrep=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at this age (for each policy)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj); % [size_first, N_a]
    else
        Policy_slice=Policy_k(:,:,:,jj); % [size_first, N_a, N_ze]
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAsset(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);
    if isNaive
        if N_z==0 && N_e==0
            Policyalt_slice=Policyalt_k(:,:,jj);
        else
            Policyalt_slice=Policyalt_k(:,:,:,jj);
        end
        [a2primeIndex_alt, a2primeProbs_alt]=CreateaprimePolicyExperienceAsset(Policyalt_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);
    end

    % Step 2: ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_z==0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
    elseif N_z==0 && N_e>0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
    elseif N_z>0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end
    if isNaive
        if N_z==0 && N_e==0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
        elseif N_z==0 && N_e>0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
        elseif N_z>0 && N_e==0
            F_alt_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        else
            F_alt_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyaltValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
        end
    end

    if jj==N_j
        if N_z==0 && N_e==0
            if isNaive, Vdrive(:,jj)=F_alt_jj; else, Vdrive(:,jj)=F_jj; end
            Vrep(:,jj)=F_jj;
        elseif N_z==0 && N_e>0
            if isNaive, Vdrive(:,:,jj)=F_alt_jj; else, Vdrive(:,:,jj)=F_jj; end
            Vrep(:,:,jj)=F_jj;
        elseif N_z>0 && N_e==0
            if isNaive, Vdrive(:,:,jj)=F_alt_jj; else, Vdrive(:,:,jj)=F_jj; end
            Vrep(:,:,jj)=F_jj;
        else
            if isNaive, Vdrive(:,:,:,jj)=F_alt_jj; else, Vdrive(:,:,:,jj)=F_jj; end
            Vrep(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));
        beta0=CreateVectorFromParams(Parameters,vfoptions.QHadditionaldiscount,jj);
        beta0beta=beta0*beta;

        % Step 3: EVnext, built from the recursion-driver value (Vdrive)
        if N_z==0 && N_e==0
            EVnext=Vdrive(:,jj+1); % [N_a]
        elseif N_z==0 && N_e>0
            EVnext=sum(Vdrive(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            EVnext=Vdrive(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            EVnext=sum(Vdrive(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj+1), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: 2x2 interpolated lookup of EVnext at each policy, using
        % (a1_lower/upper, w_a1) x (a2_lower/upper, a2primeProbs). Pass 1 is Policy (giving the
        % reported value's continuation); pass 2, when Naive, is Policyalt (the recursion driver's).
        for pass=1:(1+isNaive)
            if pass==1
                a1Lo=a1_lower; a1Up=a1_upper; wLo=w_a1_lower; wUp=w_a1_upper;
                a2pi=a2primeIndex; a2pp=a2primeProbs;
            else
                a1Lo=a1_lower_alt; a1Up=a1_upper_alt; wLo=w_a1_lower_alt; wUp=w_a1_upper_alt;
                a2pi=a2primeIndex_alt; a2pp=a2primeProbs_alt;
            end

            if N_z==0 && N_e==0
                a1l=a1Lo(:,jj); a1u=a1Up(:,jj);
                wa1l=wLo(:,jj); wa1u=wUp(:,jj);
                a2l=a2pi;    a2u=a2pi+1;
                wa2l=a2pp;   wa2u=1-a2pp;
                EV_LL=EVnext(a1l+N_a1*(a2l-1));
                EV_LU=EVnext(a1l+N_a1*(a2u-1));
                EV_UL=EVnext(a1u+N_a1*(a2l-1));
                EV_UU=EVnext(a1u+N_a1*(a2u-1));
                EVnext_pass=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            elseif N_z==0 && N_e>0
                a1l=a1Lo(:,:,jj); a1u=a1Up(:,:,jj);
                wa1l=wLo(:,:,jj); wa1u=wUp(:,:,jj);
                a2l=a2pi;    a2u=a2pi+1;
                wa2l=a2pp;   wa2u=1-a2pp;
                lin_LL=a1l+N_a1*(a2l-1); lin_LU=a1l+N_a1*(a2u-1);
                lin_UL=a1u+N_a1*(a2l-1); lin_UU=a1u+N_a1*(a2u-1);
                EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_e]);
                EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_e]);
                EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_e]);
                EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_e]);
                EVnext_pass=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            elseif N_z>0 && N_e==0
                a1l=a1Lo(:,:,jj); a1u=a1Up(:,:,jj);
                wa1l=wLo(:,:,jj); wa1u=wUp(:,:,jj);
                a2l=a2pi;    a2u=a2pi+1;
                wa2l=a2pp;   wa2u=1-a2pp;
                zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z]
                lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
                lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
                EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z]);
                EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z]);
                EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z]);
                EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z]);
                EVnext_pass=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            else
                a1l=reshape(a1Lo(:,:,jj),[N_a,N_z,N_e]);  a1u=reshape(a1Up(:,:,jj),[N_a,N_z,N_e]);
                wa1l=reshape(wLo(:,:,jj),[N_a,N_z,N_e]); wa1u=reshape(wUp(:,:,jj),[N_a,N_z,N_e]);
                a2l=reshape(a2pi,[N_a,N_z,N_e]); a2u=a2l+1;
                wa2l=reshape(a2pp,[N_a,N_z,N_e]); wa2u=1-wa2l;
                zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);
                lin_LL=a1l+N_a1*(a2l-1)+zidxoffset; lin_LU=a1l+N_a1*(a2u-1)+zidxoffset;
                lin_UL=a1u+N_a1*(a2l-1)+zidxoffset; lin_UU=a1u+N_a1*(a2u-1)+zidxoffset;
                EV_LL=reshape(EVnext(lin_LL(:)),[N_a,N_z,N_e]);
                EV_LU=reshape(EVnext(lin_LU(:)),[N_a,N_z,N_e]);
                EV_UL=reshape(EVnext(lin_UL(:)),[N_a,N_z,N_e]);
                EV_UU=reshape(EVnext(lin_UU(:)),[N_a,N_z,N_e]);
                EVnext_pass=wa1l.*wa2l.*EV_LL + wa1l.*wa2u.*EV_LU + wa1u.*wa2l.*EV_UL + wa1u.*wa2u.*EV_UU;
            end

            if pass==1
                EVnext_atP=EVnext_pass;
            else
                EVnext_atPa=EVnext_pass;
            end
        end

        % Step 5: Vdrive carries beta (and, when Naive, Policyalt's return and continuation);
        % Vrep carries beta0*beta at Policy and is what gets reported as V.
        if N_z==0 && N_e==0
            if isNaive
                Vdrive(:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,jj)=F_jj+beta0beta*EVnext_atP;
        elseif N_z==0 && N_e>0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,:,jj)=F_jj+beta0beta*EVnext_atP;
        elseif N_z>0 && N_e==0
            if isNaive
                Vdrive(:,:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,:,jj)=F_jj+beta0beta*EVnext_atP;
        else
            if isNaive
                Vdrive(:,:,:,jj)=F_alt_jj+beta*EVnext_atPa;
            else
                Vdrive(:,:,:,jj)=F_jj+beta*EVnext_atP;
            end
            Vrep(:,:,:,jj)=F_jj+beta0beta*EVnext_atP;
        end
    end
end

%% Output: V is the reported (beta0*beta) value; Valt is the recursion-driver (beta) value
if N_z==0 && N_e==0
    V   =reshape(Vrep,   [n_a, N_j]);
    Valt=reshape(Vdrive, [n_a, N_j]);
elseif N_z==0 && N_e>0
    V   =reshape(Vrep,   [n_a, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V   =reshape(Vrep,   [n_a, n_z, N_j]);
    Valt=reshape(Vdrive, [n_a, n_z, N_j]);
else
    V   =reshape(Vrep,   [n_a, n_z, vfoptions.n_e, N_j]);
    Valt=reshape(Vdrive, [n_a, n_z, vfoptions.n_e, N_j]);
end


end
