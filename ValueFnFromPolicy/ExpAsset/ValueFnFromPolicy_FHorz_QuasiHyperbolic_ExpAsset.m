function [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has an experience asset (vfoptions.experienceasset>=1)
% and quasi-hyperbolic discounting.
%
%   Naive:         V=Vtilde (beta0*beta at Policy);  Valt = exponential value (beta at Policyalt, drives recursion).
%   Sophisticated: V=Vhat   (beta0*beta at Policy);  Valt = Vunderbar (beta at Policy, drives recursion).
% The continuation (EVnext) is ALWAYS built from the recursion-driver value (Vdrive).
%
% Structural base: ValueFnFromPolicy_FHorz_ExpAsset (the a2prime interpolation is carried over
% untouched). QH bookkeeping mirrors ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAssete.
%
% The interpolated lookup is written once and run over a pass loop ({Policy} or {Policy,Policyalt}),
% rather than duplicated, so there is a single copy of the four-shock-case interpolation to check
% against the exponential source.

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset_SemiExo(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    [V,Valt]=ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset_GI(Policy,Policyalt,isNaive,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Setup (mirrors ValueFnFromPolicy_FHorz_ExpAsset)
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
    error('ValueFnFromPolicy_FHorz_QuasiHyperbolic_ExpAsset: experienceasset requires at least one decision variable (the one driving a2prime)')
end
l_d=length(n_d);
l_a=length(n_a);

l_a2=vfoptions.experienceasset; % l_a2 = number of a2 (experience-asset) dims
if length(n_a)<=l_a2
    n_a1=0;
    N_a1=0;
    l_a1=0;
else
    n_a1=n_a(1:end-l_a2);
    N_a1=prod(n_a1);
    l_a1=length(n_a1);
end
n_a2=n_a(end-l_a2+1:end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_aprime=l_a1;

if isfield(vfoptions,'l_dexperienceasset')
    l_d2=vfoptions.l_dexperienceasset;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
n_d2=n_d(end-l_d2+1:end);

temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+(l_a2>=2)+1:end}};
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

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues, Policy in Kron form, and the a1prime joint index -- for Policy, and Policyalt if Naive
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]);
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_j]);
    a1prime_idx=ones(N_a,N_j,'gpuArray');
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]);
    if N_z>0 && N_e==0
        Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_j]);
    elseif N_z==0 && N_e>0
        Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_e, N_j]);
    else
        Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z*N_e, N_j]);
    end
    a1prime_idx=ones(N_a,N_ze,N_j,'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

if isNaive
    PolicyaltValues=PolicyInd2Val_FHorz(Policyalt,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
    if N_z==0 && N_e==0
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,1,3]);
        Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_j]);
        a1prime_idx_alt=ones(N_a,N_j,'gpuArray');
    else
        PolicyaltValuesPermute=permute(PolicyaltValues,[2,3,1,4]);
        if N_z>0 && N_e==0
            Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_z, N_j]);
        elseif N_z==0 && N_e>0
            Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_e, N_j]);
        else
            Policyalt_k=reshape(Policyalt,[l_d+l_a1, N_a, N_z*N_e, N_j]);
        end
        a1prime_idx_alt=ones(N_a,N_ze,N_j,'gpuArray');
    end
    for ii=1:l_a1
        comp=shiftdim(Policyalt_k(l_d+ii, :, :, :),1);
        a1prime_idx_alt=a1prime_idx_alt+cumprods_a1(ii)*(comp-1);
    end
end

%% Joint gridvals for ReturnFn (z + e combined when both present)
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

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs at Policy (and at Policyalt if Naive)
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj);
    else
        Policy_slice=Policy_k(:,:,:,jj);
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

    % Step 2: ReturnFn at policy (and at Policyalt if Naive)
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
        % Terminal period: no continuation, so each value is just the return at its own policy
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
            EVnext=Vdrive(:,jj+1);
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

        % Step 4: interpolated lookup of EVnext at each policy's a2prime. Pass 1 is Policy (giving
        % the reported value's continuation); pass 2, when Naive, is Policyalt (the recursion
        % driver's). The interpolation itself is identical, hence the pass loop.
        for pass=1:(1+isNaive)
            if pass==1
                a2pi=a2primeIndex; a2pp=a2primeProbs; a1pi=a1prime_idx;
            else
                a2pi=a2primeIndex_alt; a2pp=a2primeProbs_alt; a1pi=a1prime_idx_alt;
            end

            if N_z==0 && N_e==0
                if l_a2==1
                    if N_a1==0
                        aprime_low=a2pi;
                        aprime_up =a2pi+1;
                    else
                        a1p=a1pi(:,jj); % [N_a, 1]
                        aprime_low=a1p+N_a1*(a2pi-1);
                        aprime_up =a1p+N_a1*(a2pi);
                    end
                    EV_low=EVnext(aprime_low);
                    EV_up =EVnext(aprime_up);
                    EVnext_pass=a2pp.*EV_low+(1-a2pp).*EV_up; % [N_a, 1]
                else
                    % l_a2==2: a2pi/a2pp are [N_a, l_a2] per-dim factored.
                    % Nested 2-corner with skipinterp at each level for bit-exactness when V flat.
                    n_a2_1=n_a2(1);
                    loIdx_1=a2pi(:,1); % [N_a, 1]
                    loIdx_2=a2pi(:,2);
                    prob_1=a2pp(:,1);
                    prob_2=a2pp(:,2);

                    if N_a1==0
                        a1p=zeros(N_a,1,'gpuArray'); N_a1_eff=1;
                    else
                        a1p=a1pi(:,jj); N_a1_eff=N_a1;
                    end
                    aprime_ll=a1p+N_a1_eff*(loIdx_1+n_a2_1*(loIdx_2-1)-1);
                    aprime_hl=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*(loIdx_2-1)-1);
                    aprime_lh=a1p+N_a1_eff*(loIdx_1+n_a2_1*loIdx_2-1);
                    aprime_hh=a1p+N_a1_eff*((loIdx_1+1)+n_a2_1*loIdx_2-1);
                    V_ll=EVnext(aprime_ll);
                    V_hl=EVnext(aprime_hl);
                    V_lh=EVnext(aprime_lh);
                    V_hh=EVnext(aprime_hh);

                    p1_loy=prob_1; p1_loy(V_ll==V_hl)=0;
                    c_ll=p1_loy.*V_ll; c_ll(isnan(c_ll))=0;
                    c_hl=(1-p1_loy).*V_hl; c_hl(isnan(c_hl))=0;
                    EV_loy=c_ll+c_hl;
                    p1_hiy=prob_1; p1_hiy(V_lh==V_hh)=0;
                    c_lh=p1_hiy.*V_lh; c_lh(isnan(c_lh))=0;
                    c_hh=(1-p1_hiy).*V_hh; c_hh(isnan(c_hh))=0;
                    EV_hiy=c_lh+c_hh;
                    p2=prob_2; p2(EV_loy==EV_hiy)=0;
                    c_loy=p2.*EV_loy; c_loy(isnan(c_loy))=0;
                    c_hiy=(1-p2).*EV_hiy; c_hiy(isnan(c_hiy))=0;
                    EVnext_pass=c_loy+c_hiy; % [N_a, 1]
                end
            elseif N_z==0 && N_e>0
                if N_a1==0
                    aprime_low=a2pi;
                    aprime_up =a2pi+1;
                else
                    a1p=a1pi(:,:,jj); % [N_a, N_e]
                    aprime_low=a1p+N_a1*(a2pi-1);
                    aprime_up =a1p+N_a1*(a2pi);
                end
                EV_low=reshape(EVnext(aprime_low(:)),[N_a,N_e]);
                EV_up =reshape(EVnext(aprime_up(:)), [N_a,N_e]);
                EVnext_pass=a2pp.*EV_low+(1-a2pp).*EV_up;
            elseif N_z>0 && N_e==0
                if N_a1==0
                    aprime_low=a2pi;
                    aprime_up =a2pi+1;
                else
                    a1p=a1pi(:,:,jj);
                    aprime_low=a1p+N_a1*(a2pi-1);
                    aprime_up =a1p+N_a1*(a2pi);
                end
                zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z]
                lin_low=aprime_low+zidxoffset;
                lin_up =aprime_up +zidxoffset;
                EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z]);
                EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z]);
                EVnext_pass=a2pp.*EV_low+(1-a2pp).*EV_up;
            else
                a2pIdx=reshape(a2pi,[N_a, N_z, N_e]);
                a2pPrb=reshape(a2pp,[N_a, N_z, N_e]);
                if N_a1==0
                    aprime_low=a2pIdx;
                    aprime_up =a2pIdx+1;
                else
                    a1p=reshape(a1pi(:,:,jj),[N_a, N_z, N_e]);
                    aprime_low=a1p+N_a1*(a2pIdx-1);
                    aprime_up =a1p+N_a1*(a2pIdx);
                end
                zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]); % [1, N_z, 1]
                lin_low=aprime_low+zidxoffset;
                lin_up =aprime_up +zidxoffset;
                EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z,N_e]);
                EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z,N_e]);
                EVnext_pass=a2pPrb.*EV_low+(1-a2pPrb).*EV_up;
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
