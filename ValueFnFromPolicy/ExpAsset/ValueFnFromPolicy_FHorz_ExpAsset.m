function V=ValueFnFromPolicy_FHorz_ExpAsset(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has an experience asset (vfoptions.experienceasset==1).
% a = [a1, a2]: a2 is the experience asset; a2prime is determined by aprimeFn(d_expasset, a2).
% Policy stores d and a1prime only (a2prime is implicit). For lookup of V at next period, a2prime is
% (in general) fractional and is linearly interpolated onto the a2_grid via a2primeIndex/a2primeProbs.

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    V=ValueFnFromPolicy_FHorz_ExpAsset_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_ExpAsset_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    return
end

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
    error('ValueFnFromPolicy_FHorz_ExpAsset: experienceasset requires at least one decision variable (the one driving a2prime)')
end
l_d=length(n_d);
l_a=length(n_a);

% Split a into a1 (standard) and a2 (experience asset)
if isscalar(n_a)
    n_a1=0;
    N_a1=0;
    error('ValueFnFromPolicy_FHorz_ExpAsset: case with no a1 (experience asset as only asset) not yet implemented')
else
    n_a1=n_a(1:end-1);
    N_a1=prod(n_a1);
end
n_a2=n_a(end);
N_a2=prod(n_a2);
a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);
l_aprime=l_a1; % Policy stores a1prime only (a2prime implicit)

% Which d affects the experience asset (default: last d only)
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

% Combined shock dim for CreateaprimePolicyExperienceAsset
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

%% PolicyValues (PolicyInd2Val_FHorz handles experienceasset internally, drops a2prime; auto-adds n_e)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
% PolicyValues shape:
% - N_z==0 && N_e==0: [l_daprime, N_a, N_j]
% - else: [l_daprime, N_a, N_ze, N_j]
if N_z==0 && N_e==0
    PolicyValuesPermute=permute(PolicyValues,[2,1,3]); % [N_a, l_daprime, N_j]
else
    PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]
end

%% Reshape Policy to canonical Kron form for CreateaprimePolicyExperienceAsset
% Target: [l_d+l_a1, N_a, N_ze, N_j] (or [l_d+l_a1, N_a, N_j] if no shocks)
if N_z==0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_j]);
elseif N_z>0 && N_e==0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z, N_j]);
elseif N_z==0 && N_e>0
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_e, N_j]);
else
    Policy_k=reshape(Policy,[l_d+l_a1, N_a, N_z*N_e, N_j]);
end

%% Build a1prime joint index across N_a1 dims (from Policy)
% a1prime components are positions l_d+1 .. l_d+l_a1
if N_z==0 && N_e==0
    a1prime_idx=ones(N_a,N_j,'gpuArray');
else
    a1prime_idx=ones(N_a,N_ze,N_j,'gpuArray');
end
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy_k(l_d+ii, :, :, :),1);
    a1prime_idx=a1prime_idx+cumprods_a1(ii)*(comp-1);
end

%% Joint gridvals for ReturnFn (z + e combined when both present)
if N_z>0 && N_e>0
    joint_zegridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        joint_zegridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% Backward iteration
if N_z==0 && N_e==0
    V=zeros(N_a, N_j, 'gpuArray');
elseif N_z==0 && N_e>0
    V=zeros(N_a, N_e, N_j, 'gpuArray');
elseif N_z>0 && N_e==0
    V=zeros(N_a, N_z, N_j, 'gpuArray');
else
    V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    % Step 1: a2primeIndex, a2primeProbs for each state at this age
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z==0 && N_e==0
        Policy_slice=Policy_k(:,:,jj); % [l_d+l_a1, N_a]
    else
        Policy_slice=Policy_k(:,:,:,jj); % [l_d+l_a1, N_a, N_ze]
    end
    [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAsset(Policy_slice, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);
    % a2primeIndex, a2primeProbs shape:
    % - N_z==0 && N_e==0: [N_a, 1]
    % - else: [N_a, N_ze]

    % Step 2: ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_z==0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,jj), l_daprime, n_a, 0, a_gridvals, []);
        % F_jj shape: [N_a, 1]
    elseif N_z==0 && N_e>0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, vfoptions.n_e, a_gridvals, vfoptions.e_gridvals_J(:,:,jj));
        % F_jj shape: [N_a, N_e]
    elseif N_z>0 && N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        % F_jj shape: [N_a, N_z]
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, joint_zegridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end

    if jj==N_j
        if N_z==0 && N_e==0
            V(:,jj)=F_jj;
        elseif N_z==0 && N_e>0
            V(:,:,jj)=F_jj;
        elseif N_z>0 && N_e==0
            V(:,:,jj)=F_jj;
        else
            V(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3: build EVnext indexed by (anext, z_from) -- integrate over e' (iid) then over z' (markov)
        if N_z==0 && N_e==0
            EVnext=V(:,jj+1); % [N_a]
        elseif N_z==0 && N_e>0
            % e iid: integrate over e' to collapse to EVnext indexed by anext only
            EVnext=sum(V(:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -1), 2); % [N_a, 1]
        elseif N_z>0 && N_e==0
            % EVnext(anext, z_from) = sum_{z_to} pi(z_from, z_to) * V(anext, z_to)
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            % Integrate over e' then over z'
            EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: interpolated lookup using a1prime_idx and a2primeIndex/Probs
        % For each current state (a, [z, e]):
        %   aprime_low_kron = a1prime + N_a1 * (a2primeIndex - 1)
        %   aprime_up_kron  = a1prime + N_a1 *  a2primeIndex
        %   EVnext_atpolicy = a2primeProbs * EVnext[aprime_low, ...] + (1-a2primeProbs) * EVnext[aprime_up, ...]
        if N_z==0 && N_e==0
            a1p=a1prime_idx(:,jj); % [N_a]
            aprime_low=a1p+N_a1*(a2primeIndex-1); % [N_a, 1]
            aprime_up =a1p+N_a1*(a2primeIndex);
            EV_low=EVnext(aprime_low);
            EV_up =EVnext(aprime_up);
            EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up; % [N_a, 1]
            V(:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z==0 && N_e>0
            % a1prime_idx, a2primeIndex shape: [N_a, N_e]; EVnext shape: [N_a, 1]
            a1p=a1prime_idx(:,:,jj); % [N_a, N_e]
            aprime_low=a1p+N_a1*(a2primeIndex-1);
            aprime_up =a1p+N_a1*(a2primeIndex);
            EV_low=reshape(EVnext(aprime_low(:)),[N_a,N_e]);
            EV_up =reshape(EVnext(aprime_up(:)), [N_a,N_e]);
            EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        elseif N_z>0 && N_e==0
            % a1prime_idx, a2primeIndex shape: [N_a, N_z]; EVnext shape: [N_a, N_z]
            a1p=a1prime_idx(:,:,jj);
            aprime_low=a1p+N_a1*(a2primeIndex-1);
            aprime_up =a1p+N_a1*(a2primeIndex);
            zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z]
            lin_low=aprime_low+zidxoffset; % broadcast: [N_a, N_z]
            lin_up =aprime_up +zidxoffset;
            EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z]);
            EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z]);
            EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        else
            % a1prime_idx, a2primeIndex shape: [N_a, N_z*N_e]; EVnext shape: [N_a, N_z]
            % For each (a, z, e), look up at aprime in dim 1 of EVnext, and z (=current state's z) in dim 2.
            a1p=reshape(a1prime_idx(:,:,jj),[N_a, N_z, N_e]);
            a2pIdx=reshape(a2primeIndex,[N_a, N_z, N_e]);
            a2pPrb=reshape(a2primeProbs,[N_a, N_z, N_e]);
            aprime_low=a1p+N_a1*(a2pIdx-1);
            aprime_up =a1p+N_a1*(a2pIdx);
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]); % [1, N_z, 1]
            lin_low=aprime_low+zidxoffset;
            lin_up =aprime_up +zidxoffset;
            EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z,N_e]);
            EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z,N_e]);
            EVnext_atpolicy=a2pPrb.*EV_low+(1-a2pPrb).*EV_up;
            V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
        end
    end
end

%% Reshape V out of Kron form
if N_z==0 && N_e==0
    V=reshape(V, [n_a, N_j]);
elseif N_z==0 && N_e>0
    V=reshape(V, [n_a, vfoptions.n_e, N_j]);
elseif N_z>0 && N_e==0
    V=reshape(V, [n_a, n_z, N_j]);
else
    V=reshape(V, [n_a, n_z, vfoptions.n_e, N_j]);
end


end
