function varargout=ValueFnFromPolicy_FHorz_ExpAssetz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
% Compute V from a given Policy when the model has experienceassetz (vfoptions.experienceassetz>=1).
% experienceassetz: a2prime = aprimeFn(d_expasset, a2, z) -- depends on the Markov shock z.
% Requires N_z>0; may have e too.

%% Dispatch to SemiExo subfn if n_semiz>0
if prod(vfoptions.n_semiz)>0
    V=ValueFnFromPolicy_FHorz_ExpAssetz_SemiExo(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end
%% Dispatch to GI subfn if gridinterplayer==1
if vfoptions.gridinterplayer==1
    V=ValueFnFromPolicy_FHorz_ExpAssetz_GI(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
    varargout={V};
    return
end

%% Setup
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
    error('ValueFnFromPolicy_FHorz_ExpAssetz: experienceassetz requires N_z>0 (aprimeFn depends on z)')
end
if N_d==0
    error('ValueFnFromPolicy_FHorz_ExpAssetz: experienceassetz requires at least one decision variable (the one driving a2prime)')
end
l_d=length(n_d);
l_z=length(n_z);

% Split a into a1 (standard) and a2 (experience asset)
if isscalar(n_a)
    n_a1=0;
    N_a1=0;
    error('ValueFnFromPolicy_FHorz_ExpAssetz: case with no a1 (experience asset as only asset) not yet implemented')
else
    n_a1=n_a(1:end-1);
    N_a1=prod(n_a1);
end
n_a2=n_a(end);
% N_a2=prod(n_a2);
% a1_grid=a_grid(1:sum(n_a1));
a2_grid=a_grid(sum(n_a1)+1:end);
l_a1=length(n_a1);
l_a2=length(n_a2);

% Which d affects the experience asset (default: last d only)
if isfield(vfoptions,'l_dexperienceassetz')
    l_d2=vfoptions.l_dexperienceassetz;
else
    l_d2=1;
end
whichisdforexpasset=(l_d-l_d2+1):l_d;
% n_d2=n_d(end-l_d2+1:end);

% aprimeFnParamNames: first inputs are (d_expasset..., a2, z)
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+1:end}};
else
    aprimeFnParamNames={};
end

% Combined shock dim (z always present here)
if N_e==0
    N_ze=N_z;
else
    N_ze=N_z*N_e;
end

ReturnFnParamNames=ReturnFnParamNamesFn(ReturnFn,n_d,n_a,n_z,N_j,vfoptions,Parameters);

a_gridvals=CreateGridvals(n_a,a_grid,1);

%% PolicyValues (PolicyInd2Val_FHorz handles experienceassetz internally)
PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions,1);
l_daprime=size(PolicyValues,1); % = l_d + l_a1
% PolicyValues shape: [l_daprime, N_a, N_ze, N_j]
PolicyValuesPermute=permute(PolicyValues,[2,3,1,4]); % [N_a, N_ze, l_daprime, N_j]

%% Reshape Policy to canonical Kron form
Policy=reshape(Policy,[l_d+l_a1, N_a, N_ze, N_j]);

%% Build a1prime joint index across N_a1 dims (from Policy)
Policy_a1prime=ones(N_a, N_ze, N_j, 'gpuArray');
cumprods_a1=[1, cumprod(n_a1(1:end-1))];
for ii=1:l_a1
    comp=shiftdim(Policy(l_d+ii, :, :, :),1);
    Policy_a1prime=Policy_a1prime+cumprods_a1(ii)*(comp-1);
end

%% Joint z+e gridvals for ReturnFn when both present
if N_e>0
    ze_gridvals_J=zeros(N_z*N_e, length(n_z)+length(vfoptions.n_e), N_j, 'gpuArray');
    for jj=1:N_j
        ze_gridvals_J(:,:,jj)=[repmat(z_gridvals_J(:,:,jj),N_e,1), repelem(vfoptions.e_gridvals_J(:,:,jj),N_z,1)];
    end
end

%% V allocation
if N_e==0
    V=zeros(N_a, N_z, N_j, 'gpuArray');
else
    V=zeros(N_a, N_z, N_e, N_j, 'gpuArray');
end

%% Backward iteration
for reverse_j=0:N_j-1
    jj=N_j-reverse_j;

    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);

    % Step 1: a2primeIndex, a2primeProbs (helper expects Policy shape [L, N_a, N_z]; loop over e when N_e>0)
    if N_e==0
        Policy_jj=Policy(:,:,:,jj); % [l_d+l_a1, N_a, N_z]
        [a2primeIndex, a2primeProbs]=CreateaprimePolicyExperienceAssetz(Policy_jj, aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
        % shape [N_a, N_z]
    else
        Policy_zE=reshape(Policy(:,:,:,jj),[l_d+l_a1, N_a, N_z, N_e]);
        a2primeIndex=zeros(N_a, N_z, N_e, 'gpuArray');
        a2primeProbs=zeros(N_a, N_z, N_e, 'gpuArray');
        for e_idx=1:N_e
            [a2pi, a2pp]=CreateaprimePolicyExperienceAssetz(Policy_zE(:,:,:,e_idx), aprimeFn, whichisdforexpasset, n_d, n_a1, n_a2, n_z, 0,N_z,0, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
            a2primeIndex(:,:,e_idx)=a2pi;
            a2primeProbs(:,:,e_idx)=a2pp;
        end
    end

    % Step 2: ReturnFn at policy
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,ReturnFnParamNames,jj);
    if N_e==0
        F_jj=EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, n_z, a_gridvals, z_gridvals_J(:,:,jj));
        % [N_a, N_z]
    else
        F_jj=reshape(EvalFnOnAgentDist_Grid(ReturnFn, FnToEvaluateParamsCell, PolicyValuesPermute(:,:,:,jj), l_daprime, n_a, [n_z,vfoptions.n_e], a_gridvals, ze_gridvals_J(:,:,jj)), [N_a, N_z, N_e]);
    end

    if jj==N_j
        if N_e==0
            V(:,:,jj)=F_jj;
        else
            V(:,:,:,jj)=F_jj;
        end
    else
        beta=prod(gpuArray(CreateVectorFromParams(Parameters,DiscountFactorParamNames,jj)));

        % Step 3: EVnext(anext, z_from) -- integrate e' (iid) then z' (markov)
        if N_e==0
            EVnext=V(:,:,jj+1)*pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        else
            EVnext=sum(V(:,:,:,jj+1) .* shiftdim(vfoptions.pi_e_J(:,jj), -2), 3); % [N_a, N_z, 1]
            EVnext=reshape(EVnext,[N_a,N_z]) * pi_z_J(:,:,jj)'; % [N_a, N_z]
            EVnext(isnan(EVnext))=0;
        end

        % Step 4: interpolated lookup
        if N_e==0
            a1p=Policy_a1prime(:,:,jj); % [N_a, N_z]
            aprime_low=a1p+N_a1*(a2primeIndex-1);
            aprime_up =a1p+N_a1*(a2primeIndex);
            zidxoffset=N_a*gpuArray(0:N_z-1); % [1, N_z]
            lin_low=aprime_low+zidxoffset;
            lin_up =aprime_up +zidxoffset;
            EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z]);
            EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z]);
            EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;
            V(:,:,jj)=F_jj+beta*EVnext_atpolicy;
        else
            a1p=reshape(Policy_a1prime(:,:,jj),[N_a, N_z, N_e]);
            aprime_low=a1p+N_a1*(a2primeIndex-1);
            aprime_up =a1p+N_a1*(a2primeIndex);
            zidxoffset=reshape(N_a*gpuArray(0:N_z-1),[1,N_z,1]);
            lin_low=aprime_low+zidxoffset;
            lin_up =aprime_up +zidxoffset;
            EV_low=reshape(EVnext(lin_low(:)),[N_a,N_z,N_e]);
            EV_up =reshape(EVnext(lin_up(:)), [N_a,N_z,N_e]);
            EVnext_atpolicy=a2primeProbs.*EV_low+(1-a2primeProbs).*EV_up;
            V(:,:,:,jj)=F_jj+beta*EVnext_atpolicy;
        end
    end
end

%% Reshape V out of Kron form
if N_e==0
    V=reshape(V, [n_a, n_z, N_j]);
else
    V=reshape(V, [n_a, n_z, vfoptions.n_e, N_j]);
end



varargout={V};

end
