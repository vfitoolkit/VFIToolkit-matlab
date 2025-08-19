function StationaryDist=StationaryDist_FHorz_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end);
% Split endogenous assets into the standard ones and the experience asset
if isscalar(n_a)
    n_a1=0;
else
    n_a1=n_a(1:end-1);
end
n_a2=n_a(end); % n_a2 is the experience asset

if ~isfield(simoptions,'aprimeFn')
    error('To use an experience asset you must define simoptions.aprimeFn')
end
if isfield(simoptions,'a_grid')
    % a_grid=simoptions.a_grid;
    % a1_grid=simoptions.a_grid(1:sum(n_a1));
    a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
else
    error('To use an experience asset you must define simoptions.a_grid')
end
if isfield(simoptions,'d_grid')
    d_grid=simoptions.d_grid;
else
    error('To use an experience asset you must define simoptions.d_grid')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

%%
if n_z(1)==0 && N_e==0
    StationaryDist=StationaryDist_FHorz_ExpAsset_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,d_grid,a2_grid,Parameters,simoptions);
    return
end

%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);


%%
if isfield(simoptions,'n_e')
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z*N_e,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z*N_e,N_j]);
    n_ze=[n_z,simoptions.n_e];
    N_ze=N_z*N_e;
else
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    n_ze=n_z;
    N_ze=N_z;
end
% NOTE: have rolled e into z
jequaloneDistKron=gpuArray(jequaloneDistKron); % make sure it is on gpu

%%

% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_aprime=zeros(N_a,N_ze,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_ze,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_ze, d_grid, a2_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a==1
        Policy_aprime(:,:,1,jj)=aprimeIndexes;
        Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
    elseif l_a==2 % experience asset and one other asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
    elseif l_a==3 % experience asset and two other assets
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:2));
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end
    PolicyProbs(:,:,1,jj)=aprimeProbs;
    PolicyProbs(:,:,2,jj)=1-aprimeProbs;
end
    

%%
if simoptions.gridinterplayer==0
    % Note: N_z=0 && N_e=0 is a different code
    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    elseif N_z==0 % just e
        error('Have not yet impelmented N_e in StationaryDist_FHorz_Case1_ExpAsset (contact me)')
    else % both z and e
        error('Have not yet impelmented N_e in StationaryDist_FHorz_Case1_ExpAsset (contact me)')
    end
elseif simoptions.gridinterplayer==1
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:2,:) lower grid point for a1 is unchanged 
    Policy_aprime(:,:,3:4,:)=Policy_aprime(:,:,3:4,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_ze,1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1:2,:)=PolicyProbs(:,:,1:2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,3:4,:)=PolicyProbs(:,:,3:4,:).*aprimeProbs_upper; % upper a1

    % Note: N_z=0 && N_e=0 is a different code
    if N_e==0 % just z
        Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1);
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprimez,PolicyProbs,4,N_a,N_z,N_j,pi_z_J,Parameters);
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,4,N_a,N_e,N_j,pi_e_J,Parameters);
    else % both z and e
        Policy_aprimez=Policy_aprime+repmat(N_a*gpuArray(0:1:N_z-1),1,N_e);
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprimez,PolicyProbs,4,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,Parameters);
    end
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
