function StationaryDist=StationaryDist_FHorz_ExpAssetz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,z_gridvals_J,pi_z_J,Parameters,simoptions)

if isUnderlyingType(jequaloneDist,'single')
    precision='single';
    precision_cast=@(x) single(x);
    precision_index='int32';
    precision_index_cast=@(x) int32(x);
else
    precision='double';
    precision_cast=@(x) double(x);
    precision_index='double';
    precision_index_cast=@(x) x;
end

%% Setup related to experience asset
n_d2=n_d(end-simoptions.l_dexperienceassetz+1:end);
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
l_z=length(n_z);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_z)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+1:end}}; % the first inputs will always be (d2,a2,z)
else
    aprimeFnParamNames={};
end

N_e=prod(simoptions.n_e);
N_z=prod(n_z);

%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);

%%
if N_e==0
    n_ze=n_z;
    N_ze=N_z;
else
    n_ze=[n_z,simoptions.n_e];
    N_ze=N_z*N_e;
end

jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_ze,N_j]);

%% expassetz transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_aprime=zeros(N_a,N_ze,2,N_j,precision_index,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_ze,2,N_j,precision,'gpuArray'); % The third dimension is lower/upper grid point
whichisdforexpassetz=length(n_d)-simoptions.l_dexperienceassetz+1:length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj,precision);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAssetz(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpassetz, n_d, n_a1,n_a2, n_z, 0,N_z,N_e, d_grid, a2_grid, z_gridvals_J(:,:,jj), aprimeFnParamsVec);
    aprimeIndexes=precision_index_cast(aprimeIndexes);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_ze]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).

    if l_a==1 % just experience asset
        Policy_aprime(:,:,1,jj)=aprimeIndexes;
        Policy_aprime(:,:,2,jj)=aprimeIndexes+1;
    elseif l_a==2 % one other asset, then experience asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+n_a(1);
    elseif l_a==3 % two other assets, then experience asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:2));
    elseif l_a==4 % three other assets, then experience asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1)+prod(n_a(1:3))*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:3));
    elseif l_a==5 % four other assets, then experience asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(shiftdim(Policy(l_d+3,:,:,jj),1)-1)+prod(n_a(1:3))*(shiftdim(Policy(l_d+4,:,:,jj),1)-1)+prod(n_a(1:4))*(aprimeIndexes-1);
        Policy_aprime(:,:,2,jj)=Policy_aprime(:,:,1,jj)+prod(n_a(1:4));
    else
        error('Not yet implemented experience asset with length(n_a)>5')
    end

    PolicyProbs(:,:,1,jj)=aprimeProbs;
    PolicyProbs(:,:,2,jj)=1-aprimeProbs;
end



%%
if simoptions.gridinterplayer==0
    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_z,N_j,pi_z_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,2,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
    end
elseif simoptions.gridinterplayer==1
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:2,:) lower grid point for a1 is unchanged
    Policy_aprime(:,:,3:4,:)=Policy_aprime(:,:,3:4,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim(double(Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_ze,1,N_j]); % probability of upper grid point (from L2 index; end-1 because end is now L2flag)
    PolicyProbs(:,:,1:2,:)=PolicyProbs(:,:,1:2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,3:4,:)=PolicyProbs(:,:,3:4,:).*aprimeProbs_upper; % upper a1

    if N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,4,N_a,N_z,N_j,pi_z_J,Parameters);
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_aprime,PolicyProbs,4,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
    end
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
% else
    % If 1 then leave output in Kron form
    % StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
