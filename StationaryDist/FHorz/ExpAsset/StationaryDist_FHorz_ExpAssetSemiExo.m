function StationaryDist=StationaryDist_FHorz_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,pi_z_J,Parameters,simoptions)

%% Experience asset and semi-exogenous state
n_d3=n_d(end); % decision variable that controls semi-exogenous state
n_d2=n_d(end-1); % decision variables that controls experience asset
if length(n_d)>2
    n_d1=n_d(1:end-2);
    l_d1=length(n_d1);
else
    % n_d1=0;
    l_d1=0;
end
l_d2=length(n_d2); % wouldn't be here if no d2
l_d3=length(n_d3); % wouldn't be here if no d3


%% Setup related to experience asset
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
% l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end


%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);

if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

%%
if N_z==0
    if N_e==0
        n_bothze=simoptions.n_semiz;
        N_bothze=N_semiz;
    else
        n_bothze=[simoptions.n_semiz,simoptions.n_e];
        N_bothze=N_semiz*N_e;
    end
else
    if N_e==0
        n_bothze=[simoptions.n_semiz,n_z];
        N_bothze=N_semiz*N_z;
    else
        n_bothze=[simoptions.n_semiz,n_z,simoptions.n_e];
        N_bothze=N_semiz*N_z*N_e;
    end
end

jequaloneDist=gpuArray(jequaloneDist); % make sure it is on gpu
jequaloneDist=reshape(jequaloneDist,[N_a*N_bothze,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_bothze,N_j]);


%% expasset transitions
% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_aprime=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d)-1;  % is just saying which is the decision variable that influences the experience asset (it is the 'second last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_bothze, d_grid, a2_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_bothze]
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


%% Policy_dsemiexo

% d3 is the variable relevant for the semi-exogenous asset. 
if l_d3==1
    Policy_dsemiexo=shiftdim(Policy(l_d1+l_d2+1,:,:,:),1);
elseif l_d3==2
    Policy_dsemiexo=shiftdim(Policy(l_d1+l_d2+1,:,:,:)+n_d(l_d1+l_d2+1)*Policy(l_d1+l_d2+2,:,:,:),1);
elseif l_d3==3
    Policy_dsemiexo=shiftdim(Policy(l_d1+l_d2+1,:,:,:)+n_d(l_d1+l_d2+1)*Policy(l_d1+l_d2+2,:,:,:)+n_d(l_d1+l_d2+1)*n_d(l_d1+l_d2+2)*Policy(l_d1+l_d2+3,:,:,:),1); 
elseif l_d3==4
    Policy_dsemiexo=shiftdim(Policy(l_d1+l_d2+1,:,:,:)+n_d(l_d1+l_d2+1)*Policy(l_d1+l_d2+2,:,:,:)+n_d(l_d1+l_d2+1)*n_d(l_d1+l_d2+2)*Policy(l_d1+l_d2+3,:,:,:)+n_d(l_d1+l_d2+1)*n_d(l_d1+l_d2+2)*n_d(l_d1+l_d2+3)*Policy(l_d1+l_d2+4,:,:,:),1);
end


%%
if simoptions.gridinterplayer==0
    if N_z==0 && N_e==0
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_j,pi_semiz_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    elseif N_e==0 % just z
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_e,N_j,pi_semiz_J,simoptions.pi_e_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    else % both z and e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    end
elseif simoptions.gridinterplayer==1
    % (a,z,2,j)
    Policy_aprime=repmat(Policy_aprime,1,1,2,1);
    PolicyProbs=repmat(PolicyProbs,1,1,2,1);
    % Policy_aprime(:,:,1:2,:) lower grid point for a1 is unchanged 
    Policy_aprime(:,:,3:4,:)=Policy_aprime(:,:,3:4,:)+1; % add one to a1, to get upper grid point

    aprimeProbs_upper=reshape(shiftdim((Policy(end,:,:,:)-1)/(simoptions.ngridinterp+1),1),[N_a,N_bothze,1,N_j]); % probability of upper grid point (from L2 index)
    PolicyProbs(:,:,1:2,:)=PolicyProbs(:,:,1:2,:).*(1-aprimeProbs_upper); % lower a1
    PolicyProbs(:,:,3:4,:)=PolicyProbs(:,:,3:4,:).*aprimeProbs_upper; % upper a1

    if N_z==0 && N_e==0
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,4,N_a,N_semiz,N_j,pi_semiz_J,Parameters);    
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    elseif N_e==0 % just z
        Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1);
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprimez,PolicyProbs,4,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    elseif N_z==0 % just e
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,4,N_a,N_semiz,N_e,N_j,pi_semiz_J,simoptions.pi_e_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    else % both z and e
        Policy_aprimez=Policy_aprime+repmat(N_a*gpuArray(0:1:N_z-1),1,N_e);
        StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_e_raw(jequaloneDist,AgeWeightParamNames,Policy_dsemiexo,Policy_aprimez,PolicyProbs,4,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,simoptions.pi_e_J,Parameters);
        StationaryDist=gpuArray(StationaryDist); % NEED TO MOVE TAN IMPROVEMENT TO GPU
    end
end



if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_bothze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_bothze,N_j]);
end

end
