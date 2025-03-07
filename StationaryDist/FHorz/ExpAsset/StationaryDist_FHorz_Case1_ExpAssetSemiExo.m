function StationaryDist=StationaryDist_FHorz_Case1_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Experience asset and semi-exogenous state
n_d3=n_d(end); % decision variable that controls semi-exogenous state
n_d2=n_d(end-1); % decision variables that controls experience asset
if length(n_d)>2
    n_d1=n_d(1:end-2);
else
    n_d1=0;
end

%% Setup related to experience asset
% Split endogenous assets into the standard ones and the experience asset
if length(n_a)==1
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
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end


%% Setup related to semi-exogenous state (an exogenous state whose transition probabilities depend on a decision variable)
if ~isfield(simoptions,'n_semiz')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.n_semiz')
end
if ~isfield(simoptions,'semiz_grid')
    error('When using simoptions.SemiExoShockFn you must declare simoptions.semiz_grid')
end
d3_grid=gpuArray(d_grid(sum(n_d1)+sum(n_d2)+1:end));
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
l_semiz=length(simoptions.n_semiz);
temp=getAnonymousFnInputNames(simoptions.SemiExoStateFn);
if length(temp)>(1+l_semiz+l_semiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{1+l_semiz+l_semiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
else
    SemiExoStateFnParamNames={};
end
N_semiz=prod(simoptions.n_semiz);
pi_semiz_J=zeros(N_semiz,N_semiz,n_d3,N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d3,simoptions.n_semiz,d3_grid,gpuArray(simoptions.semiz_grid),simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end


%%
if n_z(1)==0
    error('Not yet implemented n_z=0 with experienceasset, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

n_bothz=[simoptions.n_semiz,n_z];
N_bothz=N_semiz*N_z;

%%
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_bothz*N_e,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothz*N_e,N_j]);
    n_bothze=[simoptions.n_semiz,n_z,simoptions.n_e];
    N_bothze=N_bothz*N_e;
else
    jequaloneDistKron=reshape(jequaloneDist,[N_a*N_bothz,1]);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_bothz,N_j]);
    n_bothze=[simoptions.n_semiz,n_z];
    N_bothze=N_bothz;
end
% NOTE: have rolled e into z

%% expasset transitions
% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_a2prime=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_bothze,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d)-1;  % is just saying which is the decision variable that influences the experience asset (it is the 'second last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_bothze, d_grid, a2_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
    Policy_a2prime(:,:,1,jj)=a2primeIndexes; % lower grid point
    Policy_a2prime(:,:,2,jj)=a2primeIndexes+1; % upper grid point
    PolicyProbs(:,:,1,jj)=a2primeProbs;
    PolicyProbs(:,:,2,jj)=1-a2primeProbs;
end

if l_a==1 % just experienceasset
    Policy_aprime=Policy_a2prime;
elseif l_a==2 % one other asset, then experience asset
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*(Policy_a2prime(:,:,:,1,jj)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*Policy_a2prime(:,:,:,1,jj); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then experience asset
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+2,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1,jj)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+2,:,:,:),[N_a,N_bothze,1,1,N_j])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1,jj); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Not yet implemented experienceassetu with length(n_a)>3')
end

%%
N_d1=prod(n_d1);
N_d2=prod(n_d2);

% % Only d variables we need are the ones for the semi-exogenous asset
% Policy_dsemiexo=shiftdim(PolicyKron(l_d,:,:,:); % The last d variable is the relevant one for the semi-exogenous asset. 
% Rather than actually create Policy_dsemiexo we just pass this as the input to the simulation/iteration commands

% Note: N_z=0 is a different code
if isfield(simoptions,'n_e')
    error('Have not yet impelmented N_e in StationaryDist_FHorz_Case1_ExpAssetSemiExo (contact me)')
else % no e
    StationaryDist=StationaryDist_FHorz_Case1_Iteration_SemiExo_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,shiftdim(Policy(l_d,:,:,:),1),Policy_aprime,PolicyProbs,N_d1,N_d2,N_a,N_z,N_semiz,N_j,pi_z_J,pi_semiz_J,Parameters,simoptions); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_bothze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_bothze,N_j]);
end

end
