function StationaryDist=StationaryDist_FHorz_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z,N_j,pi_semiz_J,pi_z_J,Parameters,simoptions)

%% Experience asset and semi-exogenous state
n_d3=n_d(end); % decision variable that controls semi-exogenous state
n_d2=n_d(end-1); % decision variables that controls experience asset
if length(n_d)>2
    n_d1=n_d(1:end-2);
    l_d1=length(n_d1);
else
    n_d1=0;
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
if n_z(1)==0
    error('Not yet implemented n_z=0 with experienceasset, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);

% n_bothz=[simoptions.n_semiz,n_z];
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
    Policy_aprime=zeros([N_a,N_bothze,1,N_j]);
    Policy_aprime(:,:,1,:)=gather(reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,N_j])+n_a(1)*(Policy_a2prime(:,:,1,:)-1));
    Policy_aprime(:,:,2,:)=gather(reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,N_j])+n_a(1)*Policy_a2prime(:,:,1,:)); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then experience asset
    Policy_aprime=zeros([N_a,N_bothze,1,N_j]);
    Policy_aprime(:,:,1,:)=gather(reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,N_j])+n_a(1)*reshape(Policy(l_d+2,:,:,:)-1,[N_a,N_bothze,1,N_j])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,1,:)-1));
    Policy_aprime(:,:,2,:)=gather(reshape(Policy(l_d+1,:,:,:),[N_a,N_bothze,1,N_j])+n_a(1)*reshape(Policy(l_d+2,:,:,:)-1,[N_a,N_bothze,1,N_j])+n_a(1)*n_a(2)*Policy_a2prime(:,:,1,:)); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Not yet implemented experienceassetu with length(n_a)>3')
end

%%

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

% Note: N_z=0 is a different code
if isfield(simoptions,'n_e')
    StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,Parameters);
else % no e
    StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters);
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
