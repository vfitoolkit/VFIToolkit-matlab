function  StationaryDist=StationaryDist_InfHorz_InheritAsset(StationaryDistKron,Policy,n_d,n_a,n_z,pi_z,Parameters,simoptions)

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
    error('To use an inheritance asset you must define simoptions.aprimeFn')
end
if isfield(simoptions,'a_grid')
    % a_grid=simoptions.a_grid;
    % a1_grid=simoptions.a_grid(1:sum(n_a1));
    a2_grid=simoptions.a_grid(sum(n_a1)+1:end);
else
    error('To use an inheritance asset you must define simoptions.a_grid')
end
if isfield(simoptions,'d_grid')
    d_grid=simoptions.d_grid;
else
    error('To use an inheritance asset you must define simoptions.d_grid')
end
if isfield(simoptions,'z_grid')
    z_grid=simoptions.z_grid;
else
    error('To use an inheritance asset you must define simoptions.z_grid')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_z=length(n_z);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+2*l_z)
    aprimeFnParamNames={temp{l_d2+2*l_z+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end


%%
l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

z_gridvals=CreateGridvals(n_z,z_grid,1);

%%
StationaryDistKron=reshape(StationaryDistKron,[N_a*N_z,1]);
Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);

N_zprime=N_z; % just to make code easier to read

%% Policy is currently about d and a1prime. Convert it to being about aprime as that is what we need for simulation.
Policy_a2prime=zeros(N_a,N_z,N_zprime,2,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_z,N_zprime,2,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforinheritasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
[a2primeIndexes, a2primeProbs]=CreateaprimePolicyInheritanceAsset(Policy,simoptions.aprimeFn, whichisdforinheritasset, n_d, n_a1,n_a2, n_z, n_z, gpuArray(d_grid), a2_grid, gpuArray(z_gridvals), gpuArray(z_gridvals), aprimeFnParamsVec);
% Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_zprime]
% Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
Policy_a2prime(:,:,:,1)=a2primeIndexes; % lower grid point
Policy_a2prime(:,:,:,2)=a2primeIndexes+1; % upper grid point
PolicyProbs(:,:,:,1)=a2primeProbs; % probability of lower grid point
PolicyProbs(:,:,:,2)=1-a2primeProbs; % probability of upper grid point

if l_a==1 % just inheritanceasset
    Policy_aprime=Policy_a2prime;
elseif l_a==2 % one other asset, then inheritance asset
    Policy_aprime(:,:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*(Policy_a2prime(:,:,:,1)-1);
    Policy_aprime(:,:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*Policy_a2prime(:,:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then inheritance asset
    Policy_aprime(:,:,:,1)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*reshape(Policy(l_d+2,:,:),[N_a,N_z,1])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1)-1);
    Policy_aprime(:,:,:,2)=reshape(Policy(l_d+1,:,:),[N_a,N_z,1])+n_a(1)*reshape(Policy(l_d+2,:,:),[N_a,N_z,1])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Not yet implemented inheritanceasset with length(n_a)>3')
end


%%
% Policy depends on zprime
StationaryDist=StationaryDist_InfHorz_Iteration_zprime_nProbs_raw(StationaryDistKron,Policy_aprime,PolicyProbs,2,N_a,N_z,pi_z,simoptions); % zero is n_d, because we already converted Policy to only contain aprime

StationaryDist=gpuArray(StationaryDist); % move output to gpu

if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_z]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_z]);
end

















end
