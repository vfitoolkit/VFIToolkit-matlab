function StationaryDist=StationaryDist_FHorz_ExpAssetu(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Setup related to experience asset
l_a=length(n_a);
n_d2=n_d(end);
% Split endogenous assets into the standard ones and the experience asset
if l_a==1
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


if isfield(simoptions,'n_u')
    n_u=simoptions.n_u;
else
    error('To use an experience assetu you must define vfoptions.n_u')
end
if isfield(simoptions,'u_grid')
    u_grid=simoptions.u_grid;
else
    error('To use an experience assetu you must define vfoptions.u_grid')
end
if isfield(simoptions,'pi_u')
    pi_u=simoptions.pi_u;
else
    error('To use an experience assetu you must define vfoptions.pi_u')
end


% Make sure u_grid and pi_u are on gpu
u_grid=gpuArray(u_grid);
pi_u=gpuArray(pi_u);
simoptions.u_grid=gpuArray(simoptions.u_grid); % needed to by some subfns

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
l_u=length(n_u);
temp=getAnonymousFnInputNames(simoptions.aprimeFn);
if length(temp)>(l_d2+l_a2+l_u)
    aprimeFnParamNames={temp{l_d2+l_a2+l_u+1:end}}; % the first inputs will always be (d2,a2,u)
else
    aprimeFnParamNames={};
end

%%
if n_z(1)==0
    StationaryDist=StationaryDist_FHorz_ExpAssetu_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
    return
end

%%
l_d=length(n_d);

N_a=prod(n_a);
N_z=prod(n_z);
N_u=prod(n_u);


%%
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
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


%%
% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy_a2prime=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_ze,N_u,2,N_j,'gpuArray'); % probabilities of grid points
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAssetu_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z,n_u, d_grid, a2_grid,u_grid, aprimeFnParamsVec);
    % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z,N_u]
    % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
    Policy_a2prime(:,:,:,1,jj)=a2primeIndexes; % lower grid point
    Policy_a2prime(:,:,:,2,jj)=a2primeIndexes+1; % upper grid point
    % Encode the u probabilities (pi_u) into the PolicyProbs
    PolicyProbs(:,:,:,1,jj)=a2primeProbs.*shiftdim(pi_u,-2); % lower grid point probability (and probability of u)
    PolicyProbs(:,:,:,2,jj)=(1-a2primeProbs).*shiftdim(pi_u,-2); % upper grid point probability (and probability of u)
end

if l_a==1 % just experienceassetu
    Policy_aprime=Policy_a2prime;
elseif l_a==2 % one other asset, then experience assetu
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*(Policy_a2prime(:,:,:,1,jj)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*Policy_a2prime(:,:,:,1,jj); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a==3 % two other assets, then experience assetu
    Policy_aprime(:,:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+2,:,:,:)-1,[N_a,N_ze,1,1,N_j])+n_a(1)*n_a(2)*(Policy_a2prime(:,:,:,1,jj)-1);
    Policy_aprime(:,:,:,2,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_ze,1,1,N_j])+n_a(1)*reshape(Policy(l_d+2,:,:,:)-1,[N_a,N_ze,1,1,N_j])+n_a(1)*n_a(2)*Policy_a2prime(:,:,:,1,jj); % Note: upper grid point minus 1 is anyway just lower grid point
elseif l_a>3
    error('Not yet implemented experienceassetu with length(n_a)>3')
end

%%
% Note: N_z=0 is a different code
if isfield(simoptions,'n_e')
    StationaryDist=StationaryDist_FHorz_Iteration_uProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_e,N_u,N_j,pi_z_J,simoptions.pi_e_J,Parameters);
else
    StationaryDist=StationaryDist_FHorz_Iteration_uProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_u,N_j,pi_z_J,Parameters);
end


if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_ze,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_ze,N_j]);
end

end
