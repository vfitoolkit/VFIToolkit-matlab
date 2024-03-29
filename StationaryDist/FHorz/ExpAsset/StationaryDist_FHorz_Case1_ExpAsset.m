function StationaryDist=StationaryDist_FHorz_Case1_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions)

%% Setup related to experience asset
n_d2=n_d(end);
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

l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);


%%
if n_z(1)==0
    StationaryDist=StationaryDist_FHorz_Case1_ExpAsset_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,d_grid,a2_grid,Parameters,simoptions);
    return
end

if isfield(simoptions,'n_e')
    if n_z(1)==0
        error('Not yet implemented n_z=0 with n_e and experienceasset, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
    else
%         StationaryDist=StationaryDist_FHorz_Case1_ExpAsset_e(jequaloneDist,AgeWeightParamNames,Policy,n_d1,n_d2,n_a,n_z,N_j,pi_z,aprimeFn,Parameters,simoptions);
    end
    return
end


%%
jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);
if simoptions.parallel~=2 && simoptions.parallel~=4
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
    pi_z_J=gather(pi_z_J);
end

% Policy is currently about d and a1prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
Policy_aprime=zeros(N_a,N_z,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
PolicyProbs=zeros(N_a,N_z,2,N_j,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,jj);
    [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,:,jj),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, d_grid, a2_grid, aprimeFnParamsVec);
    if l_a==1
        Policy_aprime(:,:,1,jj)=a2primeIndexes;
        Policy_aprime(:,:,2,jj)=a2primeIndexes+1;
        PolicyProbs(:,:,1,jj)=a2primeProbs;
        PolicyProbs(:,:,2,jj)=1-a2primeProbs;
    elseif l_a==2 % experience asset and one other asset
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(a2primeIndexes-1);
        Policy_aprime(:,:,2,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(a2primeIndexes-1+1);
        PolicyProbs(:,:,1,jj)=a2primeProbs;
        PolicyProbs(:,:,2,jj)=1-a2primeProbs;
    elseif l_a==3 % experience asset and two other assets
        Policy_aprime(:,:,1,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(a2primeIndexes-1);
        Policy_aprime(:,:,2,jj)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(a2primeIndexes-1+1);
        PolicyProbs(:,:,1,jj)=a2primeProbs;
        PolicyProbs(:,:,2,jj)=1-a2primeProbs;       
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end
end


if simoptions.iterate==0
    PolicyProbs=gather(PolicyProbs); % simulation is always with cpu
    Policy_aprime=gather(Policy_aprime);
    if simoptions.parallel>=3
        % Sparse matrix is not relevant for the simulation methods, only for iteration method
        simoptions.parallel=2; % will simulate on parallel cpu, then transfer solution to gpu
    end
    StationaryDist=StationaryDist_FHorz_Case1_Simulation_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z_J, Parameters, simoptions);
elseif simoptions.iterate==1
    StationaryDist=StationaryDist_FHorz_Case1_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z_J,Parameters); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.parallel==2
    StationaryDist=gpuArray(StationaryDist); % move output to gpu
end
if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDist,[n_a,n_z,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
end

end
