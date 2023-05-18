function StationaryDist=StationaryDist_FHorz_Case1_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,aprimeFn,Parameters,simoptions)

%% Check for the age weights parameter, and make sure it is a row vector
if size(Parameters.(AgeWeightParamNames{1}),2)==1 % Seems like column vector
    Parameters.(AgeWeightParamNames{1})=Parameters.(AgeWeightParamNames{1})'; 
    % Note: assumed there is only one AgeWeightParamNames
end

%% Check that the age one distribution is of mass one
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
end

%% Setup related to experience asset
if isfield(simoptions,'aprimeFn')
    aprimeFn=simoptions.aprimeFn;
else
    error('To use an experience asset you must define simoptions.aprimeFn')
end
if ~isfield(simoptions,'a_grid')
    error('To use an experience asset you must define simoptions.a_grid')
end

n_d2=n_d(end);
n_a2=n_a(end);

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(aprimeFn);
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
n_d1=n_d(1:end-1);
n_d2=n_d(end); % n_d2 is the decision variable that influences the transition probabilities of the semi-exogenous state
% d1_grid=simoptions.d_grid(1:sum(n_d1));
d2_grid=gpuArray(simoptions.d_grid(sum(n_d1)+1:end));
% Create the transition matrix in terms of (d,zprime,z) for the semi-exogenous states for each age
l_semiz=length(simoptions.n_semiz);
temp=getAnonymousFnInputNames(simoptions.SemiExoStateFn);
if length(temp)>(1+l_semiz+l_semiz) % This is largely pointless, the SemiExoShockFn is always going to have some parameters
    SemiExoStateFnParamNames={temp{1+l_semiz+l_semiz+1:end}}; % the first inputs will always be (d,semizprime,semiz)
else
    SemiExoStateFnParamNames={};
end
N_semiz=prod(simoptions.n_semiz);
pi_semiz_J=zeros(N_semiz,N_semiz,n_d2,N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,simoptions.n_semiz,d2_grid,simoptions.semiz_grid,simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end


%%
if isfield(simoptions,'n_e')
    if n_z(1)==0
        error('Not yet implemented n_z=0 with n_e and experienceasset, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
    else
%         StationaryDist=StationaryDist_FHorz_Case1_ExpAsset_e(jequaloneDist,AgeWeightParamNames,Policy,n_d1,n_d2,n_a,n_z,N_j,pi_z,aprimeFn,Parameters,simoptions);
    end
    return
end

if n_z(1)==0
    error('Not yet implemented n_z=0 with experienceasset, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

l_d=length(n_d);
l_a=length(n_a);

N_a=prod(n_a);
N_z=prod(n_z);

N_bothz=N_z*N_semiz;

if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=3-(gpuDeviceCount>0); % 3 (sparse) if cpu, 2 if gpu
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.outputkron=0; % If 1 then leave output in Kron form
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=3-(gpuDeviceCount>0); % 3 (sparse) if cpu, 2 if gpu
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if isfield(simoptions,'outputkron')==0
        simoptions.outputkron=0; % If 1 then leave output in Kron form
    end
end

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_bothz,1]);
if simoptions.parallel~=2 && simoptions.parallel~=4
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
    pi_z=gather(pi_z);
end

% Policy is currently about d and a2prime. Convert it to being about aprime
% as that is what we need for simulation, and we can then just send it to standard Case1 commands.
Policy=reshape(Policy,[size(Policy,1),N_a,N_bothz,N_j]);
Policy_aprime=zeros(N_a,N_bothz,N_j,2,'gpuArray'); % The fourth dimension is lower/upper grid point
PolicyProbs=zeros(N_a,N_bothz,N_j,2,'gpuArray'); % The fourth dimension is lower/upper grid point
whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
for jj=1:N_j
    aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames,N_j);
    [aprimeIndexes, aprimeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy(:,:,:,jj),aprimeFn, whichisdforexpasset, n_a, N_bothz, gpuArray(simoptions.a_grid), aprimeFnParamsVec);
    if l_a==1
        Policy_aprime(:,:,jj,1)=aprimeIndexes;
        Policy_aprime(:,:,jj,2)=aprimeIndexes+1;
        PolicyProbs(:,:,jj,1)=aprimeProbs;
        PolicyProbs(:,:,jj,2)=1-aprimeProbs;
    elseif l_a==2 % experience asset and one other asset
        Policy_aprime(:,:,jj,1)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1);
        Policy_aprime(:,:,jj,2)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(aprimeIndexes-1+1);
        PolicyProbs(:,:,jj,1)=aprimeProbs;
        PolicyProbs(:,:,jj,2)=1-aprimeProbs;
    elseif l_a==3 % experience asset and two other assets
        Policy_aprime(:,:,jj,1)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1);
        Policy_aprime(:,:,jj,2)=shiftdim(Policy(l_d+1,:,:,jj),1)+n_a(1)*(shiftdim(Policy(l_d+2,:,:,jj),1)-1)+prod(n_a(1:2))*(aprimeIndexes-1+1);
        PolicyProbs(:,:,jj,1)=aprimeProbs;
        PolicyProbs(:,:,jj,2)=1-aprimeProbs;       
    else
        error('Not yet implemented experience asset with length(n_a)>3')
    end
end

% % Only d variables we need are the ones for the semi-exogenous asset
% Policy_dsemiexo=shiftdim(PolicyKron(l_d,:,:,:); % The last d variable is the relevant one for the semi-exogenous asset. 
% Rather than actually create Policy_dsemiexo we just pass this as the input to the simulation/iteration commands

if simoptions.iterate==0
    if simoptions.parallel>=3
        % Sparse matrix is not relevant for the simulation methods, only for iteration method
        simoptions.parallel=2; % will simulate on parallel cpu, then transfer solution to gpu
    end
    StationaryDistKron=StationaryDist_FHorz_Case1_Simulation_SemiExo_TwoProbs_raw(gather(jequaloneDistKron),AgeWeightParamNames,gather(reshape(Policy(l_d,:,:,:),[N_a,N_z,N_semiz,N_j])),gather(reshape(Policy_aprime,[N_a,N_z,N_semiz,N_j,2])),gather(reshape(PolicyProbs,[N_a,N_z,N_semiz,N_j,2])),N_a,N_z,N_semiz,N_j,pi_z,pi_semiz_J, Parameters, simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_SemiExo_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,shiftdim(Policy(l_d,:,:,:),1),Policy_aprime,PolicyProbs,N_a,N_z,N_semiz,N_j,pi_z,pi_semiz_J,Parameters,simoptions); % zero is n_d, because we already converted Policy to only contain aprime
end

if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDistKron,[n_a,[n_z,simoptions.n_semiz],N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDistKron,[N_a,N_bothz,N_j]);
end

end
