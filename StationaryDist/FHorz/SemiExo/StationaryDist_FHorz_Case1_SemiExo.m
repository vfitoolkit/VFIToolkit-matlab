function StationaryDist=StationaryDist_FHorz_Case1_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)

%% Check for the age weights parameter, and make sure it is a row vector
if size(Parameters.(AgeWeightParamNames{1}),2)==1 % Seems like column vector
    Parameters.(AgeWeightParamNames{1})=Parameters.(AgeWeightParamNames{1})'; 
    % Note: assumed there is only one AgeWeightParamNames
end

%% Check that the age one distribution is of mass one
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
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
n_semiz=simoptions.n_semiz;
N_semiz=prod(n_semiz);
pi_semiz_J=zeros(N_semiz,N_semiz,n_d2,N_j);
for jj=1:N_j
    SemiExoStateFnParamValues=CreateVectorFromParams(Parameters,SemiExoStateFnParamNames,jj);
    pi_semiz_J(:,:,:,jj)=CreatePiSemiZ(n_d2,simoptions.n_semiz,d2_grid,simoptions.semiz_grid,simoptions.SemiExoStateFn,SemiExoStateFnParamValues);
end

%%
if isfield(simoptions,'n_e')
    if n_z(1)==0
        error('Not yet implemented n_z=0 with n_e and SemiExo, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
    else
        StationaryDist=StationaryDist_FHorz_Case1_SemiExo_e(jequaloneDist,AgeWeightParamNames,Policy,n_d1,n_d2,n_a,n_z,n_semiz,N_j,pi_z,pi_semiz_J,Parameters,simoptions);
    end
    return
end

if n_z(1)==0
    error('Not yet implemented n_z=0 with SemiExo, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
end

N_a=prod(n_a);
N_z=prod(n_z);
% N_semiz=prod(n_semiz);

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

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z*N_semiz,1]);
if simoptions.parallel~=2 && simoptions.parallel~=4
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
    pi_z=gather(pi_z);
end

PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, [n_z,simoptions.n_semiz],N_j);

if simoptions.iterate==0
    if simoptions.parallel>=3
        % Sparse matrix is not relevant for the simulation methods, only for iteration method
        simoptions.parallel=2; % will simulate on parallel cpu, then transfer solution to gpu
    end
    StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d1,n_d2,N_a,N_z,N_semiz,N_j,pi_z,pi_semiz_J,Parameters,simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,n_d1,n_d2,N_a,N_z,N_semiz,N_j,pi_z,pi_semiz_J,Parameters,simoptions);
end

if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,simoptions.n_semiz,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDistKron,[N_a,N_z,N_semiz,N_j]);
end

end
