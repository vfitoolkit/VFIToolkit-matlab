function StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)
%%
if isempty(n_d)
    n_d=0;
end
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.iterate=1;
    if simoptions.iterate==1
        simoptions.parallel=3-(gpuDeviceCount>0); % 3 (sparse) if cpu, 2 if gpu
    elseif simoptions.iterate==0
        simoptions.parallel=1+(gpuDeviceCount>0); % 1 if cpu, 2 if gpu
    end
    simoptions.tanimprovement=1;  % Mostly hardcoded, but in simplest case of (a,z) you can try out the alternatives
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0;
    simoptions.riskyasset=0;
    simoptions.residualasset=0;
    simoptions.tolerance=10^(-9);
    simoptions.outputkron=0; % If 1 then leave output in Kron form
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=10^4;
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions,'parallel')
        if simoptions.iterate==1
            simoptions.parallel=3-(gpuDeviceCount>0); % 3 (sparse) if cpu, 2 if gpu
        elseif simoptions.iterate==0
            simoptions.parallel=1+(gpuDeviceCount>0); % 1 if cpu, 2 if gpu
        end
    end
    if ~isfield(simoptions,'tanimprovement')
        simoptions.tanimprovement=1; % Mostly hardcoded, but in simplest case of (a,z) you can try out the alternatives
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'ncores')
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetu')
        simoptions.experienceassetu=0;
    end
    if ~isfield(simoptions,'riskyasset')
        simoptions.riskyasset=0;
    end
    if ~isfield(simoptions,'residualasset')
        simoptions.residualasset=0;
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'outputkron')
        simoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(simoptions,'loopovere')
        simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
    end
end


%% Check for the age weights parameter, and make sure it is a row vector
if size(Parameters.(AgeWeightParamNames{1}),2)==1 % Seems like column vector
    Parameters.(AgeWeightParamNames{1})=Parameters.(AgeWeightParamNames{1})'; 
    % Note: assumed there is only one AgeWeightParamNames
end

%% Check that the age one distribution is of mass one
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
end

%% Set up pi_z_J (transition matrix for markov exogenous state z, depending on age)
if ismatrix(pi_z)
    if simoptions.parallel==2
        pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
    else
        pi_z_J=pi_z.*ones(1,1,N_j);
    end
elseif ndims(pi_z)==3
    pi_z_J=pi_z;
end
if isfield(simoptions,'pi_z_J')
    pi_z_J=simoptions.pi_z_J;
elseif isfield(simoptions,'ExogShockFn')
    N_z=prod(n_z);
    pi_z_J=zeros(N_z,N_z,N_j);
    for jj=1:N_j
        ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, ExogShockFnParamNames,jj);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        pi_z_J(:,:,jj)=pi_z;
    end
end

%% Set up pi_e_J (if relevant)
if isfield(simoptions,'n_e')
    if isfield(simoptions,'pi_e')
        simoptions.pi_e_J=simoptions.pi_e.*ones(1,N_j,'gpuArray');
    else
        % simoptions.pi_e_J=simoptions.pi_e_J;
    end
end

%%
if isfield(simoptions,'SemiExoStateFn')
    if simoptions.experienceasset==1
        StationaryDist=StationaryDist_FHorz_Case1_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        return
    end
    if N_z==0
        StationaryDist=StationaryDist_FHorz_Case1_SemiExo_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
    elseif isfield(simoptions,'n_e')
        StationaryDist=StationaryDist_FHorz_Case1_SemiExo_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
    else
        StationaryDist=StationaryDist_FHorz_Case1_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    end
    return
end
if simoptions.experienceasset==1
    StationaryDist=StationaryDist_FHorz_Case1_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.experienceassetu==1
    StationaryDist=StationaryDist_FHorz_Case1_ExpAssetu(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.riskyasset==1
    StationaryDist=StationaryDist_FHorz_Case1_RiskyAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.residualasset==1
    StationaryDist=StationaryDist_FHorz_Case1_ResidAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end

if isfield(simoptions,'n_e')
    if n_z(1)==0
        StationaryDist=StationaryDist_FHorz_Case1_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
    else
        StationaryDist=StationaryDist_FHorz_Case1_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
    end
    return
end

if n_z(1)==0
    StationaryDist=StationaryDist_FHorz_Case1_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
    return
end

%% Solve the baseline case
jequaloneDist=reshape(jequaloneDist,[N_a*N_z,1]);
Policy=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);
if simoptions.iterate==0
    Policy=gather(Policy);
    jequaloneDist=gather(jequaloneDist);    
end
pi_z_J=gather(pi_z_J);


if simoptions.iterate==0
    StationaryDistKron=StationaryDist_FHorz_Case1_Simulation_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z_J,Parameters,simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z_J,Parameters,simoptions);
end

if simoptions.outputkron==0
    StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);
else
    % If 1 then leave output in Kron form
    StationaryDist=reshape(StationaryDistKron,[N_a,N_z,N_j]);
end

end
