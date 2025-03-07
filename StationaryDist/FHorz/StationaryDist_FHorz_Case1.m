function StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)
%%
if isempty(n_d)
    n_d=0;
end
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.parallel=1+(gpuDeviceCount>0); % 1 if cpu, 2 if gpu
    simoptions.tanimprovement=1;  % Mostly hardcoded, but in simplest case of (a,z) you can try out the alternatives
    simoptions.verbose=0;
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0;
    simoptions.riskyasset=0;
    simoptions.residualasset=0;
    simoptions.tolerance=10^(-9);
    simoptions.outputkron=0; % If 1 then leave output in Kron form
    simoptions.loopovere=0; % default is parallel over e, 1 will loop over e, 2 will parfor loop over e
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
            simoptions.parallel=1+(gpuDeviceCount>0); % 1 if cpu, 2 if gpu
    end
    if ~isfield(simoptions,'tanimprovement')
        simoptions.tanimprovement=1; % Mostly hardcoded, but in simplest case of (a,z) you can try out the alternatives
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
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

% Check for something that used to be an option, but no longer is
if isfield(simoptions,'iterate')
    if simoptions.iterate==0
        error('simoptions.iterate=0 is no longer supported (has been eliminated from VFI Toolkit)')
    end
end

%% Check for the age weights parameter, and make sure it is a row vector
if size(Parameters.(AgeWeightParamNames{1}),2)==1 % Seems like column vector
    Parameters.(AgeWeightParamNames{1})=Parameters.(AgeWeightParamNames{1})'; 
    % Note: assumed there is only one AgeWeightParamNames
end
% And check that the age weights sum to one
if abs((sum(Parameters.(AgeWeightParamNames{1}))-1))>10^(-15)
    warning('StationaryDist: The age-weights do not sum to one')
end

%% Set up pi_z_J (transition matrix for markov exogenous state z, depending on age)
if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
    simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
end

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
elseif isfield(simoptions,'ExogShockFn') && ~isa(jequaloneDist, 'function_handle')
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
elseif isfield(simoptions,'ExogShockFn') && isa(jequaloneDist, 'function_handle')
    % Need to keep z_grid as it will be needed for jequaloneDist
    N_z=prod(n_z);
    pi_z_J=zeros(N_z,N_z,N_j);
    simoptions.z_grid_J=zeros(N_z,length(n_z),N_j);
    for jj=1:N_j
        ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, ExogShockFnParamNames,jj);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
        pi_z_J(:,:,jj)=pi_z;
        simoptions.z_grid_J(:,:,jj)=CreateGridvals(n_z,z_grid,1);
    end
end


%% If age one distribution is input as a function, then evaluate it
% Note: we might need to update z_grid based on ExogShockFn, so this has to come after the pi_z section
if isa(jequaloneDist, 'function_handle')
    jequaloneDistFn=jequaloneDist;
    clear jequaloneDist
    % figure out any parameters
    temp=getAnonymousFnInputNames(jequaloneDistFn);
    if length(temp)>4 % first 4 are a_grid,z_grid,n_a,n_z
        jequaloneDistFnParamNames={temp{5:end}}; % the first inputs will always be (a_grid,z_grid,n_a,n_z,...)
    else
        jequaloneDistFnParamNames={};
    end
    jequaloneParamsCell={};
    for pp=1:length(jequaloneDistFnParamNames)
        jequaloneParamsCell{pp}=Parameters.(jequaloneDistFnParamNames{pp});
    end
    % make sure a_grid and z_grid have been put in simoptions
    if ~isfield(simoptions,'a_grid')
        error('When using jequaloneDist as a function you must put a_grid into simoptions.a_grid')
    elseif ~isfield(simoptions,'z_grid')
        error('When using jequaloneDist as a function you must put z_grid into simoptions.z_grid')
    end

    jequaloneDist=jequaloneDistFn(simoptions.a_grid,simoptions.z_grid,n_a,n_z,jequaloneParamsCell{:});
end

% Check that the age one distribution is of mass one
if abs(sum(jequaloneDist(:))-1)>10^(-9)
    error('The jequaloneDist must be of mass one')
end


%% Set up pi_e_J (if relevant)
if isfield(simoptions,'n_e') % THIS IS SILLY (DON'T THINK IT DOES ANYTHING WRONG, BUT IS NOT UP TO CURRENT TOOLKIT NOTATIONAL STANDARDS)
    if isfield(simoptions,'pi_e')
        simoptions.pi_e_J=simoptions.pi_e.*ones(1,N_j,'gpuArray');
    else
        % simoptions.pi_e_J=simoptions.pi_e_J;
    end
end

%% Non-standard endogenous states
if simoptions.experienceasset==1
    if isfield(simoptions,'SemiExoStateFn')
        StationaryDist=StationaryDist_FHorz_Case1_ExpAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        return
    end
    StationaryDist=StationaryDist_FHorz_Case1_ExpAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.experienceassetu==1
    StationaryDist=StationaryDist_FHorz_Case1_ExpAssetu(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.riskyasset==1
    if isfield(simoptions,'SemiExoStateFn')
    StationaryDist=StationaryDist_FHorz_Case1_RiskyAssetSemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        return
    end
    StationaryDist=StationaryDist_FHorz_Case1_RiskyAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end
if simoptions.residualasset==1
    StationaryDist=StationaryDist_FHorz_Case1_ResidAsset(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
    return
end

%% Standard endogenous states
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
else
    N_e=0;
end

if isfield(simoptions,'SemiExoStateFn')
    if N_e==0
        if N_z==0
            StationaryDist=StationaryDist_FHorz_Case1_SemiExo_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
        else
            StationaryDist=StationaryDist_FHorz_Case1_SemiExo(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        end
    else
        if N_z==0
            error('Not yet implemented N_e=0 N_z>0 with SemiExo, email me and I will do it (or you can just pretend by using n_z=1 and pi_z=1, not using the value of z anywhere)')
        else
            StationaryDist=StationaryDist_FHorz_Case1_SemiExo_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
        end
    end
else
    if N_e==0
        if N_z==0
            StationaryDist=StationaryDist_FHorz_Case1_noz(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
        else
            StationaryDist=StationaryDist_FHorz_Case1_raw(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
        end
    else
        if N_z==0
            StationaryDist=StationaryDist_FHorz_Case1_noz_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,N_j,Parameters,simoptions);
        else
            StationaryDist=StationaryDist_FHorz_Case1_e(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,simoptions.pi_e_J,Parameters,simoptions);
        end
    end
end



end
