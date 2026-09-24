function StationaryDist_new = StationaryDist_VFHorz_Case1(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_z, N_j, pi_z_J, Parameters, simoptions)

% STATIONARYDIST_VFHORZ_CASE1
% Master Orchestrator for V-World Forward Simulation.
% Inspects the model structure and dispatches to the correct vectorized tensor engine.

% --- 1. Identify Model Features from simoptions ---

if exist('simoptions','var')==0
    simoptions.gridinterplayer=0; % =1 Policy interpolates between grid points (must match vfoptions.interpgridlayer)
    % Alternative endo states
    simoptions.experienceasset=0;
    simoptions.experienceassetu=0;
    simoptions.experienceassete=0;
    simoptions.experienceassetz=0;
    simoptions.experienceassetze=0;
    simoptions.experienceassetsemiz=0;
    simoptions.riskyasset=0;
    simoptions.residualasset=0;
    % Exogenous shocks
    simoptions.n_e=0;
    simoptions.n_semiz=0;
    % Things that are really just for internal usage
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.outputkron=0; % If 1 then leave output in Kron form
    simoptions.alreadygridvals=0; % =1 when calling as a subcommand
    simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    simoptions.jequaloneDistAge=1; % jequaloneDist is the distribution at this age (=1 is the standard first period)
    simoptions.optimize_nProbs = 0; % Default to off
    simoptions.precision=underlyingType(jequaloneDist);
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0; % =1 Policy interpolates between grid points (must match vfoptions.interpgridlayer)
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('When using simoptions.gridinterplayer=1 you must set simoptions.ngridinterp (number of points to interpolate for aprime between each consecutive pair of points in a_grid)')
        end
    end
    % Alternative endo states
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetu')
        simoptions.experienceassetu=0;
    end
    if ~isfield(simoptions,'experienceassete')
        simoptions.experienceassete=0;
    end
    if ~isfield(simoptions,'experienceassetz')
        simoptions.experienceassetz=0;
    end
    if ~isfield(simoptions,'experienceassetze')
        simoptions.experienceassetze=0;
    end
    if ~isfield(simoptions,'experienceassetsemiz')
        simoptions.experienceassetsemiz=0;
    end
    if ~isfield(simoptions,'riskyasset')
        simoptions.riskyasset=0;
    end
    if ~isfield(simoptions,'residualasset')
        simoptions.residualasset=0;
    end
    % Exogenous shocks
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    % Things that are really just for internal usage
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'outputkron')
        simoptions.outputkron=0; % If 1 then leave output in Kron form
    end
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0; % =1 when calling as a subcommand
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
    if ~isfield(simoptions,'jequaloneDistAge')
        simoptions.jequaloneDistAge=1; % jequaloneDist is the distribution at this age (=1 is the standard first period)
    end
    % Some options require certain other inputs, and these have to be on the GPU
    if isfield(simoptions,'d_grid')
        simoptions.d_grid=gpuArray(simoptions.d_grid);
    elseif simoptions.experienceasset>=1 || simoptions.experienceassetz>=1 || simoptions.experienceassete>=1 || simoptions.experienceassetze>=1 || simoptions.experienceassetu>=1 || simoptions.experienceassetsemiz>=1
        error('When using any kind of experience asset you must set simoptions.d_grid')
    elseif simoptions.riskyasset==1
        error('When using a risky asset you must set simoptions.d_grid')
    elseif simoptions.residualasset>=1 && n_d(1)>0
        error('When using a residual asset you must set simoptions.d_grid')
    end
    if isfield(simoptions,'a_grid')
        simoptions.a_grid=gpuArray(simoptions.a_grid);
    elseif simoptions.experienceasset>=1 || simoptions.experienceassetz>=1 || simoptions.experienceassete>=1 || simoptions.experienceassetze>=1 || simoptions.experienceassetu>=1 || simoptions.experienceassetsemiz>=1
        error('When using any kind of experience asset you must set simoptions.a_grid')
    end
    if isfield(simoptions,'z_grid')
        simoptions.z_grid=gpuArray(simoptions.z_grid);
    elseif simoptions.experienceassetz>=1
        error('When using experienceassetz you must set simoptions.z_grid')
    elseif simoptions.experienceassetze>=1
        error('When using experienceassetze you must set simoptions.z_grid')
    elseif simoptions.residualasset>=1
        if n_z(1)>0
            error('When using a residual asset you must set simoptions.z_grid')
        else
            z_gridvals_J=[];
        end
    end
    if ~isfield(simoptions, 'optimize_nProbs')
        simoptions.optimize_nProbs = 0;
    end
    if ~isfield(simoptions, 'precision')
        simoptions.precision=underlyingType(jequaloneDist);
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

%%
if simoptions.parallel<2
   % CPU can be used, but only for the basics. Is kept separate here so that the rest of the codes can just assume you have GPU and work with it.
   StationaryDist=StationaryDist_FHorz_CPU(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
   return
end

has_expasset  = simoptions.experienceasset == 1;
has_expassetz = simoptions.experienceassetz == 1;
has_semiz     = prod(simoptions.n_semiz)>0;

if has_semiz
    % Unpack Semi-Exo requirements
    n_semiz = simoptions.n_semiz;

    % Fetch the pre-calculated transition tensor
    if isfield(simoptions, 'pi_semiz_J')
        pi_semiz_J = simoptions.pi_semiz_J;
    else
        %% Semi-exogenous shock gridvals and pi
        % Internally, only ever use age-dependent joint-grids
        % Would be great to get these from vfoptions already calculated...
        simoptions = SemiExogShockSetup_FHorz(n_d, N_j, simoptions.d_grid, Parameters, simoptions, 3);
        pi_semiz_J = simoptions.pi_semiz_J;
    end
else
    n_semiz = 0;
    pi_semiz_J = [];
end

% --- 1.5. Legacy Compatibility: Inflate pi_z_J to 3D ---
% Legacy iteration functions (e.g., StationaryDist_FHorz_SemiExo) 
% strictly index pi_z_J(:,:,jj) without checking dimensions.
if size(pi_z_J, 3) == 1 && N_j > 1
    pi_z_J = repmat(pi_z_J, [1, 1, N_j]);
end

% --- 2. Dispatch to Specific Orchestrators ---
if (has_expasset || has_expassetz)
    if has_semiz
        % Route to the Universal ExpAssetSemiZ Simulator
        tic;
        StationaryDist_new = StationaryDist_VFHorz_ExpAssetsemiz(...
            jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, ...
            N_j, pi_semiz_J, pi_z_J, Parameters, simoptions);
        time_new=toc;
        tic;
        StationaryDist_ref = StationaryDist_FHorz_ExpAssetsemiz(...
            jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, ...
            N_j, pi_semiz_J, pi_z_J, Parameters, simoptions);
        time_ref=toc;
        if any(StationaryDist_new ~= StationaryDist_ref)
            error("StationaryDist_new ~= StationaryDist_ref")
        end
    else
        % Route to StationaryDist_VFHorz_ExpAsset
        tic;
        StationaryDist_new = StationaryDist_VFHorz_ExpAsset(...
            jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_z, ...
            N_j, simoptions.z_gridvals_J, pi_z_J, Parameters, simoptions);
        time_new=toc;
        StationaryDist_ref = StationaryDist_VFHorz_ExpAsset(...
            jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_z, ...
            N_j, simoptions.z_gridvals_J, pi_z_J, Parameters, simoptions);
        time_ref=toc;
        if any(StationaryDist_new ~= StationaryDist_ref)
            error("StationaryDist_new ~= StationaryDist_ref")
        end
    end

    fprintf('time reference: %.2f seconds; time difference: %.2f seconds; time ratio to ref: %.0f%%\n', time_ref, time_new-time_ref, 100*time_new/time_ref);

elseif has_semiz
    % No aprimeFn tensor, so use standard dispatcher
    fprintf("StationaryDist_VFHorz_Case1 deferring to reference StationaryDist_FHorz_SemiExo\n");
    StationaryDist_new=StationaryDist_FHorz_SemiExo( ...
        jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_semiz,n_z, ...
        N_j,pi_semiz_J,pi_z_J,Parameters,simoptions);

else
    % Future: Route to standard StationaryDist_VFHorz (Base Model)
    persistent original_func

    if isempty(original_func)
        % Get the name of this file (e.g., 'my_shadowing_function.m')
        this_file = 'StationaryDist_FHorz_Case1.m'; 

        % Find all instances on the MATLAB path
        all_paths = which(this_file, '-all'); 

        % If the shadowing file is in IntroToLifeCycleModels, skip past that
        shadowed_file_path = all_paths{1 + logical(strfind(all_paths{1},'IntroToLifeCycleModels'))};

        % Extract the directory containing the shadowed function
        shadowed_dir = fileparts(shadowed_file_path);

        % Temporarily change directories to get a clean handle to it
        current_dir = cd(shadowed_dir);
        original_func = str2func(this_file(1:end-2)); % chop off the .m at the end 
        cd(current_dir); % Go back safely
    end

    % Your custom wrapper code goes here...
    fprintf("StationaryDist_VFHorz_Case1 deferring to reference StationaryDist_FHorz_Case1\n");
    % Call the shadowed function using the saved handle
    StationaryDist_new = original_func( ...
        jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z, ...
        N_j,pi_z_J,Parameters,simoptions);

    return

    % When no longer testing double-barrel style, just call the function directly
    StationaryDist_new=StationaryDist_FHorz_Case1( ...
        jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z, ...
        N_j,pi_z_J,Parameters,simoptions);
end


end