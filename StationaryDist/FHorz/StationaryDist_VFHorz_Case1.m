function StationaryDist = StationaryDist_VFHorz_Case1(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_z, N_j, pi_z_J, Parameters, simoptions)

% STATIONARYDIST_VFHORZ_CASE1
% Master Orchestrator for V-World Forward Simulation.
% Inspects the model structure and dispatches to the correct vectorized tensor engine.

% --- 1. Identify Model Features from simoptions ---
has_expasset = isfield(simoptions, 'experienceasset') && simoptions.experienceasset == 1;
has_semiz    = isfield(simoptions, 'n_semiz') && ~isempty(simoptions.n_semiz);

% --- 2. Dispatch to Specific Orchestrators ---
if has_expasset && has_semiz
    % Unpack Semi-Exo requirements
    n_semiz = simoptions.n_semiz;

    % Fetch the pre-calculated transition tensor
    if isfield(simoptions, 'pi_semiz_J')
        pi_semiz_J = simoptions.pi_semiz_J;
    else
        error('V-World Error: simoptions.pi_semiz_J is required. Please attach the tensor calculated during VFI to simoptions.');
    end

    % Route to the Universal ExpAssetSemiZ Simulator
    StationaryDist = StationaryDist_VFHorz_ExpAssetsemiz(...
        jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, ...
        N_j, pi_semiz_J, pi_z_J, Parameters, simoptions);

elseif has_expasset
    % Future: Route to StationaryDist_VFHorz_ExpAsset
    error('V-World: ExpAsset (without semiz) dispatcher not yet implemented.');

elseif has_semiz
    % Future: Route to StationaryDist_VFHorz_SemiExo
    error('V-World: SemiExo dispatcher not yet implemented.');

else
    % Future: Route to standard StationaryDist_VFHorz (Base Model)
    error('V-World: Base Model dispatcher not yet implemented.');
end


end