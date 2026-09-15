function AgeConditionalStats = LifeCycleProfiles_VFHorz_Case1(StationaryDist, Policy, FnsToEvaluate, Parameters, DiscountFactorParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions)
% LIFECYCLEPROFILES_VFHORZ_CASE1
% A clean routing shim that shields modelers from internal toolkit workarounds.
% It intercepts the standard LifeCycleProfiles call, checks if the model 
% uses Experience Assets (which output fewer policy layers), and pads the 
% Policy tensor to prevent standard toolkit profilers from crashing.

% 1. Determine expected layers vs actual layers
% Standard models expect a policy layer for every decision and every asset.
expected_layers = length(n_d) + length(n_a);
actual_layers = size(Policy, 1);

% 2. Intercept and Pad if it's an Experience Asset policy
if actual_layers < expected_layers
    if isfield(simoptions, 'verbose') && simoptions.verbose == 1
        fprintf('LifeCycleProfiles_VFHorz_Case1: Padding %d-layer Experience Asset Policy to %d layers for profiler compatibility.\n', actual_layers, expected_layers);
    end

    % Pre-allocate the padded array
    padded_size = size(Policy);
    padded_size(1) = expected_layers;
    Policy_padded = ones(padded_size, 'like', Policy);

    % Dynamically copy the actual layers across all trailing N-dimensions
    colons = repmat({':'}, 1, ndims(Policy) - 1);
    Policy_padded(1:actual_layers, colons{:}) = Policy;
else
    % Standard model, no padding required
    Policy_padded = Policy;
end

% 3. Fall through to the standard toolkit graphing utility
AgeConditionalStats = LifeCycleProfiles_FHorz_Case1(StationaryDist, Policy_padded, ...
    FnsToEvaluate, Parameters, DiscountFactorParamNames, n_d, n_a, n_z, N_j, ...
    d_grid, a_grid, z_grid, simoptions);
end