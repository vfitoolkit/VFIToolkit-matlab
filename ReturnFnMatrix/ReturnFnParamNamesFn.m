function ReturnFnParamNames = ReturnFnParamNamesFn(ReturnFn, n_d, n_a, n_z, N_j, vfoptions, Parameters)
% RETURNFNPARAMNAMESFN (V-World Upgraded)
% Extracts parameter names dynamically using vectorized struct field lookups,
% while providing user-friendly error handling for misspelled parameters.

% 1. Get all input names from the anonymous function
input_names = getAnonymousFnInputNames(ReturnFn);

% 2. Vectorized Dictionary Lookup
param_mask = isfield(Parameters, input_names);

% 3. V-World Heuristic Validation: States must precede Parameters!
first_param_idx = find(param_mask, 1);

if ~isempty(first_param_idx)
    % Check if any inputs AFTER the first parameter are missing from the struct
    invalid_mask = ~param_mask(first_param_idx:end);

    if any(invalid_mask)
        % Find the exact name of the misspelled parameter
        tail_names = input_names(first_param_idx:end);
        bad_names = tail_names(invalid_mask);

        error_msg = sprintf('\nError: Cannot find the parameter ''%s'' in the Parameters structure.\n', bad_names{1});
        error_msg = [error_msg, sprintf('It appears after valid parameters (like ''%s'') in your ReturnFn signature, so it is assumed to be a parameter. Did you misspell it, or forget to add it to Params?', input_names{first_param_idx})];
        error(error_msg);
    end
end

% 4. Apply the logical mask to extract only the parameters
ReturnFnParamNames = input_names(param_mask);

% Fallback safety
if isempty(ReturnFnParamNames)
    warning('No parameters found in ReturnFn matching the Parameters struct.');
end


end