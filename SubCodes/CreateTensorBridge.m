function [harnessFn] = CreateTensorBridge(userReturnFn)
% Introspect the original user function
funcInfo = functions(userReturnFn);

if strcmp(funcInfo.type, 'anonymous')
    % Extract the core function name from the anonymous string: @(args)CoreFunction(args)
    tokens = regexp(funcInfo.function, '\)\s*([a-zA-Z0-9_]+)\s*\(', 'tokens');
    if ~isempty(tokens)
        baseName = tokens{1}{1};
    else
        error('TensorBridge:ParseError', 'Could not extract base function name from: %s', funcInfo.function);
    end
else
    % It is already a direct, simple handle
    baseName = funcInfo.function;
end

% 1. How many arguments does the anonymous wrapper/caller provide? (e.g., 6)
num_wrapper_args = nargin(userReturnFn);
if num_wrapper_args < 0
    error('TensorBridge:VararginNotSupported', 'varargin is not supported in the Tensor Bridge.');
end

% 2. How many arguments does the ACTUAL core math file expect? (e.g., 2)
num_base_args = nargin(baseName);

% 3. Generate signatures
% The function definition accepts all arguments passed by the orchestrator
wrapperParams = strjoin(arrayfun(@(x) sprintf('in%d', x), 1:num_wrapper_args, 'UniformOutput', false), ', ');

% But arrayfun ONLY passes the first N arguments that the core math expects
baseParams = strjoin(arrayfun(@(x) sprintf('in%d', x), 1:num_base_args, 'UniformOutput', false), ', ');

% Define the auto-generated wrapper name
wrapperName = [baseName, '_AutoBridge'];
fileName = [wrapperName, '.m'];

% Generate the tensor bridge harness file
fid = fopen(fileName, 'w');
if fid == -1
    error('TensorBridge:FileError', 'Could not create harness file %s', fileName);
end

fprintf(fid, 'function F = %s(%s)\n', wrapperName, wrapperParams);
fprintf(fid, '    %% AUTO-GENERATED TENSOR BRIDGE HARNESS\n');
fprintf(fid, '    %% Truncates %d inputs from caller to the %d inputs expected by the kernel\n\n', num_wrapper_args, num_base_args);

fprintf(fid, '    F = arrayfun(@%s, %s);\n', baseName, baseParams);

fprintf(fid, 'end\n');
fclose(fid);

% Ensure MATLAB recognizes the newly written file immediately
rehash;

% Return the wrapped function handle to be injected into vfoptions
harnessFn = str2func(wrapperName);


end