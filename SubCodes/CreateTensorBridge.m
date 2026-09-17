function [harnessFn] = CreateTensorBridge(userReturnFn)
% Introspect the original user function
funcInfo = functions(userReturnFn);
baseName = funcInfo.function;

% Extract the parameter names using VFI Toolkit's existing introspection tools
% (Assuming ReturnFnParamNamesFn or similar is accessible here)
paramNames = ReturnFnParamNamesFn(userReturnFn);
paramsList = strjoin(paramNames, ', ');

% Define the auto-generated wrapper name
wrapperName = [baseName, '_AutoBridge'];
fileName = [wrapperName, '.m'];

% Generate the tensor bridge harness file
fid = fopen(fileName, 'w');
if fid == -1
    error('TensorBridge:FileError', 'Could not create harness file %s', fileName);
end

fprintf(fid, 'function F = %s(%s)\n', wrapperName, paramsList);
fprintf(fid, '    %% AUTO-GENERATED TENSOR BRIDGE HARNESS\n');
fprintf(fid, '    %% Dynamically compiles %s into a CUDA kernel\n\n', baseName);

fprintf(fid, '    F = arrayfun(@%s, %s);\n', baseName, paramsList);

fprintf(fid, 'end\n');
fclose(fid);

% Ensure MATLAB recognizes the newly written file immediately
rehash;

% Return the wrapped function handle to be injected into vfoptions
harnessFn = str2func(wrapperName);


end