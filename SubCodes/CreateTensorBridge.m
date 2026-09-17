function [harnessFn] = CreateTensorBridge(userReturnFn)
funcInfo = functions(userReturnFn);

if strcmp(funcInfo.type, 'anonymous')
    tokens = regexp(funcInfo.function, '\)\s*([a-zA-Z0-9_]+)\s*\(', 'tokens');
    if ~isempty(tokens)
        baseName = tokens{1}{1};
    else
        error('TensorBridge:ParseError', 'Could not extract base function name from: %s', funcInfo.function);
    end
else
    baseName = funcInfo.function;
end

num_base_args = nargin(baseName);
wrapperName = [baseName, '_AutoBridge'];
fileName = [wrapperName, '.m'];

fid = fopen(fileName, 'w');
if fid == -1
    error('TensorBridge:FileError', 'Could not create harness file %s', fileName);
end

fprintf(fid, 'function F = %s(varargin)\n', wrapperName);
fprintf(fid, '    %% AUTO-GENERATED TENSOR BRIDGE HARNESS (Variadic)\n');
fprintf(fid, '    num_expected = %d;\n', num_base_args);
fprintf(fid, '    num_provided = length(varargin);\n\n');

fprintf(fid, '    if num_provided > num_expected\n');
fprintf(fid, '        args_to_pass = varargin(1:num_expected);\n');
fprintf(fid, '    elseif num_provided < num_expected\n');
fprintf(fid, '        warning(''TensorBridge:MissingArgs'', ''%%s expected %%d arguments but received %%d. Padding with zeros. Check parameter introspection!'', ''%s'', num_expected, num_provided);\n', baseName);
fprintf(fid, '        padding = num2cell(zeros(1, num_expected - num_provided));\n');
fprintf(fid, '        args_to_pass = [varargin, padding];\n');
fprintf(fid, '    else\n');
fprintf(fid, '        args_to_pass = varargin;\n');
fprintf(fid, '    end\n\n');

fprintf(fid, '    F = arrayfun(@%s, args_to_pass{:});\n', baseName);
fprintf(fid, 'end\n');
fclose(fid);

rehash;
harnessFn = str2func(wrapperName);


end