function CreateCleanedReturnFn(NameOfReturnFnFile, ReturnFnParamNames, ReturnFnParams)

FID1=fopen([NameOfReturnFnFile,'.m']);
FID2=fopen('TempReturnFn.m','w');

tline = fgetl(FID1);
% The first line will be 'function etc', so clean this up now
tline = strrep(tline, NameOfReturnFnFile, 'TempReturnFn');

while ischar(tline)
    if strcmp('%PARAMETERVALUESHERE%',tline)
        for ii=1:length(ReturnFnParams)
            fprintf(FID2, [ReturnFnParamNames{ii}, ' = ', num2str(ReturnFnParams(ii)),'; \n']);
        end
    else
        fprintf(FID2,[tline,' \n']);
    end
    tline = fgetl(FID1);
end

fclose(FID1);
fclose(FID2);

end