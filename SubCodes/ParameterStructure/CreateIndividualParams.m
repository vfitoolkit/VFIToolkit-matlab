function CreateIndividualParams(Parameters)

% CreateIndividualParams looks for a structure called 'Parameters' and 
% then creates individual objects named after the fields of the structure 
% and containing the values of it's fields

ParamNames=fieldnames(Parameters);
nFields=length(ParamNames);

for iField = 1:nFields
    assignin('caller',ParamNames{iField},Parameters.(ParamNames{iField}));
end

end