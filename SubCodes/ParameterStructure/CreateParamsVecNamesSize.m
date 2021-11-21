function [ParamsVec,ParamsNames,ParamsSize]=CreateParamsVecNamesSize(Parameters)
% Creates ParamsVec,ParamsNames, & ParamsSize from Parameters
% You can do the reverse using CreateVecNamesSizeParams().
%
% ParamsVec is a column vector containing all the parameters in Params (if a parameter is a matrix, it gets reshaped)
% ParamsNames is a cell containing strings for the name of each parameter
% ParamsSize is a matrix, each row of which contains the size of the parameters (is only really needed so it is possible to reconstruct Parameters)

ParamsNames=fieldnames(Parameters);
nFields=length(ParamsNames);

ParamsSize=zeros(nFields,2); % All parameters are assumed to be no more than 2-dimensional

ParamsVec=[];
for iField = 1:nFields
    temp=Parameters.(ParamsNames{iField});
    ParamsSize(iField,:)=size(temp);
    ParamsVec=[ParamsVec; reshape(temp,[numel(temp),1])];
end

end

