function VectorOfParamValues=CreateVectorFromParams(ParamNames, Parameters)

% CreateVectorFromParams looks in structure called 'Parameters' and 
% then creates a vector containing the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames)

nCalibParams=length(ParamNames);
ParamNames=fieldnames(Parameters);
nFields=length(ParamNames);

VectorOfParamValues=nan(nCalibParams,1);
for iCalibParam = 1:nCalibParams
    for iField=1:nFields
        if strcmp(ParamNames{iCalibParam},ParamNames{iField})
            VectorOfParamValues(iCalibParam)=Parameters.(ParamNames{iField});
        end
    end
end

end