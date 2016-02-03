function VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames)

% CreateVectorFromParams looks in structure called 'Parameters' and 
% then creates a vector containing the values of it's fields that
% correspond to those field names in ParamNames (and in the order
% given by CalibParamNames)

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

VectorOfParamValues=nan(nCalibParams,1);
for iCalibParam = 1:nCalibParams
    found=0;
    for iField=1:nFields
        if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
            VectorOfParamValues(iCalibParam)=Parameters.(FullParamNames{iField});
            found=1;
        end
    end
    if found==0 % Have added this check so that user can see if they are missing a parameter
        fprintf(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
    end
end

end