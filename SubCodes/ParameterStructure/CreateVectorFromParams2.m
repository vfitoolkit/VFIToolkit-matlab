function VectorOfParamValues=CreateVectorFromParams2(Parameters,ParamNames)
% Does exactly the same as CreateVectorFromParams except that 'Parameters'
% are themselves allowed to be vector valued. It is however slower than
% CreateVectorFromParams. (Does not allow for age or type dependent
% parameters). Currently the use of this is actually for
% 'EstimationTargets', not 'Parameters'.
%
% VectorOfParamValues=CreateVectorFromParams(Parameters,ParamNames)

nCalibParams=length(ParamNames);
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);

VectorOfParamValues=nan; % Will be a row vector
for iCalibParam = 1:nCalibParams
    found=0;
    for iField=1:nFields
        if strcmp(ParamNames{iCalibParam},FullParamNames{iField})
            if iCalibParam==1
                VectorOfParamValues(iCalibParam)=Parameters.(FullParamNames{iField});
            else
                temp=Parameters.(FullParamNames{iField});
                if isrow(temp)
                    VectorOfParamValues=[VectorOfParamValues,temp];
                else
                    VectorOfParamValues=[VectorOfParamValues,temp'];
                end
            end
            found=1;
        end
    end
    if found==0 % Have added this check so that user can see if they are missing a parameter
        fprintf(['FAILED TO FIND PARAMETER ',ParamNames{iCalibParam}])
    end
end


end