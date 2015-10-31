function Parameters=CreateParamsStrucFromParamsVec(ParamNames, ParametersVec)
%%
% Assumes that the order of the variables in ParametersVec is the same as
% in ParamNames. This will always be true if ParametersVec was created
% using CreateVectorFromParams (I SHOULD MAKE IT SO THAT THIS IS NOT
% NECESSARY)
%%

nCalibParams=length(ParamNames);

for iCalibParam = 1:nCalibParams
    Parameters.(ParamNames{iCalibParam})=ParametersVec(iCalibParam);
end

end