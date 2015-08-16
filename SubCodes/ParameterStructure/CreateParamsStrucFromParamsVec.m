function Parameters=CreateParamsStrucFromParamsVec(ParamNames, ParametersVec)

% Assumes that the order of the variables in ParametersVec is the same as
% in ParamNames. This will always be true if ParametersVec was created
% using CreateVectorFromParams

nCalibParams=length(ParamNames);

for iCalibParam = 1:nCalibParams
    Parameters.(ParamNames{iCalibParam})=ParametersVec(iCalibParam);
end

end