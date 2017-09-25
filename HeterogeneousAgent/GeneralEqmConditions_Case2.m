function  [GeneralEqmConditionsVec]=GeneralEqmConditions_Case2(SSvalues_AggVars,p, GeneralEqmEqns, Parameters, GeneralEqmEqnParamNames)
% This code is actually identical to GeneralEqmConditions_Case1 anyway

GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns),'gpuArray')*Inf;
for i=1:length(GeneralEqmEqns)
    if isempty(GeneralEqmEqnParamNames(i).Names)  % check for 'GeneralEqmEqnParamNames(i).Names={}'
        GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(SSvalues_AggVars, p);
    else
        GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames(i).Names));
        GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
        for jj=1:length(GeneralEqmEqnParamsVec)
            GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
        end

        GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(SSvalues_AggVars, p, GeneralEqmEqnParamsCell{:});
    end
    
end

end
