function  [GeneralEqmConditionsVec]=GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters, GeneralEqmEqnParamNames)

GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns))*Inf;
for i=1:length(GeneralEqmEqns)
    if isempty(GeneralEqmEqnParamNames(i).Names)  % check for 'GeneralEqmEqnParamNames(i).Names={}'
        GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(AggVars, p);
    else
        GeneralEqmEqnParamsVec=CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames(i).Names);
        GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
        for jj=1:length(GeneralEqmEqnParamsVec)
            GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
        end

        GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(AggVars, p, GeneralEqmEqnParamsCell{:});
    end
end


end
