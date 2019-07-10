function  [GeneralEqmConditionsVec]=GeneralEqmConditions_Case2(AggVars,p, GeneralEqmEqns, Parameters, GeneralEqmEqnParamNames, Parallel)
% This code is actually identical to GeneralEqmConditions_Case1 anyway

if nargin<6 || Parallel==2 % nargin<6 makes this the default when Parallel is not inputted
    GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns),'gpuArray')*Inf;
    for i=1:length(GeneralEqmEqns)
        if isempty(GeneralEqmEqnParamNames(i).Names)  % check for 'GeneralEqmEqnParamNames(i).Names={}'
            GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(AggVars, p);
        else
            GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames(i).Names));
            GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
            for jj=1:length(GeneralEqmEqnParamsVec)
                GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
            end
            
            GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(AggVars, p, GeneralEqmEqnParamsCell{:});
        end
    end
else
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

end
