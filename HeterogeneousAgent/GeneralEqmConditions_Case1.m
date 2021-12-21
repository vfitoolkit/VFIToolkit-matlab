function  [GeneralEqmConditionsVec]=GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters, GeneralEqmEqnParamNames, Parallel)
% Parallel is an optional input

if exist('Parallel','var')==0 || isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

%%
if Parallel==2 || Parallel==4
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
