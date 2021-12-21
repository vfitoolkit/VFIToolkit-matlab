function  [GeneralEqmConditionsVec]=GeneralEqmConditions_Case1_new(GeneralEqmEqns, GeneralEqmEqnInputNames,Parameters, Parallel)
% Parallel is an optional input

if exist('Parallel','var')==0 || isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

if Parallel==2 || Parallel==4
    GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns),'gpuArray')*Inf;
    for ii=1:length(GeneralEqmEqns)
        GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnInputNames(ii).Names));
        GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
        for jj=1:length(GeneralEqmEqnParamsVec)
            GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
        end
        
        GeneralEqmConditionsVec(ii)=GeneralEqmEqns{ii}(GeneralEqmEqnParamsCell{:});
    end
else
    GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns))*Inf;
    for ii=1:length(GeneralEqmEqns)
        GeneralEqmEqnParamsVec=CreateVectorFromParams(Parameters,GeneralEqmEqnInputNames(ii).Names);
        GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
        for jj=1:length(GeneralEqmEqnParamsVec)
            GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
        end
        
        GeneralEqmConditionsVec(ii)=GeneralEqmEqns{ii}(GeneralEqmEqnParamsCell{:});
    end
end

end
