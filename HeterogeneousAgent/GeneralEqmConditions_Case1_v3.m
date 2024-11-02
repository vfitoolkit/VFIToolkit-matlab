function GeneralEqmConditionsValue=GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters)
% Stripped back version based on cell [v2 is based on structure, this one is faster as the names of inputs have been precomputed]
GeneralEqmConditionsValue=zeros(1,length(GeneralEqmEqnsCell));
for gg=1:length(GeneralEqmEqnsCell)
    GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames(gg).Names));
    GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
    for jj=1:length(GeneralEqmEqnParamsVec)
        GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
    end
    GeneralEqmConditionsValue(gg)=GeneralEqmEqnsCell{gg}(GeneralEqmEqnParamsCell{:});
end

end
