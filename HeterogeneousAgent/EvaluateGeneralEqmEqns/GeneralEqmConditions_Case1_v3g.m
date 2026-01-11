function GeneralEqmConditionsValue=GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters)
% Input should be just one General Eqm Condition, and must be in the cell form (not the structure form)
% v3g: the g indicates just one general eqm condtiion [v3 includes loop over g]

GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames));
GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
for pp=1:length(GeneralEqmEqnParamsVec)
    GeneralEqmEqnParamsCell(pp,1)={GeneralEqmEqnParamsVec(pp)};
end

GeneralEqmConditionsValue=GeneralEqmEqnsCell(GeneralEqmEqnParamsCell{:});

end
