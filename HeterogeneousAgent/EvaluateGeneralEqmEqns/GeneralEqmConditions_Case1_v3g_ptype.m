function GeneralEqmConditionsValue=GeneralEqmConditions_Case1_v3g_ptype(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters)
% Input should be just one General Eqm Condition, and must be in the cell form (not the structure form)
% v3g: the g indicates just one general eqm condtiion [v3 includes loop over g]

GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames,1)); % the ,1 is only difference of _ptype
GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
for jj=1:length(GeneralEqmEqnParamsVec)
    GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
end

GeneralEqmConditionsValue=GeneralEqmEqnsCell(GeneralEqmEqnParamsCell{:});

end
