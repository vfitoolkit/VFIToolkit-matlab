function GeneralEqmConditionsValue=GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters)
% Input should be just one General Eqm Condition, and must be in the cell form (not the structure form)
% This is a CPU based version of _v3g. Identical except this does not use GPU.

GeneralEqmEqnParamsVec=gather(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames));
GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
for jj=1:length(GeneralEqmEqnParamsVec)
    GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
end

GeneralEqmConditionsValue=GeneralEqmEqnsCell(GeneralEqmEqnParamsCell{:});

end
