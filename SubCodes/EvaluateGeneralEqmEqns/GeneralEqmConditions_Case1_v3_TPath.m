function GeneralEqmConditionsValue=GeneralEqmConditions_Case1_v3_TPath(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters,T)
% Input should be just one General Eqm Condition, and must be in the cell form (not the structure form)
% This is a CPU based version of _v3g. Identical except this does not use GPU.

% _TPath is that it does all the periods t=1:1:T at once
GeneralEqmConditionsValue=zeros(1,T);
for tt=1:T
    GeneralEqmEqnParamsVec=gather(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames,tt));
    GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
    for pp=1:length(GeneralEqmEqnParamsVec)
        GeneralEqmEqnParamsCell(pp,1)={GeneralEqmEqnParamsVec(pp)};
    end

    GeneralEqmConditionsValue(tt)=GeneralEqmEqnsCell(GeneralEqmEqnParamsCell{:});
end

end
