function GeneralEqmConditionsVec=GeneralEqmConditions_Case1_v2(GeneralEqmEqns, Parameters, Parallel)
% Parallel is an optional input

if exist('Parallel','var')==0 || isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

%% Implement handling of GeneralEqmEqns
%if isstruct(GeneralEqmEqns)
GEInputParamNames=fieldnames(GeneralEqmEqns);
for ff=1:length(GEInputParamNames)
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEInputParamNames{ff}));
    GeneralEqmEqnParamNames(ff).Names=temp;
    GeneralEqmEqns2{ff}=GeneralEqmEqns.(GEInputParamNames{ff});
end
GeneralEqmEqns=GeneralEqmEqns2;

%%
if Parallel==2 || Parallel==4
    GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns),'gpuArray')*Inf;
    for i=1:length(GeneralEqmEqns)
        GeneralEqmEqnParamsVec=gpuArray(CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames(i).Names));
        GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
        for jj=1:length(GeneralEqmEqnParamsVec)
            GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
        end
        
        GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(GeneralEqmEqnParamsCell{:});
    end
else
    GeneralEqmConditionsVec=ones(1,length(GeneralEqmEqns))*Inf;
    for i=1:length(GeneralEqmEqns)
        GeneralEqmEqnParamsVec=CreateVectorFromParams(Parameters,GeneralEqmEqnParamNames(i).Names);
        GeneralEqmEqnParamsCell=cell(length(GeneralEqmEqnParamsVec),1);
        for jj=1:length(GeneralEqmEqnParamsVec)
            GeneralEqmEqnParamsCell(jj,1)={GeneralEqmEqnParamsVec(jj)};
        end
        
        GeneralEqmConditionsVec(i)=GeneralEqmEqns{i}(GeneralEqmEqnParamsCell{:});
    end
end

end
