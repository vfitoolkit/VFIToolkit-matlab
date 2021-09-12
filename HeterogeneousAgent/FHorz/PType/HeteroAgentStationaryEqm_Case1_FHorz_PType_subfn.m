function GeneralEqmCondition=HeteroAgentStationaryEqm_Case1_FHorz_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions)

%%
for pp=1:length(GEPriceParamNames) % Not sure this is needed, have it just in case they are used when calling 'GeneralEqmConditionsFn', but I am pretty sure they never would be.
    Parameters.(GEPriceParamNames{pp})=gather(p(pp));
end

if heteroagentoptions.parallel==2
    AggVars=zeros(PTypeStructure.(PTypeStructure.iistr{1}).numFnsToEvaluate,1,'gpuArray'); % numFnsToEvaluate should be independent of the ptype so just take it from the first.
else
    AggVars=zeros(PTypeStructure.(PTypeStructure.iistr{1}).numFnsToEvaluate,1); % numFnsToEvaluate should be independent of the ptype so just take it from the first.    
end

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=p(pp);
    end
    
    [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
    StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
    AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluateFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, [], PTypeStructure.(iistr).simoptions);
    
    for kk=1:PTypeStructure.(iistr).numFnsToEvaluate
        jj=PTypeStructure.(iistr).WhichFnsForCurrentPType(kk);
        if jj>0
            AggVars(kk,:)=AggVars(kk,:)+PTypeStructure.(iistr).PTypeWeight*AggVars_ii(jj,:);
        end
    end

end

% The following line is often a useful double-check if something is going wrong.
%    AggVars

% use of real() is a hack that could disguise errors, but I couldn't
% find why matlab was treating output as complex
GeneralEqmConditionsVec=real(GeneralEqmConditionsFn(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));

if heteroagentoptions.multiGEcriterion==0 
    GeneralEqmCondition=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market 
    GeneralEqmCondition=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));                                                                                                         
end

GeneralEqmCondition=gather(GeneralEqmCondition);

if heteroagentoptions.verbose==1
    fprintf('Current GE prices and GeneralEqmConditionsVec. \n')
    p
    GeneralEqmConditionsVec
end