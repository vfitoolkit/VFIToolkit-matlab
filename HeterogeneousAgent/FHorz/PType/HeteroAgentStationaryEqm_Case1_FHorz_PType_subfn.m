function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_FHorz_PType_subfn(GEprices, PTypeStructure, Parameters, GeneralEqmEqns, GEPriceParamNames, AggVarNames, nGEprices, heteroagentoptions)


%%
for pp=1:nGEprices % Not sure this is needed, have it just in case they are used when calling 'GeneralEqmConditionsFn', but I am pretty sure they never would be.
    Parameters.(GEPriceParamNames{pp})=GEprices(pp);
end

if heteroagentoptions.parallel==2
    AggVars=zeros(PTypeStructure.numFnsToEvaluate,1,'gpuArray'); % numFnsToEvaluate is independent of the ptype
else
    AggVars=zeros(PTypeStructure.numFnsToEvaluate,1); % numFnsToEvaluate is independent of the ptype
end


for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEprices(pp);
    end
    
    
    if isfinite(PTypeStructure.(iistr).N_j)
        [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, [], PTypeStructure.(iistr).simoptions);
    else  % PType actually allows for infinite horizon as well
        [~, Policy_ii]=ValueFnIter_Case1(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).Parameters);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, [], PTypeStructure.(iistr).simoptions);     
    end
    
    for kk=1:PTypeStructure.numFnsToEvaluate
        jj=PTypeStructure.(iistr).WhichFnsForCurrentPType(kk);
        if jj>0
            AggVars(kk)=AggVars(kk)+PTypeStructure.(iistr).PTypeWeight*AggVars_ii(jj);
        end
    end

end

% Note: AggVars is a matrix

% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
% if isstruct(GeneralEqmEqns)
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars(aa);
end
GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns, Parameters));
% end


% We might want to output GE conditions as a vector or structure
if heteroagentoptions.outputGEform==0 % scalar
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));
    end
    GeneralEqmConditions=gather(GeneralEqmConditions);
elseif heteroagentoptions.outputGEform==1 % vector
    GeneralEqmConditions=GeneralEqmConditionsVec;
elseif heteroagentoptions.outputGEform==2 % structure
    clear GeneralEqmConditions
    GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
    for ii=1:length(GeneralEqmEqnsNames)
        GeneralEqmConditions.(GeneralEqmEqnsNames{ii})=GeneralEqmConditionsVec(ii);
    end
end

if heteroagentoptions.verbose==1
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for ii=1:nGEprices
        fprintf('	%s: %8.4f \n',GEPriceParamNames{ii},GEprices(ii))
    end
    fprintf('Current aggregate variables: \n')
    for ii=1:length(AggVarNames)
        fprintf('	%s: %8.4f \n',AggVarNames{ii},Parameters.(AggVarNames{ii})) % Note, this is done differently here because AggVars itself has been set as a matrix
    end
    fprintf('Current GeneralEqmEqns: \n')
    GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
    for ii=1:length(GeneralEqmEqnsNames)
        fprintf('	%s: %8.4f \n',GeneralEqmEqnsNames{ii},GeneralEqmConditionsVec(ii))
    end
end



if heteroagentoptions.saveprogresseachiter==1
    save HeterAgentEqm_internal.mat GEprices Parameters GeneralEqmConditionsVec
end

end