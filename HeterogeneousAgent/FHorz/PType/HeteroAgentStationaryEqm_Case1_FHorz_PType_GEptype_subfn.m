function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_FHorz_PType_GEptype_subfn(GEprices, PTypeStructure, Parameters, GeneralEqmEqns,  GeneralEqmEqnsCell, GeneralEqmEqnParamNames,GEPriceParamNames, AggVarNames, nGEprices, GEpriceindexes, GEprice_ptype, heteroagentoptions)

%%
for pp=1:nGEprices % Not sure this is needed, have it just in case they are used when calling 'GeneralEqmConditionsFn', but I am pretty sure they never would be.
    Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
end

AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i,'gpuArray'); % Create AggVars conditional on ptype.

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        if GEprice_ptype(pp)==0
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1));
        else
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1)+ii-1);
        end
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

    AggVars_ConditionalOnPType(PTypeStructure.(iistr).FnsAndPTypeIndicator_ii,ii)=AggVars_ii;
end
AggVars=sum(AggVars_ConditionalOnPType.*PTypeStructure.ptweights,2);
% Note: AggVars is a vector

% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
GEeqnnames=fieldnames(GeneralEqmEqns);
GeneralEqmConditionsVec=zeros(1,sum(heteroagentoptions.GEptype==0)+PTypeStructure.N_i*sum(heteroagentoptions.GEptype==1));
% Some general eqm conditions are conditional on ptype, so go through one by one
gg_c=0;
for gg=1:length(GEeqnnames)
    if heteroagentoptions.GEptype(gg)==0 % Standard general eqm condition
        for pp=1:nGEprices
            Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
        end
        for aa=1:length(AggVarNames)
            Parameters.(AggVarNames{aa})=AggVars(aa);
        end
        gg_c=gg_c+1;
        GeneralEqmConditionsVec(gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
    elseif heteroagentoptions.GEptype(gg)==1 % Do this general eqm condition conditional on ptype
        for ii=1:PTypeStructure.N_i % This General eqm condition has to hold conditional on each ptype
            for pp=1:length(GEPriceParamNames)
                if GEprice_ptype(pp)==0
                    Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1));
                else
                    Parameters.(GEPriceParamNames{pp})=GEprices(GEpriceindexes(pp,1)+ii-1); % value specific to ptype
                end
            end
            for aa=1:length(AggVarNames)
                Parameters.(AggVarNames{aa})=AggVars_ConditionalOnPType(aa,ii);
            end
            gg_c=gg_c+1;
            GeneralEqmConditionsVec(gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
        end
    end
end

% We might want to output GE conditions as a vector or structure
if heteroagentoptions.outputGEform==0 % scalar
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 %the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));
    end
    if heteroagentoptions.outputgather==1
        GeneralEqmConditions=gather(GeneralEqmConditions);
    end
elseif heteroagentoptions.outputGEform==1 % vector
    GeneralEqmConditions=GeneralEqmConditionsVec;
    if heteroagentoptions.outputgather==1
        GeneralEqmConditions=gather(GeneralEqmConditions);
    end
elseif heteroagentoptions.outputGEform==2 % structure
    clear GeneralEqmConditions
    gg_c=1;
    for gg=1:length(GEeqnnames)
        if heteroagentoptions.GEptype(gg)==0 % Standard general eqm condition
            GeneralEqmConditions.(GEeqnnames{gg})=GeneralEqmConditionsVec(gg_c);
            gg_c=gg_c+1;
        elseif heteroagentoptions.GEptype(gg)==1 % Do this general eqm condition conditional on ptype
            GeneralEqmConditions.(GEeqnnames{gg})=GeneralEqmConditionsVec(gg_c:gg_c+PTypeStructure.N_i-1);
            gg_c=gg_c+PTypeStructure.N_i;
        end
    end
end


if heteroagentoptions.verbose==1
    if all(heteroagentoptions.GEptype==0)
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
    else
        numberstr=repmat(' %8.4f',1,PTypeStructure.N_i);
        % Adjust output for the fact there are multiple ptypes
        fprintf(' \n')
        fprintf('Current GE prices: \n')
        for pp=1:nGEprices
            if GEprice_ptype(pp)==1
                fprintf(['	%s:',numberstr,' \n'],GEPriceParamNames{pp},GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))
            else
                fprintf(['	%s: %8.4f \n'],GEPriceParamNames{pp},GEprices(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))
            end
        end
        fprintf('Current aggregate variables: \n')
        for aa=1:length(AggVarNames)
            fprintf('	%s: %8.4f \n',AggVarNames{aa},AggVars(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
        fprintf('Current aggregate variables, conditional on ptype: \n')
        for aa=1:length(AggVarNames)
            fprintf(['	%s:',numberstr,' \n'],AggVarNames{aa},AggVars_ConditionalOnPType(aa,:)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
        fprintf('Current GeneralEqmEqns: \n')
        GeneralEqmEqnsNames=fieldnames(GeneralEqmEqns);
        ggindex=[ones(length(GeneralEqmEqnsNames),1)+heteroagentoptions.GEptype'*(PTypeStructure.N_i-1)];
        ggindex=[[1; cumsum(ggindex(1:end-1))+1],cumsum(ggindex)];
        for gg=1:length(GeneralEqmEqnsNames)
            if heteroagentoptions.GEptype(gg)==1
                fprintf(['	%s:',numberstr,' \n'],GeneralEqmEqnsNames{gg},GeneralEqmConditionsVec(ggindex(gg,1):ggindex(gg,2)))
            else
                fprintf('	%s: %8.4f \n',GeneralEqmEqnsNames{gg},GeneralEqmConditionsVec(ggindex(gg,1):ggindex(gg,2)))            
            end
        end
    end
end



if heteroagentoptions.saveprogresseachiter==1
    save HeterAgentEqm_internal.mat GEprices Parameters GeneralEqmConditionsVec
end

end