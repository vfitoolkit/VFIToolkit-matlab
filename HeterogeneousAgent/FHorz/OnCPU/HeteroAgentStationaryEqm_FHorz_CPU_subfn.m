function GeneralEqmConditions=HeteroAgentStationaryEqm_FHorz_CPU_subfn(GEpricesvec, jequaloneDist,AgeWeightParamNames, n_d, n_a, n_z, N_j, pi_z_J, d_grid, a_grid, z_gridvals_J, ReturnFn, FnsToEvaluate, FnsToEvaluateCell, GeneralEqmEqnsCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions, simoptions, vfoptions)

heteroagentparamsvecindex=0:1:length(GEpricesvec);
[GEpricesvec,~]=ParameterConstraints_TransformParamsToOriginal(GEpricesvec,heteroagentparamsvecindex,GEPriceParamNames,heteroagentoptions);

%% 
for ii=1:nGEprices
    Parameters.(GEPriceParamNames{ii})=GEpricesvec(ii);
end

%%
[V, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

%Step 2: Calculate the Steady-state distn (given this price) and use it to assess market clearance
StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z,N_j, d_grid, a_grid, z_gridvals_J,simoptions);


%% Put GE parameters  and AggVars in structure, so they can be used for intermediateEqns and GeneralEqmEqns
% already did the basic GE params
% for pp=1:nGEprices
%     Parameters.(GEPriceParamNames{pp})=GEprices(pp);
% end
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars(aa);
end


%% Evaluate General Eqm Eqns
% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
GeneralEqmConditionsVec=zeros(1,length(GEeqnNames));
for gg=1:length(GEeqnNames)
    GeneralEqmConditionsVec(gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
end

%% We might want to output GE conditions as a vector or structure
if heteroagentoptions.outputGEform==0 % scalar
    if heteroagentoptions.multiGEcriterion==0
        GeneralEqmConditions=sum(abs(heteroagentoptions.multiGEweights.*GeneralEqmConditionsVec));
    elseif heteroagentoptions.multiGEcriterion==1 % the measure of market clearance is to take the sum of squares of clearance in each market
        GeneralEqmConditions=sqrt(sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2)));
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
    for gg=1:length(GEeqnNames)
        GeneralEqmConditions.(GEeqnNames{gg})=gather(GEeqnNames(gg));
    end
end


%% Feedback on progress
if heteroagentoptions.verbose==1
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        fprintf('	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(pp))
    end
    fprintf('Current aggregate variables: \n')
    for aa=1:length(AggVarNames)
        fprintf('	%s: %8.4f \n',AggVarNames{aa},AggVars(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
    end
    if isfield(heteroagentoptions,'intermediateEqns')
        fprintf('Current intermediateEqn variables: \n')
        for aa=1:length(intEqnnames)
            fprintf('	%s: %8.4f \n',intEqnnames{aa},intermediateEqnsVec(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
    end
    if heteroagentoptions.useCustomModelStats==1
        fprintf('Current CustomModelStats variables: \n')
        for ii=1:length(customstatnames)
            fprintf('	%s: %8.4f \n',customstatnames{ii},CustomStats.(customstatnames{ii}))
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    for gg=1:length(GEeqnNames)
        fprintf('	%s: %8.4f \n',GEeqnNames{gg},GeneralEqmConditionsVec(gg))
    end
end


% If recording the price history, do that
if heteroagentoptions.pricehistory==1
    load pricehistory.mat GEpricepath GEcondnpath itercount
    itercount=itercount+1;
    GEpricepath(:,itercount)=GEpricesvec;
    GEcondnpath(:,itercount)=GeneralEqmConditionsVec;
    save pricehistory.mat GEpricepath GEcondnpath itercount
end


end