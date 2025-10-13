function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_FHorz_PType_subfn(GEpricesvec, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions)

heteroagentparamsvecindex=0:1:length(GEpricesvec);
[GEpricesvec,penalty]=ParameterConstraints_TransformParamsToOriginal(GEpricesvec,heteroagentparamsvecindex,GEPriceParamNames,heteroagentoptions);

%% 
for pp=1:nGEprices
    Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
end

% If z (and e) are determined in GE
if any(heteroagentoptions.gridsinGE)
    for ii=1:PTypeStructure.N_i
        if heteroagentoptions.gridsinGE(ii)==1
            iistr=PTypeStructure.iistr{ii};
            % Some of the shock grids depend on parameters that are determined in general eqm
            [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
            % Note: these are actually z_gridvals and pi_z
            PTypeStructure.(iistr).vfoptions
            PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J; % Note, will be [] if no e
            PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J; % Note, will be [] if no e
        end
    end
end

%%
AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i,'gpuArray'); % Create AggVars conditional on ptype.

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
    end

    if heteroagentoptions.gridsinGE(ii)==1
        % Some of the shock grids depend on parameters that are determined in general eqm
        [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        % Convert z and e to age-dependent joint-grids and transtion matrix
        % Note: Ignores which, just redoes both z and e
        PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J; % if no e, this is just empty anyway
        PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J;
    end

    
    if isfinite(PTypeStructure.(iistr).N_j)
        [~, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).simoptions);
    else  % PType actually allows for infinite horizon as well
        [~, Policy_ii]=ValueFnIter_Case1(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).Parameters);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).simoptions);
    end
    
    AggVars_ConditionalOnPType(PTypeStructure.(iistr).FnsAndPTypeIndicator_ii,ii)=AggVars_ii;
end
AggVars=gather(sum(AggVars_ConditionalOnPType.*PTypeStructure.ptweights,2));
% Note: AggVars is a vector



%% Put GE parameters  and AggVars in structure, so they can be used for intermediateEqns and GeneralEqmEqns
% already did the basic GE params
% for pp=1:nGEprices
%     Parameters.(GEPriceParamNames{pp})=GEprices(pp);
% end
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars(aa);
end

%% Intermediate Eqns
if heteroagentoptions.useintermediateEqns==1
    % Note: intermediateEqns just take in things from the Parameters structure, as do GeneralEqmEqns (AggVars get put into structure), hence just use the GeneralEqmConditions_Case1_v3g().
    intEqnnames=fieldnames(heteroagentoptions.intermediateEqns);
    intermediateEqnsVec=zeros(1,length(intEqnnames));
    % Do the intermediateEqns, in order
    for gg=1:length(intEqnnames)
        intermediateEqnsVec(gg)=real(GeneralEqmConditions_Case1_v3g(heteroagentoptions.intermediateEqnsCell{gg}, heteroagentoptions.intermediateEqnParamNames(gg).Names, Parameters));
        Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg);
    end
end

%% Custom Model Stats
if heteroagentoptions.useCustomModelStats==1
    CustomStats=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,heteroagentoptions,vfoptions,simoptions);
    % Note: anything else you want, just 'hide' it in heteroagentoptions
    customstatnames=fieldnames(CustomStats);
    for pp=1:length(customstatnames)
        Parameters.(customstatnames{pp})=CustomStats.(customstatnames{pp});
    end
end



%% Evaluate General Eqm Eqns
% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
GeneralEqmConditionsVec=zeros(1,length(GEeqnNames));
for gg=1:length(GEeqnNames)
    GeneralEqmConditionsVec(gg)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
end


%% We might want to output GE conditions as a vector or structure
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
    for gg=1:length(GEeqnNames)
        GeneralEqmConditions.(GEeqnNames{gg})=GEeqnNames(gg);
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
    if heteroagentoptions.useintermediateEqns==1
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
