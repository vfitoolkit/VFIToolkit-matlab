function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_MixHorz_PType_subfn(GEpricesvec, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions)

heteroagentparamsvecindex=0:1:length(GEpricesvec);
[GEpricesvec,penalty]=ParameterConstraints_TransformParamsToOriginal(GEpricesvec,heteroagentparamsvecindex,GEPriceParamNames,heteroagentoptions);

if heteroagentoptions.verbose>0
    GEpricesvec_tminus1=zeros(nGEprices,1);
    AggVars_tminus1=NaN(length(AggVarNames),1);

    for pp=1:nGEprices
        GEpricesvec_tminus1(pp)=Parameters.(GEPriceParamNames{pp});
    end
    GEpricesvec_delta=GEpricesvec-GEpricesvec_tminus1; % Compute this to show max change per round
    for aa=1:length(AggVarNames)
        if isfield(Parameters,AggVarNames{aa})
            AggVars_tminus1(aa)=Parameters.(AggVarNames{aa});
        end
    end
    if heteroagentoptions.useintermediateEqns==1
        intEqnnames=fieldnames(heteroagentoptions.intermediateEqns);
        intermediateEqns_tminus1=zeros(length(intEqnnames),1);
        for aa=1:length(intEqnnames)
            if isfield(Parameters,intEqnnames{aa})
                intEqns_tminus1(aa)=Parameters.(intEqnnames{aa});
            end
        end
    end
    % We don't do anything special for CustomModelStats, which are not as easily done as others above.
end

if heteroagentoptions.verbose==2
    [~,maxidx]=max(GEpricesvec_delta.^2);
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        if pp==maxidx
            cprintf('err','	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(pp))
        else
            fprintf('	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(pp))
        end
    end
end

%% 
for pp=1:nGEprices
    Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
end


%%
AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i); % Create AggVars conditional on ptype.

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
    end
    
    if heteroagentoptions.gridsinGE(ii)==1
        if isfinite(PTypeStructure.(iistr).N_j)
            % Some of the shock grids depend on parameters that are determined in general eqm
            [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
            % Convert z and e to age-dependent joint-grids and transtion matrix
            % Note: Ignores which, just redoes both z and e
            PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J; % if no e, this is just empty anyway
            PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J;
        else
            % PType actually allows for infinite horizon as well
            % Some of the shock grids depend on parameters that are determined in general eqm
            [PTypeStructure.(iistr).z_gridvals, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_InfHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
            % Convert z and e to joint-grids and transtion matrix
            % Note: Ignores which, just redoes both z and e
            PTypeStructure.(iistr).simoptions.e_gridvals=PTypeStructure.(iistr).vfoptions.e_gridvals; % if no e, this is just empty anyway
            PTypeStructure.(iistr).simoptions.pi_e=PTypeStructure.(iistr).vfoptions.pi_e;
        end
    end
    
    % If semiz is determined in GE
    if heteroagentoptions.gridsinGE_semiexo(ii)==1
        if isfinite(PTypeStructure.(iistr).N_j)
            % Some of the shock grids depend on parameters that are determined in general eqm
            PTypeStructure.(iistr).vfoptions=SemiExogShockSetup_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,2,3);
            PTypeStructure.(iistr).simoptions.semiz_gridvals_J=PTypeStructure.(iistr).vfoptions.semiz_gridvals_J;
            PTypeStructure.(iistr).simoptions.pi_semiz_J=PTypeStructure.(iistr).vfoptions.pi_semiz_J;
        else
            error('Semiexog in InfHorz not yet implemented')
        end
    end
    
    if isfinite(PTypeStructure.(iistr).N_j)
        [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).N_j, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).simoptions);
    else  % PType actually allows for infinite horizon as well
        [V_ii, Policy_ii]=ValueFnIter_InfHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
        StationaryDist_ii=StationaryDist_InfHorz(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).Parameters);
        % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
        AggVars_ii=EvalFnOnAgentDist_AggVars_InfHorz(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals, PTypeStructure.(iistr).simoptions);
    end
    AggVars_ConditionalOnPType(PTypeStructure.(iistr).FnsAndPTypeIndicator_ii,ii)=AggVars_ii;
    % Put updated AggVars into subsequent PTypeStructure Parameters, so they can be used for subsequent PType evaluations
    FnsToEvaluate_aa=fieldnames(PTypeStructure.(iistr).FnsToEvaluate);
    for jj=ii+1:PTypeStructure.N_i
        jjstr=PTypeStructure.iistr{jj};
        for aa=1:length(AggVars_ii)
            PTypeStructure.(jjstr).Parameters.(FnsToEvaluate_aa{aa})=AggVars_ii(aa);
        end
    end

    if heteroagentoptions.useCustomModelStats==1
        V.(iistr)=V_ii;
        Policy.(iistr)=Policy_ii;
        StationaryDist.(iistr)=StationaryDist_ii;
    end
end
AggVars=sum(AggVars_ConditionalOnPType.*PTypeStructure.ptweights',2);
% Note: AggVars is a vector



%% Put GE parameters and AggVars in structure, so they can be used for intermediateEqns and GeneralEqmEqns
% already did the basic GE params
% for pp=1:nGEprices
%     Parameters.(GEPriceParamNames{pp})=GEprices(pp);
% end

% We pushed AggVars down into the PTypeStructure parameters; this puts them into the unified Parameter structure
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars(aa);
end

%% Custom Model Stats
if heteroagentoptions.useCustomModelStats==1
    StationaryDist.ptweights=PTypeStructure.ptweights;
    % A bunch of the inputs are stashed in heteroagentoptions.CustomModelStatsInputs
    % Note: CustomStats deliberately does not get AgeWeightParamNames and PTypeDistParamNames, user will anyway know them
    CustomStats=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,heteroagentoptions.CustomModelStatsInputs.N_j,PTypeStructure.Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,heteroagentoptions.CustomModelStatsInputs.z_grid,heteroagentoptions.CustomModelStatsInputs.pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    % Note: anything else you want, just 'hide' it in heteroagentoptions
    customstatnames=fieldnames(CustomStats);
    for pp=1:length(customstatnames)
        Parameters.(customstatnames{pp})=CustomStats.(customstatnames{pp});
    end
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
        GeneralEqmConditions.(GEeqnNames{gg})=GeneralEqmConditionsVec(gg);
    end
end

%% Feedback on progress
if heteroagentoptions.verbose==1 % When=2, we report these earlier
    [~,maxidx]=max(GEpricesvec_delta.^2);
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        if pp==maxidx
            cprintf('err',heteroagentoptions.verboseaccuracy1,GEPriceParamNames{pp},GEpricesvec(pp))
        else
            fprintf(heteroagentoptions.verboseaccuracy1,GEPriceParamNames{pp},GEpricesvec(pp))
        end
    end
end
if heteroagentoptions.verbose>=1
    fprintf('Current aggregate variables: \n')
    for aa=1:length(AggVarNames)
        if ~isnan(AggVars_tminus1(aa))
            cprintf('comment',heteroagentoptions.verboseaccuracy1,AggVarNames{aa},AggVars(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
        else
            fprintf(heteroagentoptions.verboseaccuracy1,AggVarNames{aa},AggVars(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
    end
    if heteroagentoptions.useintermediateEqns==1
        intEqns_delta=intermediateEqnsVec-intermediateEqns_tminus1;
        [~,maxidx]=max(intEqns_delta.^2);
        fprintf('Current intermediateEqn variables: \n')
        for aa=1:length(intEqnnames)
            if aa==maxidx
                cprintf('err',heteroagentoptions.verboseaccuracy1,intEqnnames{aa},intermediateEqnsVec(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
            else
                fprintf(heteroagentoptions.verboseaccuracy1,intEqnnames{aa},intermediateEqnsVec(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
            end
        end
    end
    if heteroagentoptions.useCustomModelStats==1
        fprintf('Current CustomModelStats variables: \n')
        for aa=1:length(customstatnames)
            fprintf(heteroagentoptions.verboseaccuracy1,customstatnames{aa},CustomStats.(customstatnames{aa}))
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    [~,maxidx]=max(GeneralEqmConditionsVec.^2);
    for gg=1:length(GEeqnNames)
        if gg==maxidx
            cprintf('err', heteroagentoptions.verboseaccuracy2,GEeqnNames{gg},GeneralEqmConditionsVec(gg))
        else
            fprintf(heteroagentoptions.verboseaccuracy2,GEeqnNames{gg},GeneralEqmConditionsVec(gg))
        end
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
