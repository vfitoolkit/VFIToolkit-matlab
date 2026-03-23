function GeneralEqmConditions=HeteroAgentStationaryEqm_Case1_PType_subfn(GEpricesvec, PTypeStructure, Parameters, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, GEeqnNames, AggVarNames, nGEprices, heteroagentoptions)

heteroagentparamsvecindex=0:1:length(GEpricesvec);
[GEpricesvec,penalty]=ParameterConstraints_TransformParamsToOriginal(GEpricesvec,heteroagentparamsvecindex,GEPriceParamNames,heteroagentoptions);

if heteroagentoptions.verbose==2
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        fprintf('	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(pp))
    end
end

%% 
for pp=1:nGEprices
    Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
end

% If z (and e) are determined in GE
if any(heteroagentoptions.gridsinGE)
    for ii=1:PTypeStructure.N_i
        if heteroagentoptions.gridsinGE(ii)==0
            iistr=PTypeStructure.iistr{ii};
            % Some of the shock grids depend on parameters that are determined in general eqm
            [PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).vfoptions]=ExogShockSetup(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
            % Note: these are actually z_gridvals and pi_z
            PTypeStructure.(iistr).simoptions.e_gridvals=PTypeStructure.(iistr).vfoptions.e_gridvals; % Note, will be [] if no e
            PTypeStructure.(iistr).simoptions.pi_e=PTypeStructure.(iistr).vfoptions.pi_e; % Note, will be [] if no e
        end
    end
end


%%
Ptype_cells=cell(1,PTypeStructure.N_i); % Hold results in case needed for CustomStats
AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i); % Create AggVars conditional on ptype.
AggVars=zeros(PTypeStructure.numFnsToEvaluate,1,'gpuArray'); % numFnsToEvaluate is independent of the ptype

for ii=1:PTypeStructure.N_i
    
    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEpricesvec(pp);
    end

    if heteroagentoptions.gridsinGE(ii)==1
        % Some of the shock grids depend on parameters that are determined in general eqm
        [PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).vfoptions]=ExogShockSetup(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
        % Convert z and e to age-dependent joint-grids and transtion matrix
        % Note: Ignores which, just redoes both z and e
        PTypeStructure.(iistr).simoptions.e_gridvals=PTypeStructure.(iistr).vfoptions.e_gridvals; % if no e, this is just empty anyway
        PTypeStructure.(iistr).simoptions.pi_e=PTypeStructure.(iistr).vfoptions.pi_e;
    end

    
    [V_ii, Policy_ii]=ValueFnIter_Case1(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).ReturnFn, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, PTypeStructure.(iistr).vfoptions);
    StationaryDist_ii=StationaryDist_Case1(Policy_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions,PTypeStructure.(iistr).Parameters);
    % PTypeStructure.(iistr).simoptions.outputasstructure=0; % Want AggVars_ii as matrix to make it easier to add them across the PTypes (is set outside this script)
    AggVars_ii=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_ii, Policy_ii, PTypeStructure.(iistr).FnsToEvaluate, PTypeStructure.(iistr).Parameters, PTypeStructure.(iistr).FnsToEvaluateParamNames, PTypeStructure.(iistr).n_d, PTypeStructure.(iistr).n_a, PTypeStructure.(iistr).n_z, PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid, PTypeStructure.(iistr).simoptions);
    if heteroagentoptions.useCustomModelStats==1
        Ptype_cells{ii}={V_ii,Policy_ii,StationaryDist_ii};
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
AggVars=sum(AggVars_ConditionalOnPType.*PTypeStructure.ptweights,2);
% Note: AggVars is a vector


%% Put GE parameters  and AggVars in structure, so they can be used for intermediateEqns and GeneralEqmEqns
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
    CustomStats=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,PTypeStructure.Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,heteroagentoptions.CustomModelStatsInputs.z_grid,heteroagentoptions.CustomModelStatsInputs.pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
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

%% Custom Model Stats
customstatnames=struct();
if heteroagentoptions.useCustomModelStats==1
    if isfield(heteroagentoptions, 'CustomModelStats')
        error("Universal PType handler for CustomModelStats not yet implemented")
    else
        for ii=1:PTypeStructure.N_i
            iistr=PTypeStructure.iistr{ii};
            if ~isfield(heteroagentoptions, iistr) || ~isfield(heteroagentoptions.(iistr), 'CustomModelStats')
                continue
            end
            CustomStats.(iistr)=heteroagentoptions.(iistr).CustomModelStats(Ptype_cells{ii}{1},Ptype_cells{ii}{2},Ptype_cells{ii}{3},PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).FnsToEvaluate,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).a_grid,PTypeStructure.(iistr).z_gridvals,PTypeStructure.(iistr).pi_z,heteroagentoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
            % Note: anything else you want, just 'hide' it in heteroagentoptions
            customstatnames.(iistr)=fieldnames(CustomStats.(iistr));
            for pp=1:length(customstatnames.(iistr))
                PTypeStructure.(iistr).Parameters.(customstatnames.(iistr){pp})=CustomStats.(iistr).(customstatnames.(iistr){pp});
            end
        end
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
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        fprintf(heteroagentoptions.verboseaccuracy1,GEPriceParamNames{pp},GEpricesvec(pp))
    end
end
if heteroagentoptions.verbose>=1
    fprintf('Current aggregate variables: \n')
    for aa=1:length(AggVarNames)
        fprintf(heteroagentoptions.verboseaccuracy1,AggVarNames{aa},AggVars(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
    end
    if heteroagentoptions.useintermediateEqns==1
        fprintf('Current intermediateEqn variables: \n')
        for aa=1:length(intEqnnames)
            fprintf(heteroagentoptions.verboseaccuracy1,intEqnnames{aa},intermediateEqnsVec(aa)) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
    end
    if heteroagentoptions.useCustomModelStats==1
        fprintf('Current CustomModelStats variables: \n')
        for ii=1:length(customstatnames)
            fprintf(heteroagentoptions.verboseaccuracy1,customstatnames{ii},CustomStats.(customstatnames{ii}))
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    for gg=1:length(GEeqnNames)
        fprintf(heteroagentoptions.verboseaccuracy2,GEeqnNames{gg},GeneralEqmConditionsVec(gg))
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