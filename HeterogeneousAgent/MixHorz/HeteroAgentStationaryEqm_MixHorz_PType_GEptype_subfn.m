function GeneralEqmConditions=HeteroAgentStationaryEqm_MixHorz_PType_GEptype_subfn(GEpricesvec, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, GEPriceParamNames, AggVarNames, nGEprices, GEpriceindexes, GEprice_ptype, heteroagentoptions)
% As HeteroAgentStationaryEqm_MixHorz_PType_subfn, but some of the general eqm conditions hold
% CONDITIONAL ON THE PERMANENT TYPE (heteroagentoptions.GEptype), and so do the prices that clear
% them. The model solving is identical, mixed horizons included - it is the prices and the equations
% that change:
%  - a price that depends on ptype occupies N_i entries of GEpricesvec, not one (GEpriceindexes
%    says which, GEprice_ptype says whether)
%  - every AggVar, and every per-type price, is also put into Parameters under its '_name' version
%    (N_ptype001, and so on), because that is what the by-ptype equations are given as inputs
%  - the intermediate eqns and the general eqm eqns are evaluated N_i times where they are marked as
%    by-ptype, indexed by gg_c (condition-and-ptype) rather than by gg (condition)
% Mirrors HeteroAgentStationaryEqm_Case1_FHorz_PType_GEptype_subfn and the InfHorz one.
%
% The one thing dropped relative to the plain subfn: the highlighting of the largest change since the
% previous iteration is only applied to the economy-wide rows. A by-ptype row is N_i numbers on one
% line, so there is no single entry to colour.

GEeqnNames=fieldnames(GeneralEqmEqns); % the callers pass the struct, not the names

heteroagentparamsvecindex=[0; GEpriceindexes(:,2)];
[GEpricesvec,penalty]=ParameterConstraints_TransformParamsToOriginal(GEpricesvec,heteroagentparamsvecindex,GEPriceParamNames,heteroagentoptions);

if heteroagentoptions.verbose>0
    GEpricesvec_tminus1=zeros(length(GEpricesvec),1); % length(GEpricesvec), not nGEprices: a by-ptype price is N_i entries
    AggVars_tminus1=NaN(length(AggVarNames),1);

    for pp=1:nGEprices
        GEpricesvec_tminus1(GEpriceindexes(pp,1):GEpriceindexes(pp,2))=Parameters.(GEPriceParamNames{pp});
    end
    GEpricesvec_delta=GEpricesvec-GEpricesvec_tminus1; % Compute this to show max change per round
    for aa=1:length(AggVarNames)
        if isfield(Parameters,AggVarNames{aa})
            AggVars_tminus1(aa)=Parameters.(AggVarNames{aa});
        end
    end
    if heteroagentoptions.useintermediateEqns==1
        intEqnnames=fieldnames(heteroagentoptions.intermediateEqns);
        % No previous-iteration vector for the intermediate eqns here: when one of them is by-ptype the
        % vector of values is N_i-expanded and would not line up with one entry per equation.
    end
    % We don't do anything special for CustomModelStats, which are not as easily done as others above.
end

if heteroagentoptions.verbose==2
    [~,maxidx]=max(GEpricesvec_delta.^2);
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        if GEprice_ptype(pp)==1
            fprintf(heteroagentoptions.verboseaccuracy1ptype,GEPriceParamNames{pp},GEpricesvec(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))
        elseif GEpriceindexes(pp,1)==maxidx
            cprintf('err','	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(GEpriceindexes(pp,1)))
        else
            fprintf('	%s: %8.4f \n',GEPriceParamNames{pp},GEpricesvec(GEpriceindexes(pp,1)))
        end
    end
end

%%
for pp=1:nGEprices % a by-ptype price is the N_i entries GEpriceindexes points at, so this is a vector for those
    Parameters.(GEPriceParamNames{pp})=GEpricesvec(GEpriceindexes(pp,1):GEpriceindexes(pp,2));
end


%%
AggVars_ConditionalOnPType=zeros(PTypeStructure.numFnsToEvaluate,PTypeStructure.N_i); % Create AggVars conditional on ptype.

% The user's own grids are needed if CustomModelStats is given them
KeepOriginalGrid=(heteroagentoptions.useCustomModelStats==1 && heteroagentoptions.CustomModelStats_usergrids==1);
for ii=1:PTypeStructure.N_i

    iistr=PTypeStructure.iistr{ii};
    for pp=1:length(GEPriceParamNames)
        if GEprice_ptype(pp)==1 % this ptype gets its own value of this price
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEpricesvec(GEpriceindexes(pp,1)+ii-1);
        else
            PTypeStructure.(iistr).Parameters.(GEPriceParamNames{pp})=GEpricesvec(GEpriceindexes(pp,1));
        end
    end

    if heteroagentoptions.gridsinGE(ii)==1
        if isfinite(PTypeStructure.(iistr).N_j)
            % Some of the shock grids depend on parameters that are determined in general eqm
            % The user's own grids are also needed if jequaloneDist as a function is given them (FHorz types only)
            KeepOriginalGrid_ii=(PTypeStructure.(iistr).simoptions.jequaloneDist_usergrids==1 || KeepOriginalGrid==1);
            [PTypeStructure.(iistr).z_gridvals_J, PTypeStructure.(iistr).pi_z_J, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_FHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3,KeepOriginalGrid_ii);
            % Convert z and e to age-dependent joint-grids and transtion matrix
            % Note: Ignores which, just redoes both z and e
            PTypeStructure.(iistr).simoptions.e_gridvals_J=PTypeStructure.(iistr).vfoptions.e_gridvals_J; % if no e, this is just empty anyway
            PTypeStructure.(iistr).simoptions.pi_e_J=PTypeStructure.(iistr).vfoptions.pi_e_J;
            if PTypeStructure.(iistr).simoptions.jequaloneDist_usergrids==1 && isfield(PTypeStructure.(iistr).vfoptions,'user_z_grid') % jequaloneDist as a function is given the user's own grids, and they must track the general eqm prices
                PTypeStructure.(iistr).simoptions.user_z_grid=PTypeStructure.(iistr).vfoptions.user_z_grid;
                PTypeStructure.(iistr).simoptions.user_pi_z=PTypeStructure.(iistr).vfoptions.user_pi_z;
            end
        else
            % PType actually allows for infinite horizon as well
            % Some of the shock grids depend on parameters that are determined in general eqm
            [PTypeStructure.(iistr).z_gridvals, PTypeStructure.(iistr).pi_z, PTypeStructure.(iistr).vfoptions]=ExogShockSetup_InfHorz(PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).z_gridvals,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3,KeepOriginalGrid);
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
            PTypeStructure.(iistr).vfoptions=SemiExogShockSetup_FHorz(PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).d_grid,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).vfoptions,3);
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
% Do the general eqm parameters that depend on ptype
for pp=1:length(GEPriceParamNames)
    if GEprice_ptype(pp)==1
        for ii=1:PTypeStructure.N_i
            Parameters.([GEPriceParamNames{pp},'_',PTypeStructure.Names_i{ii}])=GEpricesvec(GEpriceindexes(pp,1)+ii-1);
        end
    end
end
% And do all the AggVars as well
for aa=1:length(AggVarNames)
    for ii=1:PTypeStructure.N_i
        Parameters.([AggVarNames{aa},'_',PTypeStructure.Names_i{ii}])=AggVars_ConditionalOnPType(aa,ii); % Note: this will create AggVar values of zero, even where they 'dont exist', but I don't think the user will get that wrong
    end
end

%% Custom Model Stats
if heteroagentoptions.useCustomModelStats==1
    StationaryDist.ptweights=PTypeStructure.ptweights;
    if heteroagentoptions.CustomModelStats_usergrids==0 || any(heteroagentoptions.gridsinGE)
        % Internal grids, as a struct with one field per ptype [z_gridvals_J & pi_z_J for FHorz ptypes, z_gridvals & pi_z for InfHorz ptypes]
        for ii=1:PTypeStructure.N_i
            iistr=PTypeStructure.iistr{ii};
            if isfinite(PTypeStructure.(iistr).N_j)
                z_gridvals_PType.(iistr)=PTypeStructure.(iistr).z_gridvals_J;
                pi_z_PType.(iistr)=PTypeStructure.(iistr).pi_z_J;
            else
                z_gridvals_PType.(iistr)=PTypeStructure.(iistr).z_gridvals;
                pi_z_PType.(iistr)=PTypeStructure.(iistr).pi_z;
            end
        end
    end
    % A bunch of the inputs are stashed in heteroagentoptions.CustomModelStatsInputs
    % Note: CustomStats deliberately does not get AgeWeightParamNames and PTypeDistParamNames, user will anyway know them
    if heteroagentoptions.CustomModelStats_usergrids==0
        CustomStats=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,heteroagentoptions.CustomModelStatsInputs.N_j,PTypeStructure.Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,z_gridvals_PType,pi_z_PType,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    elseif heteroagentoptions.CustomModelStats_usergrids==1
        if any(heteroagentoptions.gridsinGE) % grids depend on GE prices (for at least one ptype), so give the user's own grids as rebuilt from the current prices
            for ii=1:PTypeStructure.N_i
                iistr=PTypeStructure.iistr{ii};
                user_z_grid_PType.(iistr)=PTypeStructure.(iistr).vfoptions.user_z_grid;
                user_pi_z_PType.(iistr)=PTypeStructure.(iistr).vfoptions.user_pi_z;
            end
            heteroagentoptions.CustomModelStatsInputs.z_grid=user_z_grid_PType;
            heteroagentoptions.CustomModelStatsInputs.pi_z=user_pi_z_PType;
        end
        CustomStats=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,heteroagentoptions.CustomModelStatsInputs.N_j,PTypeStructure.Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,heteroagentoptions.CustomModelStatsInputs.z_grid,heteroagentoptions.CustomModelStatsInputs.pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    end
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
    intermediateEqnsVec=zeros(1,sum(heteroagentoptions.intermediateEqnsptype==0)+PTypeStructure.N_i*sum(heteroagentoptions.intermediateEqnsptype==1));
    % Do the intermediateEqns, in order
    gg_c=0;
    for gg=1:length(intEqnnames)
        if heteroagentoptions.intermediateEqnsptype(gg)==0 % standard intermediateEqn
            gg_c=gg_c+1;
            intermediateEqnsVec(gg_c)=GeneralEqmConditions_Case1_v3g(heteroagentoptions.intermediateEqnsCell{gg}, heteroagentoptions.intermediateEqnParamNames(gg_c).Names, Parameters);
            Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg_c);

            % if the intermediateEqn is using '_name', then put it into Params as a structure with name
            intEqnnames_gg=intEqnnames{gg};
            if contains(intEqnnames_gg,'_') % potential uses '_name'
                for ii=1:PTypeStructure.N_i
                    lname=length(PTypeStructure.Names_i{ii});
                    if length(intEqnnames_gg)>lname+1 % only check if intEqnnames_gg is long enough to be possible
                        if strcmp(intEqnnames_gg(end-lname:end),['_',PTypeStructure.Names_i{ii}])
                            Parameters.(intEqnnames_gg(1:end-lname-1)).(PTypeStructure.Names_i{ii})=Parameters.(intEqnnames{gg});
                            % E.g., creates Parameters.r.ptype001 from Parameters.r_ptype001
                        end
                    end
                end
            end
        elseif heteroagentoptions.intermediateEqnsptype(gg)==1 % Do this intermediateEqn conditional on ptype
            for ii=1:PTypeStructure.N_i % This one has to hold conditional on each ptype
                gg_c=gg_c+1;
                intermediateEqnsVec(gg_c)=GeneralEqmConditions_Case1_v3g_ptype(heteroagentoptions.intermediateEqnsCell{gg}, heteroagentoptions.intermediateEqnParamNames(gg_c).Names, Parameters);
                Parameters.(intEqnnames{gg}).(PTypeStructure.Names_i{ii})=intermediateEqnsVec(gg_c);
                % also, just in case they need to be used again, add the _name version
                Parameters.([intEqnnames{gg},'_',PTypeStructure.Names_i{ii}])=intermediateEqnsVec(gg_c);
            end
        end
    end
end

%% Evaluate General Eqm Eqns
% use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
% use of real() has been disabled, see how it goes without
GeneralEqmConditionsVec=zeros(1,sum(heteroagentoptions.GEptype==0)+PTypeStructure.N_i*sum(heteroagentoptions.GEptype==1));
% Some general eqm conditions are conditional on ptype, so go through one by one
gg_c=0;
for gg=1:length(GEeqnNames)
    if heteroagentoptions.GEptype(gg)==0 % Standard general eqm condition
        gg_c=gg_c+1;
        GeneralEqmConditionsVec(gg_c)=GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg_c).Names, Parameters);
    elseif heteroagentoptions.GEptype(gg)==1 % Do this general eqm condition conditional on ptype
        for ii=1:PTypeStructure.N_i % This General eqm condition has to hold conditional on each ptype
            gg_c=gg_c+1;
            GeneralEqmConditionsVec(gg_c)=GeneralEqmConditions_Case1_v3g_ptype(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg_c).Names, Parameters);
        end
    end
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
    gg_c=1;
    for gg=1:length(GEeqnNames)
        if heteroagentoptions.GEptype(gg)==0 % Standard general eqm condition
            GeneralEqmConditions.(GEeqnNames{gg})=GeneralEqmConditionsVec(gg_c);
            gg_c=gg_c+1;
        elseif heteroagentoptions.GEptype(gg)==1 % this one holds conditional on ptype, so it is N_i values
            GeneralEqmConditions.(GEeqnNames{gg})=GeneralEqmConditionsVec(gg_c:gg_c+PTypeStructure.N_i-1);
            gg_c=gg_c+PTypeStructure.N_i;
        end
    end
end

%% Feedback on progress
if heteroagentoptions.verbose==1 % When=2, we report these earlier
    [~,maxidx]=max(GEpricesvec_delta.^2);
    fprintf(' \n')
    fprintf('Current GE prices: \n')
    for pp=1:nGEprices
        if GEprice_ptype(pp)==1
            fprintf(heteroagentoptions.verboseaccuracy1ptype,GEPriceParamNames{pp},GEpricesvec(GEpriceindexes(pp,1):GEpriceindexes(pp,2)))
        elseif GEpriceindexes(pp,1)==maxidx
            cprintf('err',heteroagentoptions.verboseaccuracy1,GEPriceParamNames{pp},GEpricesvec(GEpriceindexes(pp,1)))
        else
            fprintf(heteroagentoptions.verboseaccuracy1,GEPriceParamNames{pp},GEpricesvec(GEpriceindexes(pp,1)))
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
    fprintf('Current aggregate variables, conditional on ptype: \n')
    for aa=1:length(AggVarNames)
        fprintf(heteroagentoptions.verboseaccuracy1ptype,AggVarNames{aa},AggVars_ConditionalOnPType(aa,:))
    end
    if heteroagentoptions.useintermediateEqns==1
        % ggindex turns the equation number into its slice of intermediateEqnsVec: one entry for a
        % standard intermediate eqn, N_i entries for one that holds conditional on ptype
        fprintf('Current intermediateEqns: \n')
        ggindex=ones(length(intEqnnames),1)+heteroagentoptions.intermediateEqnsptype'*(PTypeStructure.N_i-1);
        ggindex=[[1; cumsum(ggindex(1:end-1))+1],cumsum(ggindex)];
        for gg=1:length(intEqnnames)
            if heteroagentoptions.intermediateEqnsptype(gg)==1
                fprintf(heteroagentoptions.verboseaccuracy1ptype,intEqnnames{gg},intermediateEqnsVec(ggindex(gg,1):ggindex(gg,2)))
            else
                fprintf(heteroagentoptions.verboseaccuracy1,intEqnnames{gg},intermediateEqnsVec(ggindex(gg,1):ggindex(gg,2)))
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
    ggindex=ones(length(GEeqnNames),1)+heteroagentoptions.GEptype'*(PTypeStructure.N_i-1);
    ggindex=[[1; cumsum(ggindex(1:end-1))+1],cumsum(ggindex)];
    for gg=1:length(GEeqnNames)
        if heteroagentoptions.GEptype(gg)==1
            fprintf(heteroagentoptions.verboseaccuracy2ptype,GEeqnNames{gg},GeneralEqmConditionsVec(ggindex(gg,1):ggindex(gg,2)))
        elseif ggindex(gg,1)==maxidx
            cprintf('err', heteroagentoptions.verboseaccuracy2,GEeqnNames{gg},GeneralEqmConditionsVec(ggindex(gg,1)))
        else
            fprintf(heteroagentoptions.verboseaccuracy2,GEeqnNames{gg},GeneralEqmConditionsVec(ggindex(gg,1)))
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
