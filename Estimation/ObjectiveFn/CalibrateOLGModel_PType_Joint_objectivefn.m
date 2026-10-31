function [Obj,GEout]=CalibrateOLGModel_PType_Joint_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, GEPriceParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, GeneralEqmEqnsCell, GEeqnNames, GeneralEqmEqnParamNames, usingallstats,usinglcp,usingcustomstats, targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes, AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, usingautocorr, autocorrmomentnames, autocorrcummomentsizes, FnsToEvaluate_AutoCorr, autocorrtimehorizons, usingcrosssec, crosssecmomentnames, crossseccummomentsizes, FnsToEvaluate_CrossSec, usingagecrosssec, agecrosssecmomentnames, agecrossseccummomentsizes, FnsToEvaluate_AgeCrossSec, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, heteroagentoptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats,
% LifeCycleProfiles, AutoCorrTransProbs and the two CrossSectionCovarCorr
% commands (all the PType versions). Lastly there is caliboptions.
%
% Joint: the GE prices are the last entries of CalibParamNames (calibrated alongside the other parameters), and the GE conditions
% are part of the objective.
% Second output GEout: .prices (the GE prices, in the order of GEPriceParamNames) and .conditions (the GE conditions, in the order of
% GEeqnNames), both column vectors.
% vectoroutput=1 outputs the model moments at the targeted entries followed by the GE conditions.

% Untransform the parameters (when dealing with constraints the inputs are the transformed parameters, so want to switch them back to original model parameters)
[calibparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(calibparamsvec,calibparamsvecindex,CalibParamNames,caliboptions);
% Note: ptype makes no difference to this.

if caliboptions.verbose==1
    fprintf(' \n')
    fprintf('Current parameter values: \n')
    for pp=1:nCalibParams
        if nCalibParamsFinder(pp,2)==0 % parameter does not depend on ptype
            if calibparamsvecindex(pp+1)-calibparamsvecindex(pp)==1
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'= %8.6f \n'],calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))
            else
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'=  \n'])
                calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))' % want the output as a row
            end
        else  % parameter depends on ptype
            if calibparamsvecindex(pp+1)-calibparamsvecindex(pp)==1
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'.',Names_i{nCalibParamsFinder(pp,2)},'= %8.6f \n'],calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))
            else
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'.',Names_i{nCalibParamsFinder(pp,2)},'=  \n'])
                calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))' % want the output as a row
            end
        end
    end
end

for pp=1:nCalibParams
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix(:,sum(calibomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        if nCalibParamsFinder(pp,2)==0 % parameter does not depend on ptype
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)})=reshape(currparamraw,calibparamssizes(pp,:));
        else
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)})=reshape(currparamraw,calibparamssizes(pp,:));
        end
    else
        if nCalibParamsFinder(pp,2)==0 % parameter does not depend on ptype
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)})=reshape(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)),calibparamssizes(pp,:));
        else
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)})=reshape(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)),calibparamssizes(pp,:));
        end
    end
end


%% ParametrizeParamsFn can be used to parametrize the parameters (including the distribution of permanent types)
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Do grids if those depend on parameters being calibrated (otherwise they are already done)
if caliboptions.calibrateshocks==1
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    % The user's own grids are needed if either CustomModelStats is given them
    KeepOriginalGrid=((usingcustomstats==1 && caliboptions.CustomModelStats_usergrids==1) || (heteroagentoptions.useCustomModelStats==1 && heteroagentoptions.CustomModelStats_usergrids==1));
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz_PType(n_z,z_gridvals_J,pi_z_J,N_j,Names_i,Parameters,vfoptions,3,KeepOriginalGrid);
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
end

% Same for semi-exogenous shocks
if caliboptions.calibsemiexo==1
    vfoptions=SemiExogShockSetup_FHorz_PType(n_d,N_j,Names_i,d_grid,Parameters,vfoptions,3);
    simoptions.semiz_gridvals_J=vfoptions.semiz_gridvals_J;
    simoptions.pi_semiz_J=vfoptions.pi_semiz_J;
end


%% Solve the model
% (V is kept: either CustomModelStats, from caliboptions or from heteroagentoptions, takes it)
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,pi_z_J,Parameters,simoptions);

% Note: I could make it a bit faster by just doing these AggVars for the general eqm as part of the AllStats
AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, simoptions);

AggVarNames=fieldnames(AggVars);
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars.(AggVarNames{aa}).Mean;
end

%% Custom Model Stats
if usingcustomstats==1
    if caliboptions.CustomModelStats_usergrids==0
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,pi_z_J,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    elseif caliboptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            caliboptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            caliboptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,caliboptions.CustomModelStatsInputs.z_grid,caliboptions.CustomModelStatsInputs.pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    end
end
% Custom Model Stats can be in heteroagent options or caliboptions so when doing joint-optimization we need to handle both possibilities
% (heteroagentoptions.CustomModelStats takes the same inputs HeteroAgentStationaryEqm_Case1_FHorz_PType hands it)
if heteroagentoptions.useCustomModelStats==1
    if heteroagentoptions.CustomModelStats_usergrids==0
        CustomStats2=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,heteroagentoptions.CustomModelStatsInputs.N_j,Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,z_gridvals_J,pi_z_J,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    elseif heteroagentoptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            heteroagentoptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            heteroagentoptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats2=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,heteroagentoptions.CustomModelStatsInputs.N_j,Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,heteroagentoptions.CustomModelStatsInputs.z_grid,heteroagentoptions.CustomModelStatsInputs.pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    end
    % Note: anything else you want, just 'hide' it in heteroagentoptions
    customstatnames=fieldnames(CustomStats2);
    for pp=1:length(customstatnames)
        Parameters.(customstatnames{pp})=CustomStats2.(customstatnames{pp});
    end
end


%% intermediateEqns
if heteroagentoptions.useintermediateEqns==1
    for aa=1:heteroagentoptions.nIntermediateEqns
        Parameters.(heteroagentoptions.intermediateEqnNames{aa})=real(GeneralEqmConditions_Case1_v3(heteroagentoptions.intermediateEqnsCell{aa},heteroagentoptions.intermediateEqnParamNames(aa).Names,Parameters));
    end
end

%% Evaluate General Eqm Eqns
GeneralEqmConditionsVec=zeros(1,length(GEeqnNames));
for gg=1:length(GEeqnNames)
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    GeneralEqmConditionsVec(gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
end
GEout.prices=zeros(length(GEPriceParamNames),1);
for pp=1:length(GEPriceParamNames)
    GEout.prices(pp)=gather(Parameters.(GEPriceParamNames{pp}));
end
GEout.conditions=gather(GeneralEqmConditionsVec');


%% Calculate model stats
if usingallstats==1
    % caliboptions.whichcombos=1 (CalibrateOLGModel_PType's default): the stats commands compute only the targeted (function, statistic,
    % restriction[, age], ptype or grouped) combinations, through simoptions.whichcombos and a per-combination simoptions.whichstats with a
    % trailing type dimension, built by SetupTargetMoments_FHorz (caliboptions.selectors). =0: every statistic of every targeted function
    % (whichstats all ones). The commands take differently shaped selectors, so each gets its own copy of simoptions.
    simoptions_AllStats=simoptions;
    if caliboptions.whichcombos==1
        simoptions_AllStats.whichcombos=caliboptions.selectors.AllStats.whichcombos;
        simoptions_AllStats.whichstats=caliboptions.selectors.AllStats.whichstats;
    else % caliboptions.whichcombos=0: every statistic of every targeted function
        simoptions_AllStats.whichstats=ones(1,7);
    end
    AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate_AllStats,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_AllStats);
end
if usinglcp==1
    simoptions_ACStats=simoptions;
    if caliboptions.whichcombos==1
        simoptions_ACStats.whichcombos=caliboptions.selectors.ACStats.whichcombos;
        simoptions_ACStats.whichstats=caliboptions.selectors.ACStats.whichstats;
    else % caliboptions.whichcombos=0: every statistic of every targeted function
        simoptions_ACStats.whichstats=ones(1,7);
    end
    usingacswindowlog=(isfield(caliboptions,'selectors') && isfield(caliboptions.selectors,'ACStats') && isfield(caliboptions.selectors.ACStats,'haswindowlog') && caliboptions.selectors.ACStats.haswindowlog==1);
    if usingacswindowlog % the window and log rows the targets name: simoptions.ntimeperiods, ntimeperiodstype and logwhichcombos (set whatever caliboptions.whichcombos, as they decide which outputs there are)
        simoptions_ACStats.ntimeperiods=caliboptions.selectors.ACStats.ntimeperiods;
        simoptions_ACStats.ntimeperiodstype=caliboptions.selectors.ACStats.ntimeperiodstype;
        simoptions_ACStats.logwhichcombos=caliboptions.selectors.ACStats.logwhichcombos;
        if ~isempty(caliboptions.selectors.ACStats.agegroupings)
            simoptions_ACStats.agegroupings=caliboptions.selectors.ACStats.agegroupings; % only window targets: by start age
        end
        if any(simoptions_ACStats.ntimeperiods>1)
            simoptions_ACStats.pi_z=pi_z_J; % the windows need the z transitions (a structure by ptype, or with a trailing type dimension, as pi_z)
        end
    end
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate_ACStats,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_ACStats);
    if usingacswindowlog && caliboptions.selectors.ACStats.split==1
        % the single-period targets are by age groups, so the window targets (by start age) are a second call, merged into the output
        simoptions_ACStatsW=simoptions;
        simoptions_ACStatsW.agegroupings=caliboptions.selectors.ACStatsW.agegroupings;
        if caliboptions.whichcombos==1
            simoptions_ACStatsW.whichcombos=caliboptions.selectors.ACStatsW.whichcombos;
            simoptions_ACStatsW.whichstats=caliboptions.selectors.ACStatsW.whichstats;
        else
            simoptions_ACStatsW.whichstats=ones(1,7);
        end
        simoptions_ACStatsW.ntimeperiods=caliboptions.selectors.ACStatsW.ntimeperiods;
        simoptions_ACStatsW.ntimeperiodstype=caliboptions.selectors.ACStatsW.ntimeperiodstype;
        simoptions_ACStatsW.logwhichcombos=caliboptions.selectors.ACStatsW.logwhichcombos;
        simoptions_ACStatsW.pi_z=pi_z_J;
        AgeConditionalStatsW=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate_ACStats,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_ACStatsW);
        if isfield(simoptions,'conditionalrestrictions')
            restrnamesW=fieldnames(simoptions.conditionalrestrictions);
        else
            restrnamesW={};
        end
        fW=fieldnames(AgeConditionalStatsW);
        for kk=1:length(fW)
            if any(strcmp(restrnamesW,fW{kk}))
                fR=fieldnames(AgeConditionalStatsW.(fW{kk}));
                for rr=1:length(fR)
                    if ~strcmp(fR{rr},'RestrictedSampleMass')
                        AgeConditionalStats.(fW{kk}).(fR{rr})=AgeConditionalStatsW.(fW{kk}).(fR{rr});
                    end
                end
            else
                AgeConditionalStats.(fW{kk})=AgeConditionalStatsW.(fW{kk});
            end
        end
    end
end
if usingautocorr==1
    % The three commands below take whichcombos only (no per-combination whichstats), with the trailing type dimension (a type's slot, or
    % the grouped slot which makes the command compute every type at that combination); they report what they compute.
    simoptions_AutoCorr=simoptions;
    if isfield(simoptions_AutoCorr,'agegroupings')
        simoptions_AutoCorr=rmfield(simoptions_AutoCorr,'agegroupings'); % the AutoCorr targets are per age (1 x N_j-K) whatever the age groups of the other targets; the command has no age bins
    end
    if isfield(simoptions,'timehorizons')
        simoptions_AutoCorr.timehorizons=union(simoptions.timehorizons,autocorrtimehorizons); % the horizons the targets name, plus any the user asked for
    else
        simoptions_AutoCorr.timehorizons=autocorrtimehorizons;
    end
    if caliboptions.whichcombos==1
        simoptions_AutoCorr.whichcombos=caliboptions.selectors.AutoCorr.whichcombos; % [nFns,N_j,1+nRestr,N_i+1], by start age
    end
    if isfield(caliboptions,'selectors') % the window and log rows the targets name: simoptions.ntimeperiods, ntimeperiodstype, logwhichcombos (set whatever caliboptions.whichcombos, as they decide which outputs there are)
        simoptions_AutoCorr.ntimeperiods=caliboptions.selectors.AutoCorr.ntimeperiods;
        simoptions_AutoCorr.ntimeperiodstype=caliboptions.selectors.AutoCorr.ntimeperiodstype;
        simoptions_AutoCorr.logwhichcombos=caliboptions.selectors.AutoCorr.logwhichcombos;
    end
    CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_AutoCorr,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,pi_z_J,simoptions_AutoCorr);
end
if usingcrosssec==1
    simoptions_CrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_CrossSec.whichcombos=caliboptions.selectors.CrossSec.whichcombos; % pair-shaped, [nFns,nFns,N_i+1], or [nFns,nFns,1+nRestr,N_i+1] with conditional restrictions (restricted targets on pages 2:end); in blocks of ntimeperiods with window targets
    end
    if isfield(caliboptions,'selectors') && isfield(caliboptions.selectors,'CrossSec') && isfield(caliboptions.selectors.CrossSec,'ntimeperiods') && any(caliboptions.selectors.CrossSec.ntimeperiods>1)
        % the windows the targets name (set whatever caliboptions.whichcombos, as they decide which outputs there are)
        simoptions_CrossSec.ntimeperiods=caliboptions.selectors.CrossSec.ntimeperiods;
        simoptions_CrossSec.ntimeperiodstype=caliboptions.selectors.CrossSec.ntimeperiodstype;
        simoptions_CrossSec.pi_z=pi_z_J; % the windows need the z transitions (a structure by ptype, or with a trailing type dimension, as pi_z)
    end
    CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_CrossSec,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_CrossSec);
end
if usingagecrosssec==1
    simoptions_AgeCrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_AgeCrossSec.whichcombos=caliboptions.selectors.AgeCrossSec.whichcombos; % pair-shaped per age group, [nFns,nFns,ngroups,N_i+1], or [nFns,nFns,ngroups,1+nRestr,N_i+1] with conditional restrictions; in blocks of ntimeperiods with window targets
    end
    usingagecrosssecW=(isfield(caliboptions,'selectors') && isfield(caliboptions.selectors,'AgeCrossSec') && isfield(caliboptions.selectors.AgeCrossSec,'ntimeperiods'));
    if usingagecrosssecW && any(caliboptions.selectors.AgeCrossSec.ntimeperiods>1)
        % the windows the targets name (set whatever caliboptions.whichcombos, as they decide which outputs there are)
        simoptions_AgeCrossSec.ntimeperiods=caliboptions.selectors.AgeCrossSec.ntimeperiods;
        simoptions_AgeCrossSec.ntimeperiodstype=caliboptions.selectors.AgeCrossSec.ntimeperiodstype;
        simoptions_AgeCrossSec.pi_z=pi_z_J; % the windows need the z transitions
        if ~isempty(caliboptions.selectors.AgeCrossSec.agegroupings)
            simoptions_AgeCrossSec.agegroupings=caliboptions.selectors.AgeCrossSec.agegroupings; % only window targets: by start age
        end
    end
    AgeConditionalCrossSectionCorr=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_AgeCrossSec,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_AgeCrossSec);
    if usingagecrosssecW && caliboptions.selectors.AgeCrossSec.split==1
        % the single-period targets are by age groups, so the window targets (by start age) are a second call, merged into the output
        simoptions_AgeCrossSecW=simoptions;
        if caliboptions.whichcombos==1
            simoptions_AgeCrossSecW.whichcombos=caliboptions.selectors.AgeCrossSecW.whichcombos;
        end
        simoptions_AgeCrossSecW.ntimeperiods=caliboptions.selectors.AgeCrossSecW.ntimeperiods;
        simoptions_AgeCrossSecW.ntimeperiodstype=caliboptions.selectors.AgeCrossSecW.ntimeperiodstype;
        simoptions_AgeCrossSecW.agegroupings=caliboptions.selectors.AgeCrossSecW.agegroupings;
        simoptions_AgeCrossSecW.pi_z=pi_z_J;
        AgeConditionalCrossSectionCorrW=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_AgeCrossSec,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_AgeCrossSecW);
        if isfield(simoptions,'conditionalrestrictions')
            restrnamesW=fieldnames(simoptions.conditionalrestrictions);
        else
            restrnamesW={};
        end
        fW=fieldnames(AgeConditionalCrossSectionCorrW);
        for kk=1:length(fW)
            if any(strcmp(restrnamesW,fW{kk}))
                fR=fieldnames(AgeConditionalCrossSectionCorrW.(fW{kk}));
                for rr=1:length(fR)
                    if ~isempty(regexp(fR{rr},'_\d+period(Average|Sum)(_ptype)?$','once')) % the window fields (the restriction's single-period fields are those of the first call)
                        AgeConditionalCrossSectionCorr.(fW{kk}).(fR{rr})=AgeConditionalCrossSectionCorrW.(fW{kk}).(fR{rr});
                    end
                end
            elseif ~isempty(regexp(fW{kk},'_\d+period(Average|Sum)(_ptype)?$','once'))
                AgeConditionalCrossSectionCorr.(fW{kk})=AgeConditionalCrossSectionCorrW.(fW{kk});
            end
        end
    end
end


%% Get current values of the target moments as a vector
% Each kind: walk the names of the target (two to five levels, the empty cells skipped) into the command's output, which has the same
% nesting (restrictions, permanent types, MoreInequality, CovarianceWith/CorrelationWith and the matrices alike). A target that is NaN
% in full (SetupTargetMoments_FHorz warned about it and selected nothing for it, so the command may not have created it) is skipped
% and filled with NaN (it is dropped from the objective anyway).
currentmomentvec=zeros(size(targetmomentvec));
sofar=0;
for kindc=1:5
    if kindc==1
        kindusing=usingallstats;
    elseif kindc==2
        kindusing=usinglcp;
    elseif kindc==3
        kindusing=usingautocorr;
    elseif kindc==4
        kindusing=usingcrosssec;
    else
        kindusing=usingagecrosssec;
    end
    if kindusing==1
        if kindc==1
            kindnames=allstatmomentnames; kindcum=allstatcummomentsizes; kindout=AllStats;
        elseif kindc==2
            kindnames=acsmomentnames; kindcum=acscummomentsizes; kindout=AgeConditionalStats;
        elseif kindc==3
            kindnames=autocorrmomentnames; kindcum=autocorrcummomentsizes; kindout=CorrTransProbs;
        elseif kindc==4
            kindnames=crosssecmomentnames; kindcum=crossseccummomentsizes; kindout=CrossSectionCorr;
        else
            kindnames=agecrosssecmomentnames; kindcum=agecrossseccummomentsizes; kindout=AgeConditionalCrossSectionCorr;
        end
        for cc=1:size(kindnames,1)
            if cc==1
                idx=sofar+1:sofar+kindcum(1);
            else
                idx=sofar+kindcum(cc-1)+1:sofar+kindcum(cc);
            end
            if all(isnan(targetmomentvec(idx)))
                currentmomentvec(idx)=NaN;
                continue
            end
            temp=kindout;
            for kk=1:size(kindnames,2)
                if ~isempty(kindnames{cc,kk})
                    temp=temp.(kindnames{cc,kk});
                end
            end
            currentmomentvec(idx)=temp(:);
        end
        sofar=sofar+kindcum(end);
    end
end
if usingcustomstats==1
    currentmomentvec(sofar+1:sofar+cmscummomentsizes(1))=CustomStats.(cmsmomentnames{1,1});
    for cc=2:size(cmsmomentnames,1)
        currentmomentvec(sofar+cmscummomentsizes(cc-1)+1:sofar+cmscummomentsizes(cc))=CustomStats.(cmsmomentnames{cc,1});
    end
end


%% Option to log moments (if targets are log, then this will have been already applied)
if any(caliboptions.logmoments>0) % need to log some moments
    currentmomentvec=(1-caliboptions.logmoments).*currentmomentvec + caliboptions.logmoments.*log(currentmomentvec.*caliboptions.logmoments+(1-caliboptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Evaluate the objective function (which is being minimized)
% Create Obj1 for calib targets, then Obj2 for general eqm conditions.
% These are then combined for Obj
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if caliboptions.vectoroutput==1
    % Output the vector of currentmomentvec
    % Main use it for computing derivatives of moments with respect to parameters
    Obj1=currentmomentvec(actualtarget);
    Obj2=GeneralEqmConditionsVec;
    Obj=[Obj1; Obj2'];
elseif caliboptions.vectoroutput==0 % scalar output
    % currentmomentvec is the current moment values
    % targetmomentvec is the target moment values
    % Both are column vectors

    % Note: MethodOfMoments and sum_squared are doing the same calculation, I
    % just write them in ways that make it more obvious that they do what they say.
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Obj=(targetmomentvec-currentmomentvec)'*caliboptions.weights*(targetmomentvec-currentmomentvec);
        % For the purpose of doing log(moments) I switched to the following
        % (otherwise getting silly current moments can seem attractive)
        Obj1=(currentmomentvec(actualtarget)-targetmomentvec(actualtarget))'*caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj1=sum(caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget)).^2,'omitnan');
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj1=sum(caliboptions.weights.*(log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget)).^2),'omitnan');
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj1=Obj1/caliboptions.nCalibParamsNonGE; % This is done so that the tolerances for convergence are sensible (the number of calibrated parameter names, not counting the GE prices, so the same as in the nested objective) [was length(CalibParamNames), which in joint includes the GE prices]

    if penalty>0
        if Obj1>0
            Obj1=1.2*penalty*Obj1; % 20% penalty for being too far in violation of restrictions
        else % Obj is negative, so penalty is to reduce magnitude
            Obj1=0.8*(1/penalty)*Obj1; % 20% penalty for being too far in violation of restrictions
        end
    end

    % General eqm conditions are always evaluated using sum-of-squares
    Obj2=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));

    Obj=Obj1+caliboptions.relativeGEweight*Obj2;

elseif caliboptions.vectoroutput==2
    % Weighted vector (for use with least-squares residuals algorithms)
    % Note: the outer-layers of code already took 'square root' of the weights
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Is essentially the square-root of 'MethodOfMoments' [it is the form of input used by Matlab's lsqnonlin()]
        Obj1=caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj1=caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj1=caliboptions.weights.*log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget));
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end

    % The square roots of the scalar form's weights and divisor, so that the sum of squares of this vector is exactly the scalar objective
    % (vectoroutput=0), penalty aside [was multiGEweights and relativeGEweight themselves, so the GE conditions entered the sum of squares
    % with their squares, and the targets part was not divided]
    Obj2=sqrt(heteroagentoptions.multiGEweights).*GeneralEqmConditionsVec;

    Obj=[Obj1/sqrt(caliboptions.nCalibParamsNonGE); sqrt(caliboptions.relativeGEweight)*Obj2'];

    Obj=gather(Obj); % lsqnonlin() doesn't work with gpu, so have to gather()
    if penalty>0
        Obj=[Obj; sqrt(penalty)]; % append penalty residual: contributes 'penalty' to lsqnonlin's sum(Obj.^2)
    else
        Obj=[Obj; 0]; % penalty=0 (params inside cutoffs); append 0 to keep residual length constant
    end
end


%% Verbose
if caliboptions.verbose==1
    if usingcustomstats==1
        fprintf('Current CustomModelStats variables (from caliboptions): \n')
        for ii=1:length(cmsmomentnames)
            fprintf('	%s: %8.4f \n',cmsmomentnames{ii},CustomStats.(cmsmomentnames{ii}))
        end
    end
    if heteroagentoptions.useCustomModelStats==1
        fprintf('Current CustomModelStats variables (from heteroagentoptions): \n')
        for ii=1:length(customstatnames)
            fprintf('	%s: %8.4f \n',customstatnames{ii},CustomStats2.(customstatnames{ii}))
        end
    end
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    if caliboptions.vectoroutput==0
        fprintf('Current objective fn value is %8.12f \n', Obj)
        fprintf('Current objective fn value breakdown: Obj1=%8.6f and Obj2=%8.6f (with caliboptions.relativeGEweight=%8.6f) \n', Obj1, Obj2, caliboptions.relativeGEweight)
        if penalty>0
            if Obj1>0
                fprintf('Current penalty is to multiply the targets part of the objective fn by %8.2f \n', 1.2*penalty)
            else  % Obj1 is negative, so penalty is to reduce magnitude
                fprintf('Current penalty is to multiply the targets part of the objective fn by %8.2f \n', 0.8*(1/penalty) )
            end
        end
    elseif caliboptions.vectoroutput==2
        % Obj1 and Obj2 are vectors of residuals here, so print their sums of squares (not the vectors themselves)
        fprintf('Current (sum-of-squares of) objective fn value is %8.12f \n', Obj'*Obj)
        fprintf('Current objective fn value breakdown: targets part=%8.6f and GE part=%8.6f (with caliboptions.relativeGEweight=%8.6f); their sum is the sum-of-squares above (less any penalty) \n', gather(sum(Obj1(:).^2))/caliboptions.nCalibParamsNonGE, gather(caliboptions.relativeGEweight*sum(Obj2(:).^2)), caliboptions.relativeGEweight)
        if penalty>0
            fprintf('Current penalty residual is %8.6f (it adds %8.6f to the sum-of-squares) \n', sqrt(penalty), penalty)
        end
    end

    fprintf(' \n')
    fprintf('Current GE prices (this is just repeat of part of current calibration params): \n')
    for ii=1:length(GEPriceParamNames)
        fprintf('	%s: %8.4f \n',GEPriceParamNames{ii},Parameters.(GEPriceParamNames{ii}))
    end
    fprintf('Current aggregate variables: \n')
    for ii=1:length(AggVarNames)
        fprintf('	%s: %8.4f \n',AggVarNames{ii},Parameters.(AggVarNames{ii})) % Note, this is done differently here because AggVars itself has been set as a matrix
    end
    if heteroagentoptions.useintermediateEqns==1
        fprintf('Current intermediateEqn variables: \n')
        for aa=1:length(heteroagentoptions.intermediateEqnNames)
            fprintf('	%s: %8.4f \n',heteroagentoptions.intermediateEqnNames{aa},Parameters.(heteroagentoptions.intermediateEqnNames{aa})) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    for ii=1:length(GEeqnNames)
        fprintf('	%s: %8.4f \n',GEeqnNames{ii},GeneralEqmConditionsVec(ii))
    end
end


end
