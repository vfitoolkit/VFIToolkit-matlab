function Obj=CalibrateLifeCycleModel_PType_withCohorts_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, targetmomentvec, cohortmoments, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions)
% As CalibrateLifeCycleModel_PType_objectivefn, but with cohorts entering the model at different ages
% (caliboptions.cohortagejshifter): jequaloneDist is a cell over cohorts, each a structure over ptypes; each cohort
% gets its own agent distribution (StationaryDist_Case1_FHorz_PType with simoptions.jequaloneDistAge at its entry age and
% simoptions.lastDistAge at the last age its targets need, or estimoptions.cohortlastage(cc) if given; and with the
% ptype weights scaled by the ptypes' masses in that cohort) and its own statistics, and the moments are stacked
% cohort by cohort as set up by SetupTargetMoments_FHorz_withCohorts (which gives cohortmoments).
% Age weights by cohort: simoptions.agemass_withCohort (ncohorts-by-N_j, or a structure of these by ptype) if given,
% otherwise every cohort has the AgeWeightParamNames values (the population age weights) from its entry age on (by
% ptype if that parameter is a structure).
% AutoCorr targets (TargetMoments.AutoCorr.cohortC..., see SetupTargetMoments_FHorz_withCohorts) are evaluated by
% EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType on the cohort's distribution, with simoptions.timehorizons set to the
% horizons the targets need (a restriction named in a target must exist in simoptions.conditionalrestrictions).
% Not implemented with cohorts: CustomModelStats, bootstrapped standard errors.
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.

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
    KeepOriginalGrid=(isfield(cohortmoments,'usingcustomstats') && any(cohortmoments.usingcustomstats) && caliboptions.CustomModelStats_usergrids==1); % the user's own grids, for CustomModelStats
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
vftimer=tic;
if isfield(cohortmoments,'usingcustomstats') && any(cohortmoments.usingcustomstats)
    [V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions); % CustomModelStats is given V
else
    [~, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);
end
vftime=toc(vftimer); % (for verbose)

%% Age weights and ptype weights by cohort
ncohorts=caliboptions.ncohorts;
N_i=length(Names_i);
ptweights_orig=Parameters.(PTypeDistParamNames{1}); % may come from ParametrizeParamsFn, so take it here each time
ptweights0=ptweights_orig(:);
if isempty(simoptions.agemass_withCohort)
    % Default: every cohort has the population's age weights from its entry age on (AgeWeightParamNames value at age j for age j, by ptype
    % if that parameter is a structure), so a cohort's mass falls with age as the population's does (mortality), and one cohort entering at
    % age 1 with all the mass is exactly the standard estimator. (Before 2026-10-07 the default was the entry age's weight at every age,
    % i.e. no mortality within a cohort.) Ages before entry carry no mass whatever their weight. The driver warns when this default is used.
    if isstruct(Parameters.(AgeWeightParamNames{1}))
        mewjc=struct();
        for ii=1:N_i
            mewjc.(Names_i{ii})=repmat(reshape(Parameters.(AgeWeightParamNames{1}).(Names_i{ii}),[1,N_j]),[ncohorts,1]);
        end
    else
        mewjc=repmat(reshape(Parameters.(AgeWeightParamNames{1}),[1,N_j]),[ncohorts,1]);
    end
else
    mewjc=simoptions.agemass_withCohort;
end
simoptions.agemass_withCohort=mewjc; % so StationaryDist_FHorz_Case1 knows the age weights are cohort masses (and do not sum to one); PType_Options passes it through by ptype if a structure

%% Agent distribution and statistics, cohort by cohort, and the current values of the target moments as a vector
currentmomentvec=zeros(size(targetmomentvec));
cohorttimes=zeros(ncohorts,4); % per cohort: agent dist, AllStats, LifeCycleProfiles, AutoCorr (seconds, for verbose)
lastages=zeros(1,ncohorts);
for cc=1:ncohorts
    simoptions.jequaloneDistAge=caliboptions.cohortagejshifter(cc);
    if isfield(caliboptions,'cohortlastage')
        simoptions.lastDistAge=caliboptions.cohortlastage(cc); % user override of the last age the cohort's distribution is computed to
    else
        simoptions.lastDistAge=cohortmoments.lastage(cc); % the last age this cohort's targets need (SetupTargetMoments_FHorz_withCohorts): no need to iterate the distribution beyond it
    end
    lastages(cc)=simoptions.lastDistAge;
    cohorttimer=tic;
    if isstruct(mewjc)
        for ii=1:N_i
            Parameters.(AgeWeightParamNames{1}).(Names_i{ii})=mewjc.(Names_i{ii})(cc,:);
        end
    else
        Parameters.(AgeWeightParamNames{1})=mewjc(cc,:);
    end
    % ptype weights for this cohort: the ptype's overall weight times its mass in this cohort (a ptype with no mass in this cohort gets weight zero)
    ptw=ptweights0.*caliboptions.cohortmasses(:,cc);
    Parameters.(PTypeDistParamNames{1})=ptw/sum(ptw);
    StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist{cc},AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,pi_z_J,Parameters,simoptions);
    cohorttimes(cc,1)=toc(cohorttimer); cohorttimer=tic;
    sofar=cohortmoments.offset(cc);
    % caliboptions.whichcombos=1 (the default): the stats commands compute only the targeted (function, statistic, restriction[, age], ptype or grouped)
    % combinations, through simoptions.whichcombos and a per-combination simoptions.whichstats with a trailing type dimension built for this
    % cohort by SetupTargetMoments_FHorz (cohortmoments.selectors{cc}, as caliboptions.selectors without cohorts), and the AutoCorr call only the
    % targeted start ages (by type). =0: every statistic any target of the cohort names, of every targeted function, at every age, on every
    % restriction page, for every type and the grouped stats (the 7-vector cohortmoments.*_whichstats{cc}). [Until 2026-10-23 the cohort route
    % never set whichcombos, so it always did the latter: with window targets, every window at every start age on every page.]
    if cohortmoments.usingallstats(cc)==1
        simoptions.whichstats=cohortmoments.AllStats_whichstats{cc};
        simoptions_as=simoptions;
        if caliboptions.whichcombos==1
            simoptions_as.whichcombos=cohortmoments.selectors{cc}.AllStats.whichcombos;
            simoptions_as.whichstats=cohortmoments.selectors{cc}.AllStats.whichstats;
        end
        AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy,cohortmoments.FnsToEvaluate_AllStats{cc},Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_as);
        cohorttimes(cc,2)=toc(cohorttimer); cohorttimer=tic;
        allstatmomentnames=cohortmoments.allstatmomentnames{cc};
        allstatcummomentsizes=cohortmoments.allstatcummomentsizes{cc};
        for mm=1:size(allstatmomentnames,1)
            if mm==1
                ind1=sofar+1;
            else
                ind1=sofar+allstatcummomentsizes(mm-1)+1;
            end
            ind2=sofar+allstatcummomentsizes(mm);
            if isempty(allstatmomentnames{mm,3})
                currentmomentvec(ind1:ind2)=AllStats.(allstatmomentnames{mm,1}).(allstatmomentnames{mm,2});
            else
                if isempty(allstatmomentnames{mm,4})
                    currentmomentvec(ind1:ind2)=AllStats.(allstatmomentnames{mm,1}).(allstatmomentnames{mm,2}).(allstatmomentnames{mm,3});
                else
                    currentmomentvec(ind1:ind2)=AllStats.(allstatmomentnames{mm,1}).(allstatmomentnames{mm,2}).(allstatmomentnames{mm,3}).(allstatmomentnames{mm,4});
                end
            end
        end
        sofar=sofar+allstatcummomentsizes(end);
    end
    if cohortmoments.usinglcp(cc)==1
        simoptions.whichstats=cohortmoments.ACStats_whichstats{cc};
        simoptions_lcp=simoptions;
        if caliboptions.whichcombos==1 % only the targeted (row, age, restriction, type or grouped) combinations and their statistics (the rows in blocks of the window lengths, as ntimeperiods below)
            simoptions_lcp.whichcombos=cohortmoments.selectors{cc}.ACStats.whichcombos;
            simoptions_lcp.whichstats=cohortmoments.selectors{cc}.ACStats.whichstats;
        end
        if isfield(cohortmoments,'acsntimeperiods') && (any(cohortmoments.acsntimeperiods{cc}>1) || ~isempty(cohortmoments.acslog{cc}))
            % the window and log rows the targets name: simoptions.ntimeperiods, ntimeperiodstype and logwhichcombos (rows in blocks of the
            % window lengths, the FnsToEvaluate in the order of cohortmoments.FnsToEvaluate_ACStats; pages the unrestricted and then the
            % restrictions in the order of simoptions.conditionalrestrictions; no type dimension)
            simoptions_lcp.ntimeperiods=cohortmoments.acsntimeperiods{cc};
            simoptions_lcp.ntimeperiodstype=cohortmoments.acsntimeperiodstype{cc};
            acslog=cohortmoments.acslog{cc};
            fnsACS=fieldnames(cohortmoments.FnsToEvaluate_ACStats{cc});
            if isfield(simoptions,'conditionalrestrictions')
                rnamesACS=fieldnames(simoptions.conditionalrestrictions);
            else
                rnamesACS={};
            end
            logwc=zeros(length(simoptions_lcp.ntimeperiods)*length(fnsACS),N_j,1+length(rnamesACS));
            for mm=1:size(acslog,1)
                rowACS=(find(simoptions_lcp.ntimeperiods==acslog{mm,3})-1)*length(fnsACS)+find(strcmp(fnsACS,acslog{mm,2}));
                if isempty(acslog{mm,1})
                    pageACS=1;
                else
                    pageACS=1+find(strcmp(rnamesACS,acslog{mm,1}));
                end
                logwc(rowACS,:,pageACS)=max(logwc(rowACS,:,pageACS),acslog{mm,4});
            end
            simoptions_lcp.logwhichcombos=logwc;
            if any(simoptions_lcp.ntimeperiods>1)
                simoptions_lcp.pi_z=pi_z_J; % the windows need the z transitions
            end
        end
        AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,cohortmoments.FnsToEvaluate_ACStats{cc},Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_lcp);
        cohorttimes(cc,3)=toc(cohorttimer); cohorttimer=tic;
        acsmomentnames=cohortmoments.acsmomentnames{cc};
        acscummomentsizes=cohortmoments.acscummomentsizes{cc};
        for mm=1:size(acsmomentnames,1)
            if mm==1
                ind1=sofar+1;
            else
                ind1=sofar+acscummomentsizes(mm-1)+1;
            end
            ind2=sofar+acscummomentsizes(mm);
            if isempty(acsmomentnames{mm,3})
                currentmomentvec(ind1:ind2)=AgeConditionalStats.(acsmomentnames{mm,1}).(acsmomentnames{mm,2});
            else
                if isempty(acsmomentnames{mm,4})
                    currentmomentvec(ind1:ind2)=AgeConditionalStats.(acsmomentnames{mm,1}).(acsmomentnames{mm,2}).(acsmomentnames{mm,3});
                else
                    currentmomentvec(ind1:ind2)=AgeConditionalStats.(acsmomentnames{mm,1}).(acsmomentnames{mm,2}).(acsmomentnames{mm,3}).(acsmomentnames{mm,4});
                end
            end
        end
        sofar=sofar+acscummomentsizes(end);
    end
    if cohortmoments.usingautocorr(cc)==1
        % AutoCorr targets: EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType on this cohort's distribution, at the horizons the targets need
        acrmomentnames=cohortmoments.acrmomentnames{cc};
        acrcummomentsizes=cohortmoments.acrcummomentsizes{cc};
        for mm=1:size(acrmomentnames,1)
            if ~isempty(acrmomentnames{mm,1})
                if ~isfield(simoptions,'conditionalrestrictions') || ~isfield(simoptions.conditionalrestrictions,acrmomentnames{mm,1})
                    error(['TargetMoments.AutoCorr uses the restriction ',acrmomentnames{mm,1},' but there is no simoptions.conditionalrestrictions.',acrmomentnames{mm,1}])
                end
            end
        end
        simoptions_acr=simoptions;
        simoptions_acr.timehorizons=cohortmoments.acrhorizons{cc};
        % the window and log rows the targets name: simoptions.ntimeperiods, ntimeperiodstype and logwhichcombos (rows in blocks of
        % the window lengths, the FnsToEvaluate in the order of cohortmoments.FnsToEvaluate_AutoCorr; pages the unrestricted and then
        % the restrictions in the order of simoptions.conditionalrestrictions)
        simoptions_acr.ntimeperiods=cohortmoments.acrntimeperiods{cc};
        simoptions_acr.ntimeperiodstype=cohortmoments.acrntimeperiodstype{cc};
        acrlog=cohortmoments.acrlog{cc};
        if ~isempty(acrlog)
            fnsAC=fieldnames(cohortmoments.FnsToEvaluate_AutoCorr{cc});
            if isfield(simoptions,'conditionalrestrictions')
                rnamesAC=fieldnames(simoptions.conditionalrestrictions);
            else
                rnamesAC={};
            end
            logwc=zeros(length(simoptions_acr.ntimeperiods)*length(fnsAC),N_j,1+length(rnamesAC));
            for mm=1:size(acrlog,1)
                rowAC=(find(simoptions_acr.ntimeperiods==acrlog{mm,3})-1)*length(fnsAC)+find(strcmp(fnsAC,acrlog{mm,2}));
                if isempty(acrlog{mm,1})
                    pageAC=1;
                else
                    pageAC=1+find(strcmp(rnamesAC,acrlog{mm,1}));
                end
                logwc(rowAC,:,pageAC)=max(logwc(rowAC,:,pageAC),acrlog{mm,4});
            end
            simoptions_acr.logwhichcombos=logwc;
        end
        if caliboptions.whichcombos==1
            % only the targeted (row, start age, restriction, type or grouped) combinations: simoptions.whichcombos (the layout of logwhichcombos above,
            % with a trailing type dimension of N_i+1: the ptypes in the order of Names_i, then the grouped stats) from the start ages every target
            % selects, at the level and at the log (cohortmoments.acrsel; the command reports the byproducts it computes anyway)
            acrsel=cohortmoments.acrsel{cc};
            fnsAC=fieldnames(cohortmoments.FnsToEvaluate_AutoCorr{cc});
            if isfield(simoptions,'conditionalrestrictions')
                rnamesAC=fieldnames(simoptions.conditionalrestrictions);
            else
                rnamesAC={};
            end
            wcAC=zeros(length(simoptions_acr.ntimeperiods)*length(fnsAC),N_j,1+length(rnamesAC),N_i+1);
            for mm=1:size(acrsel,1)
                rowAC=(find(simoptions_acr.ntimeperiods==acrsel{mm,3})-1)*length(fnsAC)+find(strcmp(fnsAC,acrsel{mm,2}));
                if isempty(acrsel{mm,1})
                    pageAC=1;
                else
                    pageAC=1+find(strcmp(rnamesAC,acrsel{mm,1}));
                end
                if isempty(acrsel{mm,5})
                    slotAC=N_i+1; % a grouped target
                else
                    slotAC=find(strcmp(Names_i,acrsel{mm,5}));
                end
                wcAC(rowAC,:,pageAC,slotAC)=max(wcAC(rowAC,:,pageAC,slotAC),acrsel{mm,4});
            end
            simoptions_acr.whichcombos=wcAC;
        end
        CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType(StationaryDist,Policy,cohortmoments.FnsToEvaluate_AutoCorr{cc},Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,pi_z_J,simoptions_acr);
        cohorttimes(cc,4)=toc(cohorttimer); cohorttimer=tic;
        for mm=1:size(acrmomentnames,1)
            if mm==1
                ind1=sofar+1;
            else
                ind1=sofar+acrcummomentsizes(mm-1)+1;
            end
            ind2=sofar+acrcummomentsizes(mm);
            if isempty(acrmomentnames{mm,1})
                if isempty(acrmomentnames{mm,3})
                    currentmomentvec(ind1:ind2)=CorrTransProbs.(acrmomentnames{mm,2}).(acrmomentnames{mm,4});
                else
                    currentmomentvec(ind1:ind2)=CorrTransProbs.(acrmomentnames{mm,2}).(acrmomentnames{mm,3}).(acrmomentnames{mm,4});
                end
            else
                if isempty(acrmomentnames{mm,3})
                    currentmomentvec(ind1:ind2)=CorrTransProbs.(acrmomentnames{mm,1}).(acrmomentnames{mm,2}).(acrmomentnames{mm,4});
                else
                    currentmomentvec(ind1:ind2)=CorrTransProbs.(acrmomentnames{mm,1}).(acrmomentnames{mm,2}).(acrmomentnames{mm,3}).(acrmomentnames{mm,4});
                end
            end
        end
        sofar=sofar+acrcummomentsizes(end);
    end
    if isfield(cohortmoments,'usingcustomstats') && cohortmoments.usingcustomstats(cc)==1
        % CustomModelStats targets: the user's function (as without cohorts) on this cohort's agent distribution; it is given the
        % user's own vfoptions and simoptions, with simoptions.jequaloneDistAge (the cohort's entry age) and lastDistAge set
        simoptions_cms=caliboptions.CustomModelStatsInputs.simoptions;
        simoptions_cms.jequaloneDistAge=caliboptions.cohortagejshifter(cc);
        simoptions_cms.lastDistAge=simoptions.lastDistAge;
        if caliboptions.CustomModelStats_usergrids==0
            CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,pi_z_J,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,simoptions_cms);
        else
            if caliboptions.calibrateshocks==1 % shock grids depend on parameters being estimated, so give the user's own grids as rebuilt from the current parameters
                caliboptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
                caliboptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
            end
            CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,caliboptions.CustomModelStatsInputs.z_grid,caliboptions.CustomModelStatsInputs.pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,simoptions_cms);
        end
        cmsmomentnames=cohortmoments.cmsmomentnames{cc};
        cmscummomentsizes=cohortmoments.cmscummomentsizes{cc};
        for mm=1:length(cmsmomentnames)
            if ~isfield(CustomStats,cmsmomentnames{mm})
                error(['TargetMoments.CustomModelStats.cohort',num2str(cc),'.',cmsmomentnames{mm},' is targeted, but estimoptions.CustomModelStats does not return a field of that name'])
            end
            if mm==1
                ind1=sofar+1;
            else
                ind1=sofar+cmscummomentsizes(mm-1)+1;
            end
            ind2=sofar+cmscummomentsizes(mm);
            val=gather(CustomStats.(cmsmomentnames{mm}));
            if numel(val)~=ind2-ind1+1
                error(['estimoptions.CustomModelStats returns ',cmsmomentnames{mm},' with ',num2str(numel(val)),' entries, but its target (cohort ',num2str(cc),') has ',num2str(ind2-ind1+1)])
            end
            currentmomentvec(ind1:ind2)=val(:);
        end
    end
end
Parameters.(PTypeDistParamNames{1})=ptweights_orig; % restore

%% Verbose: where the time went
if caliboptions.verbose==1
    fprintf('withCohorts objective fn: value fn %.1f seconds \n',vftime)
    for cc=1:ncohorts
        fprintf('   cohort %i (ages %i to %i): agent dist %.1f, AllStats %.1f, LifeCycleProfiles %.1f, AutoCorr %.1f seconds \n',cc,caliboptions.cohortagejshifter(cc),lastages(cc),cohorttimes(cc,1),cohorttimes(cc,2),cohorttimes(cc,3),cohorttimes(cc,4))
    end
end

%% Option to log moments (if targets are log, then this will have been already applied)
if any(caliboptions.logmoments>0) % need to log some moments
    currentmomentvec=(1-caliboptions.logmoments).*currentmomentvec + caliboptions.logmoments.*log(currentmomentvec.*caliboptions.logmoments+(1-caliboptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Evaluate the objective function (which is being minimized)
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if caliboptions.vectoroutput==1
    % Output the vector of currentmomentvec
    % Main use it for computing derivatives of moments with respect to parameters
    Obj=currentmomentvec(actualtarget);
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
        Obj=(currentmomentvec(actualtarget)-targetmomentvec(actualtarget))'*caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj=sum(caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget)).^2,'omitnan');
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj=sum(caliboptions.weights.*(log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget)).^2),'omitnan');
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj=Obj/length(CalibParamNames); % This is done so that the tolerances for convergence are sensible

    if penalty>0
        if Obj>0
            Obj=1.2*penalty*Obj; % 20% penalty for being too far in violation of restrictions
        else % Obj is negative, so penalty is to reduce magnitude
            Obj=0.8*(1/penalty)*Obj; % 20% penalty for being too far in violation of restrictions
        end
    end
elseif caliboptions.vectoroutput==2
    % Weighted vector (for use with least-squares residuals algorithms)
    % Note: the outer-layers of code already took 'square root' of the weights
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Is essentially the square-root of 'MethodOfMoments' [it is the form of input used by Matlab's lsqnonlin()]
        Obj=caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj=caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj=caliboptions.weights.*log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget));
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj=gather(Obj); % lsqnonlin() doesn't work with gpu, so have to gather()
    if penalty>0
        Obj=[Obj; sqrt(penalty)]; % append penalty residual: contributes 'penalty' to lsqnonlin's sum(Obj.^2)
    else
        Obj=[Obj; 0]; % penalty=0 (params inside cutoffs); append 0 to keep residual length constant
    end
end


%% Verbose
if caliboptions.verbose==1
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    if caliboptions.vectoroutput==0
        fprintf('Current objective fn value is %8.12f \n', Obj)
        if penalty>0
            if Obj>0
                fprintf('Current penalty is to multiply objective fn by %8.2f \n', 1.2*penalty)
            else  % Obj is negative, so penalty is to reduce magnitude
                fprintf('Current penalty is to multiply objective fn by %8.2f \n', 0.8*(1/penalty) )
            end
        end
    elseif caliboptions.vectoroutput==2
        % Obj is the vector of residuals here (with the penalty residual appended), so print its sum of squares; the penalty is a residual, not a multiplier
        % [before 2026-10-07 the penalty print below sat outside this if, so with vectoroutput=2 it tested the vector Obj>0 (ill-defined) and described a multiplier that does not exist in the vector form]
        fprintf('Current (sum-of-squares of) objective fn value is %8.12f \n', Obj'*Obj)
        if penalty>0
            fprintf('Current penalty residual is %8.6f (it adds %8.6f to the sum-of-squares) \n', sqrt(penalty), penalty)
        end
    end
end









end
