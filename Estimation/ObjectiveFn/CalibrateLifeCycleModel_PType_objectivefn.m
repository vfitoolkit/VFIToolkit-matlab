function Obj=CalibrateLifeCycleModel_PType_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, usingallstats,usinglcp,usingcustomstats, targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes, AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, usingautocorr, autocorrmomentnames, autocorrcummomentsizes, FnsToEvaluate_AutoCorr, autocorrtimehorizons, usingcrosssec, crosssecmomentnames, crossseccummomentsizes, FnsToEvaluate_CrossSec, usingagecrosssec, agecrosssecmomentnames, agecrossseccummomentsizes, FnsToEvaluate_AgeCrossSec, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats,
% LifeCycleProfiles, AutoCorrTransProbs and the two CrossSectionCovarCorr
% commands (all the PType versions). Lastly there is caliboptions.

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
    % The user's own grids are needed if CustomModelStats is given them
    KeepOriginalGrid=(usingcustomstats==1 && caliboptions.CustomModelStats_usergrids==1);
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


%% Solve the model and calculate the stats
[V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,pi_z_J,Parameters,simoptions);

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

%% Calculate model stats
if usingallstats==1
    % caliboptions.whichcombos=1 (CalibrateLifeCycleModel_PType's default): the stats commands compute only the targeted (function, statistic,
    % restriction[, age], ptype or grouped) combinations, through simoptions.whichcombos and a per-combination simoptions.whichstats with a
    % trailing type dimension, built by SetupTargetMoments_FHorz (caliboptions.selectors). =0: every statistic of every targeted function
    % (whichstats all ones). caliboptions.whichcombos is always set by the calling command. The two commands take differently shaped selectors, so each gets its own copy of simoptions.
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
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate_ACStats,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_ACStats);
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
    CorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_AutoCorr,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,pi_z_J,simoptions_AutoCorr);
end
if usingcrosssec==1
    simoptions_CrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_CrossSec.whichcombos=caliboptions.selectors.CrossSec.whichcombos; % pair-shaped, [nFns,nFns,N_i+1], or [nFns,nFns,1+nRestr,N_i+1] with conditional restrictions (restricted targets on pages 2:end)
    end
    CrossSectionCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_CrossSec,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_CrossSec);
end
if usingagecrosssec==1
    simoptions_AgeCrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_AgeCrossSec.whichcombos=caliboptions.selectors.AgeCrossSec.whichcombos; % pair-shaped per age group, [nFns,nFns,ngroups,N_i+1], or [nFns,nFns,ngroups,1+nRestr,N_i+1] with conditional restrictions
    end
    AgeConditionalCrossSectionCorr=EvalFnOnAgentDist_AgeConditionalStats_CrossSectionCovarCorr_FHorz_PType(StationaryDist,Policy,FnsToEvaluate_AgeCrossSec,Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions_AgeCrossSec);
end



%% Get current values of the target moments as a vector
% Each kind: walk the names of the target (two to five levels, the empty cells skipped) into the command's output, which has the same
% nesting (restrictions, permanent types, MoreInequality, CovarianceWith/CorrelationWith and the matrices alike). A target that is NaN
% in full (SetupTargetMoments_FHorz warned about it and selected nothing for it, so the command may not have created it) is skipped
% and filled with NaN (it is dropped from the objective anyway) [2026-10-09; before, the walk died on the absent struct].
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
    if usingcustomstats==1
        fprintf('Current CustomModelStats variables: \n')
        for ii=1:length(cmsmomentnames)
            fprintf('	%s: %8.4f \n',cmsmomentnames{ii},CustomStats.(cmsmomentnames{ii}))
        end
    end
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