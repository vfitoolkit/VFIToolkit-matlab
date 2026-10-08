function Obj=CalibrateInfHorzAgentModel_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, ParametrizeParamsFn, FnsToEvaluate, usingallstats,usingautocorr,usingcrosssec,usingcustomstats, targetmomentvec, allstatmomentnames,autocorrmomentnames,crosssecmomentnames,cmsmomentnames, allstatcummomentsizes,autocorrcummomentsizes,crossseccummomentsizes,cmscummomentsizes, AllStats_whichstats,AutoCorrStats_whichstats,CrossSecStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_AutoCorrStats, FnsToEvaluate_CrossSecStats, calibparamsvecindex, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats,
% AutoCorrTransProbs and CrossSectionCovarCorr. Lastly there is caliboptions, which also carries
% caliboptions.whichcombos (1 or 0), caliboptions.selectors and caliboptions.autocorrtimehorizons from SetupTargetMoments_InfHorz.

% Untransform the parameters (when dealing with constraints the inputs are the transformed parameters, so want to switch them back to original model parameters)
[calibparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(calibparamsvec,calibparamsvecindex,CalibParamNames,caliboptions);

if caliboptions.verbose==1
    fprintf(' \n')
    fprintf('Current parameter values: \n')
    for pp=1:length(CalibParamNames)
        if calibparamsvecindex(pp+1)-calibparamsvecindex(pp)==1
            fprintf(['    ',CalibParamNames{pp},'= %8.6f \n'],calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))
        else
            fprintf(['    ',CalibParamNames{pp},'=  \n'])
            calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))' % want the output as a row
        end
    end
end

for pp=1:length(CalibParamNames)
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix{sum(calibomitparams_counter(1:pp))}; % (a cell since 2026-10-08: one omitted-values vector per masked parameter, of any length)
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        Parameters.(CalibParamNames{pp})=currparamraw;
    else
        Parameters.(CalibParamNames{pp})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
    end
end

%% ParametrizeParamsFn can be used to parametrize the parameters
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Do grids if those depend on parameters being calibrated (otherwise they are already done)
if caliboptions.calibrateshocks==1
    % Internally, only ever use joint-grids (makes all the code much easier to write)
    % The user's own grids are needed if CustomModelStats is given them
    KeepOriginalGrid=(usingcustomstats==1 && caliboptions.CustomModelStats_usergrids==1);
    [z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_gridvals,pi_z,Parameters,vfoptions,3,KeepOriginalGrid);
    % output: z_gridvals, pi_z, vfoptions.e_gridvals, vfoptions.pi_e
    simoptions.e_gridvals=vfoptions.e_gridvals;
    simoptions.pi_e=vfoptions.pi_e;
end


%% Solve the model
[V, Policy]=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

StationaryDist=StationaryDist_InfHorz(Policy,n_d,n_a,n_z,pi_z,simoptions,Parameters);

%% Custom Model Stats
if usingcustomstats==1
    if caliboptions.CustomModelStats_usergrids==0
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    elseif caliboptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            caliboptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            caliboptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,caliboptions.CustomModelStatsInputs.z_grid,caliboptions.CustomModelStatsInputs.pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    end
end

%% Calculate model stats
% caliboptions.whichcombos=1 (CalibrateInfHorzAgentModel's default): the stats commands compute only the targeted (function, statistic,
% restriction) combinations, through simoptions.whichcombos (and, for AllStats, a per-combination simoptions.whichstats) built by
% SetupTargetMoments_InfHorz (caliboptions.selectors). =0: every statistic of every targeted function (whichstats all ones). The three
% commands take differently shaped selectors, so each gets its own copy of simoptions. The conditional restrictions stay in simoptions
% for all three (restricted AutoCorr and cross-section targets are allowed since 2026-10-08; the restrictions used to be stripped
% before those two commands).
if usingallstats==1
    simoptions_AllStats=simoptions;
    if caliboptions.whichcombos==1
        simoptions_AllStats.whichcombos=caliboptions.selectors.AllStats.whichcombos;
        simoptions_AllStats.whichstats=caliboptions.selectors.AllStats.whichstats;
    else % caliboptions.whichcombos=0: every statistic of every targeted function
        simoptions_AllStats.whichstats=ones(1,7);
    end
    AllStats=EvalFnOnAgentDist_AllStats_InfHorz(StationaryDist,Policy, FnsToEvaluate_AllStats,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_gridvals,simoptions_AllStats);
end
if usingautocorr==1
    simoptions_AutoCorr=simoptions;
    if isfield(simoptions,'timehorizons')
        simoptions_AutoCorr.timehorizons=union(simoptions.timehorizons,caliboptions.autocorrtimehorizons); % the horizons the targets name (their _kK suffixes), plus any the user asked for
    else
        simoptions_AutoCorr.timehorizons=caliboptions.autocorrtimehorizons;
    end
    if caliboptions.whichcombos==1
        simoptions_AutoCorr.whichcombos=caliboptions.selectors.AutoCorr.whichcombos; % [nFns,1+nRestr] (the command has no per-combination whichstats; it reports its byproducts)
    end
    AutoCorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz(StationaryDist,Policy,FnsToEvaluate_AutoCorrStats,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,simoptions_AutoCorr);
end
if usingcrosssec==1
    simoptions_CrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_CrossSec.whichcombos=caliboptions.selectors.CrossSec.whichcombos; % pair-shaped, [nFns,nFns,1+nRestr] (the restricted targets are on pages 2:end)
    end
    CrossSectionCovarCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz(StationaryDist,Policy,FnsToEvaluate_CrossSecStats,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_gridvals,simoptions_CrossSec);
end


%% Get current values of the target moments as a vector
% Each kind: walk the names of the target (one to four levels, the empty cells skipped) into the command's output, which has the same
% nesting (restrictions, MoreInequality, CovarianceWith/CorrelationWith and the matrices alike). An AutoCorr statistic with a _kK
% horizon suffix is read from the command's .tperiodsK.(statistic). [Before 2026-10-08 the first cross-section row always read a
% third name, so a two-level target (fn.Mean) as the first cross-section target errored.]
currentmomentvec=zeros(size(targetmomentvec));
sofar=0;
if usingallstats==1
    for cc=1:size(allstatmomentnames,1)
        if cc==1
            idx=sofar+1:sofar+allstatcummomentsizes(1);
        else
            idx=sofar+allstatcummomentsizes(cc-1)+1:sofar+allstatcummomentsizes(cc);
        end
        if all(isnan(targetmomentvec(idx))) % a wholly omitted target: nothing was selected for it, so the command may not have computed (or created) it
            currentmomentvec(idx)=NaN;
            continue
        end
        temp=AllStats;
        for kk=1:size(allstatmomentnames,2)
            if ~isempty(allstatmomentnames{cc,kk})
                temp=temp.(allstatmomentnames{cc,kk});
            end
        end
        currentmomentvec(idx)=temp(:);
    end
    sofar=sofar+allstatcummomentsizes(end);
end
if usingautocorr==1
    for cc=1:size(autocorrmomentnames,1)
        if cc==1
            idx=sofar+1:sofar+autocorrcummomentsizes(1);
        else
            idx=sofar+autocorrcummomentsizes(cc-1)+1:sofar+autocorrcummomentsizes(cc);
        end
        if all(isnan(targetmomentvec(idx))) % a wholly omitted target (e.g. .(restriction).(fn).(stat)=NaN): its combination is off, so the command did not create the struct
            currentmomentvec(idx)=NaN;
            continue
        end
        names=autocorrmomentnames(cc,:);
        names=names(~cellfun(@isempty,names));
        tk=regexp(names{end},'_k(\d+)$','tokens','once'); % a horizon suffix: stat_kK is the command's .tperiodsK.(stat)
        if ~isempty(tk)
            names=[names(1:end-1),{['tperiods',tk{1}]},{names{end}(1:end-length(tk{1})-2)}];
        end
        temp=AutoCorrTransProbs;
        for kk=1:numel(names)
            temp=temp.(names{kk});
        end
        currentmomentvec(idx)=temp(:);
    end
    sofar=sofar+autocorrcummomentsizes(end);
end
if usingcrosssec==1
    for cc=1:size(crosssecmomentnames,1)
        if cc==1
            idx=sofar+1:sofar+crossseccummomentsizes(1);
        else
            idx=sofar+crossseccummomentsizes(cc-1)+1:sofar+crossseccummomentsizes(cc);
        end
        if all(isnan(targetmomentvec(idx))) % a wholly omitted target: nothing was selected for it, so the command may not have created it
            currentmomentvec(idx)=NaN;
            continue
        end
        temp=CrossSectionCovarCorr;
        for kk=1:size(crosssecmomentnames,2)
            if ~isempty(crosssecmomentnames{cc,kk})
                temp=temp.(crosssecmomentnames{cc,kk});
            end
        end
        currentmomentvec(idx)=temp(:);
    end
    sofar=sofar+crossseccummomentsizes(end);
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
if caliboptions.vectoroutput==1 % vector output
    % Output the vector of currentmomentvec
    % Main use it for computing derivatives of moments with respect to parameters
    Obj=currentmomentvec(actualtarget);
elseif caliboptions.vectoroutput==0 % scalar output
    % currentmomentvec is the current moment values
    % targetmomentvec is the target moment values
    % Both are column vectors

    % Note: MethodOfMoments and sum_squared are doing essentially the same calculation (only different is size of weights,
    % which will be a matrix for MethodOfMoments but a vector for sum_squared), I just write them in ways that make it more
    % obvious that they do what they say.
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Obj=(targetmomentvec-currentmomentvec)'*caliboptions.weights*(targetmomentvec-currentmomentvec);
        % For the purpose of doing log(moments) I switched to the following (otherwise getting silly current moments can seem attractive)
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
