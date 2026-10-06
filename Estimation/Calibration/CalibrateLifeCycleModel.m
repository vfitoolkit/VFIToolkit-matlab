function [CalibParams,calibsummary]=CalibrateLifeCycleModel(CalibParamNames,TargetMoments,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist, AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.


%% Setup caliboptions
if ~isfield(caliboptions,'verbose')
    caliboptions.verbose=1; % sum of squares is the default
end
if ~isfield(caliboptions,'constrainpositive')
    caliboptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Convert constrained positive p into x=log(p) which is unconstrained.
    % Then use p=exp(x) in the model.
end
if ~isfield(caliboptions,'constrainpositivemethod')
    caliboptions.constrainpositivemethod='softplus'; % 'log' (uparam=log(cparam)) or 'softplus' (cparam=log(1+exp(uparam))); see ParameterConstraints_TransformParamsToUnconstrained for which suits which parameter
end
if ~isfield(caliboptions,'constrain0to1')
    caliboptions.constrain0to1={}; % names of parameters to constrained to be 0 to 1 (gets converted to binary-valued vector below)
    % Handle 0 to 1 constraints by using log-odds function to switch parameter p into unconstrained x, so x=log(p/(1-p))
    % Then use the logistic-sigmoid p=1/(1+exp(-x)) when evaluating model.
end
if ~isfield(caliboptions,'constrainAtoB')
    caliboptions.constrainAtoB={}; % names of parameters to constrained to be A to B (gets converted to binary-valued vector below)
    % Handle A to B constraints by converting y=(p-A)/(B-A) which is 0 to 1, and then treating as constrained 0 to 1 y (so convert to unconstrained x using log-odds function)
    % Once we have the 0 to 1 y (by converting unconstrained x with the logistic sigmoid function), we convert to p=A+(B-A)*y
elseif ~isempty(caliboptions.constrainAtoB)
    if ~isfield(caliboptions,'constrainAtoBlimits')
        error('You have used caliboptions.constrainAtoB, but are missing caliboptions.constrainAtoBlimits')
    end
end
if ~isfield(caliboptions,'logmoments')
    caliboptions.logmoments=0;
    % =1 means log() the model moments [target moments and CoVarMatrixDataMoments should already be based on log(moments) if you are using this+
    % =1 means applies log() to all moments, unless you specify them seperately as on next line
    % You can name moments in the same way you would for the targets, e.g.
    % caliboptions.logmoments.AgeConditionalStats.earnings.Mean=1
    % Will log that moment, but not any other moments.
    % Note: the input target moment should log(moment). Same for the covariance matrix
    % of the data moments, CoVarMatrixDataMoments, should be of the log moments.
end
if ~isfield(caliboptions,'metric')
    caliboptions.metric='sum_squared'; % sum of squares is the default
    % Other options are: sum_logratiosquared: sum of squares of the log-ratio (target/model)
end
if ~isfield(caliboptions,'weights')
    caliboptions.weights=1; % all moments have equal weights is default (this is a vector of one, just don't know the length yet :)
end
if ~isfield(caliboptions,'whichcombos')
    caliboptions.whichcombos=1; % =1: AllStats and LifeCycleProfiles compute only the targeted (function, statistic, restriction, age) combinations
    % (simoptions.whichcombos and per-combination whichstats, built from TargetMoments by SetupTargetMoments_FHorz); =0: every
    % statistic of every targeted function is computed (whichstats all ones). The moments are identical either way, =1 is faster.
end
if ~isfield(caliboptions,'toleranceparams')
    caliboptions.toleranceparams=10^(-4); % tolerance accuracy of the calibrated parameters
end
if ~isfield(caliboptions,'toleranceobjective')
    caliboptions.toleranceobjective=10^(-6); % tolerance accuracy of the objective function
end
if ~isfield(caliboptions,'fminalgo')
    caliboptions.fminalgo=8; % lsqnonlin(), recast as a least-squares residuals problem and solve it that way
    % Currently, all the caliboptions.metric choices can be done as setup as least-squares residuals problems
    % caliboptions.fminalgo=4; % CMA-ES, I tried fminsearch() by default but it regularly fails to converge to a decent solution
end
caliboptions.simulatemoments=0; % Not needed here (the objectivefn is shared with other estimation commands)
caliboptions.vectoroutput=0; % Not needed here (the objectivefn is shared with other estimation commands)

caliboptions.useCustomModelStats=0;
if isfield(caliboptions,'CustomModelStats')
    caliboptions.useCustomModelStats=1;
    if ~isfield(caliboptions,'CustomModelStats_usergrids')
        caliboptions.CustomModelStats_usergrids=0; % =0: pass internal z_gridvals_J & pi_z_J; =1: pass exactly the z_grid & pi_z the user input
    end
    % Stash some of the inputs so they can be passed to CustomModelStats later (only things we otherwise override).
    % So that user gets exactly what they input, not any internally reworked things
    if caliboptions.CustomModelStats_usergrids==1
        caliboptions.CustomModelStatsInputs.z_grid=z_grid;
        caliboptions.CustomModelStatsInputs.pi_z=pi_z;
    end
    % Need the following two as otherwise they would contain alreadygridvals=1
    caliboptions.CustomModelStatsInputs.vfoptions=vfoptions;
    caliboptions.CustomModelStatsInputs.simoptions=simoptions;
end



%% Setup for which parameters are being calibrated

% Sometimes we want to omit parameters
if isfield(caliboptions,'omitcalibparam')
    OmitCalibParamsNames=fieldnames(caliboptions.omitcalibparam);
else
    OmitCalibParamsNames={''};
end
calibparamsvec0=[]; % column vector
calibparamsvecindex=zeros(length(CalibParamNames)+1,1); % Note, first element remains zero
calibomitparams_counter=zeros(length(CalibParamNames),1); % column vector: calibomitparamsvec allows omitting the parameter for certain ages
calibomitparamsmatrix=zeros(N_j,1); % Each row is of size N_j-by-1 and holds the omitted values of a parameter
for pp=1:length(CalibParamNames)
    if any(strcmp(OmitCalibParamsNames,CalibParamNames{pp}))
        % This parameter is under an omit-mask, so need to only use part of it
        tempparam=Parameters.(CalibParamNames{pp});
        tempomitparam=caliboptions.omitcalibparam.(CalibParamNames{pp});
        % Make them both column vectors
        if size(tempparam,1)==1
            tempparam=tempparam';
        end
        if size(tempomitparam,1)==1 % (was testing tempparam, so a row omit mask was never transposed and the check below broadcast row against column)
            tempomitparam=tempomitparam';
        end
        % If the omit and initial guess do not fit together, throw an error
        if ~all(tempomitparam(~isnan(tempomitparam))==tempparam(~isnan(tempomitparam)))
            fprintf('Following are the name, omit value, and initial value that related to following error (they should be the same in the non-NaN entries to be calibated) \n')
            CalibParamNames{pp}
            caliboptions.omitcalibparam.(CalibParamNames{pp})
            Parameters.(CalibParamNames{pp})
            error('You have set an omitted calibated parameter, but the set values do not match the initial guess')
        end
        tempparam=tempparam(isnan(tempomitparam)); % only keep those which are NaN, not those with value for omitted
        % Keep the parts which should be calibated
        calibparamsvec0=[calibparamsvec0; tempparam]; % Note: it is already a column
        calibparamsvecindex(pp+1)=calibparamsvecindex(pp)+length(tempparam);
        % Store the whole thing
        calibomitparams_counter(pp)=1;
        calibomitparamsmatrix(:,sum(calibomitparams_counter))=tempomitparam;
    else
        % Get all the parameters
        if size(Parameters.(CalibParamNames{pp}),2)==1
            calibparamsvec0=[calibparamsvec0; Parameters.(CalibParamNames{pp})];
        else
            calibparamsvec0=[calibparamsvec0; Parameters.(CalibParamNames{pp})']; % transpose
        end
        calibparamsvecindex(pp+1)=calibparamsvecindex(pp)+length(Parameters.(CalibParamNames{pp}));
    end
end

% If the parameter is constrained in some way then we need to transform it
[calibparamsvec0,caliboptions]=ParameterConstraints_TransformParamsToUnconstrained(calibparamsvec0,calibparamsvecindex,CalibParamNames,caliboptions,1);
% Also converts the constraints info in caliboptions to be a vector rather than by name.



%% Setup for which moments are being targeted
% Only calculate each of AllStats and LifeCycleProfiles when being used (so as faster when not using both)
[targetmomentvec,usingallstats,usinglcp,usingcustomstats, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats,FnsToEvaluate_AllStats, acsmomentnames, acscummomentsizes, ACStats_whichstats,FnsToEvaluate_ACStats, cmsmomentnames,cmscummomentsizes,selectors]=SetupTargetMoments_FHorz(TargetMoments,FnsToEvaluate,0,N_j,simoptions);
caliboptions.selectors=selectors; % the per-combination whichcombos/whichstats of the two stats commands (used when caliboptions.whichcombos=1)


%% Set-up/check caliboptions.weights
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if isscalar(caliboptions.weights)
    caliboptions.weights=caliboptions.weights.*ones(size(targetmomentvec(actualtarget)));
else % Make sure it is a column vector
    if size(caliboptions.weights,1)==1 % currently a row vector
        caliboptions.weights=caliboptions.weights';
    end
end
if length(caliboptions.weights)~=length(targetmomentvec(actualtarget))
    error('caliboptions.weights is not the length same as number of target moments (ignoring any NaN)')
end


%% Now, a bunch of things to avoid redoing them every parameter vector we want to try
% Note: I avoid doing this for ReturnFnParamNames because they are so
% dependent on the setup. Same for FnsToEvaluateParamNames
ReturnFnParamNames=[];
FnsToEvaluateParamNames=[];

%% Set up exogenous shock grids now (so they can then just be reused every time)
% Check if using ExogShockFn or EiidShockFn, and if so, do these use a
% parameter that is being calibrated
caliboptions.calibrateshocks=0; % set to one if need to redo shocks for each new calib parameter vector
if isfield(vfoptions,'ExogShockFn')
    temp=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    % can just leave action space in here as we only use it to see if CalibParamNames is part of it
    if ~isempty(intersect(temp,CalibParamNames))
        caliboptions.calibrateshocks=1;
    end
end
if isfield(vfoptions,'EiidShockFn') % note: not elseif, can have both and either alone should trigger redoing the shocks
    temp=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    % can just leave action space in here as we only use it to see if CalibParamNames is part of it
    if ~isempty(intersect(temp,CalibParamNames))
        caliboptions.calibrateshocks=1;
    end
end
if ~isfield(simoptions,'jequaloneDist_usergrids')
    simoptions.jequaloneDist_usergrids=1; % =1: pass jequaloneDist (as a function) the z_grid in the form the user input; =0: pass the internal joint-grid form
end
if caliboptions.calibrateshocks==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    % The user's own grids are needed if jequaloneDist as a function is given them, or if CustomModelStats is
    KeepOriginalGrid=(simoptions.jequaloneDist_usergrids==1 || (caliboptions.useCustomModelStats==1 && caliboptions.CustomModelStats_usergrids==1));
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3,KeepOriginalGrid);
    if KeepOriginalGrid==1 && isfield(vfoptions,'user_z_grid')
        % ExogShockFn builds the user's own grid internally, so take it from there rather than from the z_grid input (which is then just a placeholder)
        caliboptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
        caliboptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
    end
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
    if simoptions.jequaloneDist_usergrids==1 % jequaloneDist as a function is given the user's own grids, so pass them through
        simoptions.user_z_grid=vfoptions.user_z_grid;
        simoptions.user_pi_z=vfoptions.user_pi_z;
    end
else
    % just need some placeholders (the shocks are rebuilt inside the objective function every evaluation)
    z_gridvals_J=[];
    pi_z_J=[];
end
% Regardless of whether they are done here of in _objectivefn, they will be
% precomputed by the time we get to the value fn, stationary dist, etc. So
vfoptions.alreadygridvals=1;
simoptions.alreadygridvals=1;


%%
% caliboptions.logmoments: which moments to take logs of (the targets must then already be log(moments); same for any covariance matrix of the data moments).
% Four forms: a scalar 0 (none) or 1 (all); a vector with one entry per target (same length as targetmomentvec); a vector with one entry
% per TARGET NAME (one per row of allstatmomentnames, then acsmomentnames, then cmsmomentnames, in that order), expanded over the
% entries of each; or by name, e.g. caliboptions.logmoments.AgeConditionalStats.earnings.Mean=1 (names not mentioned are 0).
% Internally it becomes a vector with one entry per target.
momentrowsizes=[]; % the number of entries of each target name, in the order the names enter targetmomentvec
allstatsizes=diff([0,allstatcummomentsizes]); acssizes=diff([0,acscummomentsizes]); cmssizes=diff([0,cmscummomentsizes]);
if usingallstats==1
    momentrowsizes=[momentrowsizes, allstatsizes];
end
if usinglcp==1
    momentrowsizes=[momentrowsizes, acssizes];
end
if usingcustomstats==1
    momentrowsizes=[momentrowsizes, cmssizes];
end
if isstruct(caliboptions.logmoments)
    logmomentnames=caliboptions.logmoments;
    caliboptions.logmoments=zeros(length(targetmomentvec),1);
    sofar=0;
    if usingallstats==1
        for ii=1:size(allstatmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'AllStats') && isfield(logmomentnames.AllStats,allstatmomentnames{ii,1}) && isfield(logmomentnames.AllStats.(allstatmomentnames{ii,1}),allstatmomentnames{ii,2})
                temp=logmomentnames.AllStats.(allstatmomentnames{ii,1}).(allstatmomentnames{ii,2});
                if isempty(allstatmomentnames{ii,3})
                    flag=temp;
                elseif isstruct(temp) && isfield(temp,allstatmomentnames{ii,3})
                    flag=temp.(allstatmomentnames{ii,3});
                end
            end
            caliboptions.logmoments(sofar+1:sofar+allstatsizes(ii))=flag; % (a scalar flag applies to every entry of the target)
            sofar=sofar+allstatsizes(ii);
        end
    end
    if usinglcp==1
        for ii=1:size(acsmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'AgeConditionalStats') && isfield(logmomentnames.AgeConditionalStats,acsmomentnames{ii,1}) && isfield(logmomentnames.AgeConditionalStats.(acsmomentnames{ii,1}),acsmomentnames{ii,2})
                temp=logmomentnames.AgeConditionalStats.(acsmomentnames{ii,1}).(acsmomentnames{ii,2});
                if isempty(acsmomentnames{ii,3})
                    flag=temp;
                elseif isstruct(temp) && isfield(temp,acsmomentnames{ii,3})
                    flag=temp.(acsmomentnames{ii,3});
                end
            end
            caliboptions.logmoments(sofar+1:sofar+acssizes(ii))=flag;
            sofar=sofar+acssizes(ii);
        end
    end
    if usingcustomstats==1
        for ii=1:size(cmsmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'CustomModelStats') && isfield(logmomentnames.CustomModelStats,cmsmomentnames{ii,1})
                flag=logmomentnames.CustomModelStats.(cmsmomentnames{ii,1});
            end
            caliboptions.logmoments(sofar+1:sofar+cmssizes(ii))=flag;
            sofar=sofar+cmssizes(ii);
        end
    end
elseif any(caliboptions.logmoments>0)
    if isscalar(caliboptions.logmoments)
        caliboptions.logmoments=ones(length(targetmomentvec),1); % log all of them
    elseif length(caliboptions.logmoments)==length(targetmomentvec)
        caliboptions.logmoments=reshape(caliboptions.logmoments,[length(targetmomentvec),1]); % already one entry per target
    elseif length(caliboptions.logmoments)==length(momentrowsizes)
        caliboptions.logmoments=repelem(reshape(caliboptions.logmoments,[],1),reshape(momentrowsizes,[],1)); % one entry per target name, expanded over the entries of each
    else
        fprintf('Relevant to following error: length(caliboptions.logmoments)=%i \n', length(caliboptions.logmoments))
        fprintf('Relevant to following error: number of target names=%i, number of target entries=%i \n', length(momentrowsizes), length(targetmomentvec))
        error('You are using caliboptions.logmoments, but length(caliboptions.logmoments) matches neither the number of target names nor the number of target entries')
    end
else
    caliboptions.logmoments=zeros(length(targetmomentvec),1);
end
%% Set up the objective function and the initial calibration parameter vector
if caliboptions.fminalgo~=8
    CalibrationObjectiveFn=@(calibparamsvec) CalibrateLifeCycleModel_objectivefn(calibparamsvec,CalibParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,usingcustomstats,targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes,AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, calibparamsvecindex, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions);
elseif caliboptions.fminalgo==8
    caliboptions.vectoroutput=2;
    weightsbackup=caliboptions.weights;
    caliboptions.weights=sqrt(caliboptions.weights); % To use a weighting matrix in lsqnonlin(), we work with the square-roots of the weights
    CalibrationObjectiveFn=@(calibparamsvec) CalibrateLifeCycleModel_objectivefn(calibparamsvec,CalibParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,usingcustomstats,targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes, AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, calibparamsvecindex, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions);
    caliboptions.weights=weightsbackup; % change it back now that we have set up CalibrateLifeCycleModel_objectivefn()
end

% calibparamsvec0 is our initial guess for calibparamsvec


%% Choosing algorithm for the optimization problem
% https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
minoptions = optimset('TolX',caliboptions.toleranceparams,'TolFun',caliboptions.toleranceobjective);
if caliboptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    caliboptions.multiGEcriterion=0;
    [calibparamsvec,calibobjvalue]=fzero(CalibrationObjectiveFn,calibparamsvec0,minoptions);
elseif caliboptions.fminalgo==1
    [calibparamsvec,calibobjvalue]=fminsearch(CalibrationObjectiveFn,calibparamsvec0,minoptions);
elseif caliboptions.fminalgo==2
    % Use the optimization toolbox so as to take advantage of automatic differentiation
    z=optimvar('z',length(calibparamsvec0));
    optimfun=fcn2optimexpr(CalibrationObjectiveFn, z);
    prob = optimproblem("Objective",optimfun);
    z0.z=calibparamsvec0;
    [sol,calibobjvalue]=solve(prob,z0);
    calibparamsvec=sol.z;
    % Note, doesn't really work as automatic differentiation is only for
    % supported functions, and the objective here is not a supported function
elseif caliboptions.fminalgo==3
    goal=zeros(length(calibparamsvec0),1);
    weight=ones(length(calibparamsvec0),1); % I already implement weights via caliboptions
    [calibparamsvec,calibsummaryVec] = fgoalattain(CalibrationObjectiveFn,calibparamsvec0,goal,weight);
    calibobjvalue=sum(abs(calibsummaryVec));
elseif caliboptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
    % https://en.wikipedia.org/wiki/CMA-ES
    % https://cma-es.github.io/
    % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
    if ~isfield(caliboptions,'insigma')
        % insigma: initial coordinate wise standard deviation(s)
        caliboptions.insigma=0.3*abs(calibparamsvec0)+0.1*(calibparamsvec0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
    end
    if ~isfield(caliboptions,'inopts')
        % inopts: options struct, see defopts below
        caliboptions.inopts=[];
    end
    % varargin (unused): arguments passed to objective function
    if caliboptions.verbose==1
        disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
    end
	% This is a minor edit of cmaes, because I want to use 'CalibrationObjectiveFn' as a function_handle, but the original cmaes code only allows for 'CalibrationObjectiveFn' as a string
    [calibparamsvec,calibobjvalue,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(CalibrationObjectiveFn,calibparamsvec0,caliboptions.insigma,caliboptions.inopts); % ,varargin);
elseif caliboptions.fminalgo==5
    % Update based on rules in caliboptions.fminalgo5.howtoupdate
    error('fminalgo=5 is not possible with model calibration/estimation')
elseif caliboptions.fminalgo==6
    if ~isfield(caliboptions,'lb') || ~isfield(caliboptions,'ub')
        error('When using constrained optimization (caliboptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using caliboptions.lb and caliboptions.ub')
    end
    [calibparamsvec,calibobjvalue]=fmincon(CalibrationObjectiveFn,calibparamsvec0,[],[],[],[],caliboptions.lb,caliboptions.ub,[],minoptions);
elseif caliboptions.fminalgo==7 % fsolve()
    error('cannot use fminalgo=7 for estimation (as fsolve() is a multi-objective method)')
elseif caliboptions.fminalgo==8 % lsqnonlin()
    minoptions = optimoptions('lsqnonlin','FiniteDifferenceStepSize',1e-2,'TolX',caliboptions.toleranceparams,'TolFun',caliboptions.toleranceobjective);
    [calibparamsvec,calibobjvalue]=lsqnonlin(CalibrationObjectiveFn,calibparamsvec0,[],[],[],[],[],[],[],minoptions);
end


%% Model moments at the solution, for calibsummary (one more evaluation of the objective, as a vector)
caliboptions_summary=caliboptions;
caliboptions_summary.vectoroutput=1;
caliboptions_summary.verbose=0;
calibsummary.currentmomentvec=gather(CalibrateLifeCycleModel_objectivefn(calibparamsvec,CalibParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,usingcustomstats,targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes,AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, calibparamsvecindex, calibomitparams_counter, calibomitparamsmatrix, caliboptions_summary, vfoptions,simoptions));
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
calibsummary.targetmomentvec=targetmomentvec(actualtarget); % the targets, in the order of the names (AllStats, then AgeConditionalStats, then CustomModelStats), NaN entries dropped
calibsummary.logmoments=caliboptions.logmoments; % one entry per target: the current moments above are log() where this is 1 (the targets were given as logs there)

%% Clean up outputs
% If the parameter is constrained in some way then we need to un-transform it
[calibparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(calibparamsvec,calibparamsvecindex,CalibParamNames,caliboptions);
if sum(penalty)>0
    warning('penalty for the parameter constraints is non-zero (some parameters are not satisfying the constraints)')
end
for pp=1:length(CalibParamNames)
    % Now store the unconstrained values
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix(:,sum(calibomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        CalibParams.(CalibParamNames{pp})=currparamraw;
    else
        CalibParams.(CalibParamNames{pp})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
    end
end

calibsummary.objvalue=calibobjvalue; % Output the objective value











end