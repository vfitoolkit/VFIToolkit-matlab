function [EstimParams, EstimParamsConfInts,estsummary]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments,WeightingMatrix,CoVarMatrixDataMoments,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions)
% Note: Inputs are EstimParamNames,TargetMoments, WeightingMatrix, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is estimoptions.

% Performs method of moments estimation, minimizing
%    (M_d-M_m(theta))' W (M_d-M_m(theta))
% where M_d are data moments, M_m are model moments that depend on a vector
% of parameters to be estimated (theta), and W is a weighting matrix.

% EstimParamNames: field containing the names of the parameters to be estimated
% TargetMoments: structure containing the moments to be targeted
% WeightingMatrix: the weighting matrix W
% CoVarMatrixDataMoments: the covariance matrix of data moments (needed to compute the standard errors of the estimated parameters)

%% Setup estimoptions
if ~isfield(estimoptions,'verbose')
    estimoptions.verbose=1; % sum of squares is the default
end
if ~isfield(estimoptions,'constrainpositive')
    estimoptions.constrainpositive={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Convert constrained positive p into x=log(p) which is unconstrained.
    % Then use p=exp(x) in the model.
end
if ~isfield(estimoptions,'constrain0to1')
    estimoptions.constrain0to1={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Handle 0 to 1 constraints by using log-odds function to switch parameter p into unconstrained x, so x=log(p/(1-p))
    % Then use the logistic-sigmoid p=1/(1+exp(-x)) when evaluating model.
end
if ~isfield(estimoptions,'constrainAtoB')
    estimoptions.constrainAtoB={}; % names of parameters to constrained to be positive (gets converted to binary-valued vector below)
    % Handle A to B constraints by converting y=(p-A)/(B-A) which is 0 to 1, and then treating as constrained 0 to 1 y (so convert to unconstrained x using log-odds function)
    % Once we have the 0 to 1 y (by converting unconstrained x with the logistic sigmoid function), we convert to p=A+(B-a)*y
else
    if ~isfield(estimoptions,'constrainAtoBlimits')
        error('You have used estimoptions.constrainAtoB, but are missing estimoptions.constrainAtoBlimits')
    end
end
if ~isfield(estimoptions,'logmoments')
    estimoptions.logmoments=0; 
    % =1 means log() the model moments [target moments and CoVarMatrixDataMoments should already be based on log(moments) if you are using this+
    % =1 means applies log() to all moments, unless you specify them seperately as on next line
    % You can name moments in the same way you would for the targets, e.g.
    % estimoptions.logmoments.AgeConditionalStats.earnings.Mean=1
    % Will log that moment, but not any other moments.
    % Note: the input target moment should log(moment). Same for the covariance matrix
    % of the data moments, CoVarMatrixDataMoments, should be of the log moments.
end
if ~isfield(estimoptions,'confidenceintervals')
    estimoptions.confidenceintervals=90; % the default is to report 90-percent confidence intervals
end
if ~isfield(estimoptions,'eedefault')
    estimoptions.eedefault=3; % 1,2,3 or 4: Default epsilon value is epsilonraw*epsilonmodvec(eedefault)
    % Controls how big is the epsilon used to calculate derivatives as finite difference
    % Roughly, 1 means e-08, 2 means e-06, 3 means e-04, 4 means e-02,
end
if ~isfield(estimoptions,'toleranceparams')
    estimoptions.toleranceparams=10^(-4); % tolerance accuracy of the calibrated parameters
end
if ~isfield(estimoptions,'toleranceobjective')
    estimoptions.toleranceobjective=10^(-6); % tolerance accuracy of the objective function
end
if ~isfield(estimoptions,'fminalgo')
    estimoptions.fminalgo=8; % lsqnonlin(), recast GMM as a least-squares residuals problem and solve it that way
end
if ~isfield(estimoptions,'iterateGMM')
    estimoptions.iterateGMM=1;
    % =1; default, no iteration
    % =2, uses two-iteration efficient GMM
    % Note: When doing two-iteration efficient GMM, just input CoVarMatrixDataMoments=[]
    % Note: Can do more than 2 iterations,  e.g., estimoptions.iterateGMM=5 will do five-iteration efficient GMM
end
if ~isfield(estimoptions,'bootstrapStdErrors')
    estimoptions.bootstrapStdErrors=0; % =1, bootstraps the standard errors (instead of based on derivatives, which is the default)
end
if ~isfield(estimoptions,'numbootstrapssims')
    % When doing two-step GMM, or bootstraping Standard Errors
    estimoptions.numbootstrapsims=100; % Number of simulations
end
if ~isfield(estimoptions,'numberinvidualsperbootstrapsim')
    % When doing two-step GMM, or bootstraping Standard Errors
    estimoptions.numberinvidualsperbootstrapsim=1000; 
    % Note that each individual simulation will be N_j periods, so total
    % 'observations' is more like estimoptions.numberinvidualsperbootstrapsim*N_j
    % [Note: you cannot use simoptions.numbersims, as that is overwritten by estimoptions.numberinvidualsperbootstrapsim]
end
if ~isfield(estimoptions,'efficientW')
    estimoptions.efficientW=0; % =1, Calculates std error of parameters under assumption that the weighting matrix is efficient (that the weighting matrix is the inverse of the covariance matrix of the data moments)
end
if ~isfield(estimoptions,'skipestimation')
    estimoptions.skipestimation=0; % =1, skips the estimation, is here so you can do estimation, and then rerun later to bootstrap the standard errors without reestimating the whole model
end
% Following are estimoptions used internally, but which the user won't want to set themselves
estimoptions.vectoroutput=0; % Set to zero to get point estimates, then later set to one as part of computing Jacobian matrix J (needed for Sigma, among other things).
estimoptions.simulatemoments=0; % Set to zero to get point estimates, is needed later if you bootstrap standard errors
% estimoptions.rngindex will be set below if you have estimoptions.simulatemoments=1 to bootstrap standard errors
estimoptions.metric='MethodOfMoments';
% estimoptions.weights=WeightingMatrix; is set below, after check it is correct size
if ~isfield(estimoptions,'previousiterations')
    estimoptions.previousiterations.niters=0; % gets incremented for each iteration when using estimoptions.iterateGMM
end

% Optional:
% E.g., estimoptions.CalibParamsNames={'theta'}, then estimoptions will
% include a measure of sensitivity of estimated parameters to the
% pre-calibrated parameters (here the pre-calibrated parameter is 'theta')


if estimoptions.iterateGMM>1
    if ~isempty(CoVarMatrixDataMoments)
        warning('You have estimoptions.iterateGMM>1, so the contents of CoVarMatrixDataMoments are going to be ignored (as they are irrelevant to two-step efficient GMM [Nothing wrong, just warning, you can pass CoVarMatrixDataMoments=[] to get rid of this msg]')
    end
end
if estimoptions.bootstrapStdErrors==1
    if ~isempty(CoVarMatrixDataMoments)
        warning('You have estimoptions.bootstrapStdErrors=1, so the contents of CoVarMatrixDataMoments are going to be ignored (as they are irrelevant to bootstrapped std errors for GMM [Nothing wrong, just warning, you can pass CoVarMatrixDataMoments=[] to get rid of this msg]')
    end
end



%% Setup for which parameters are being estimated. Create the parameters as a vector: estimparamsvec0.

% Sometimes we want to omit parameters
if isfield(estimoptions,'omitestimparam')
    OmitEstimParamsNames=fieldnames(estimoptions.omitestimparam);
else
    OmitEstimParamsNames={''};
end
estimparamsvec0=[]; % column vector
estimparamsvecindex=zeros(length(EstimParamNames)+1,1); % Note, first element remains zero
estimomitparams_counter=zeros(length(EstimParamNames),1); % column vector: estimomitparamsvec allows omiting the parameter for certain ages
estimomitparamsmatrix=zeros(N_j,1); % Each row is of size N_j-by-1 and holds the omited values of a parameter
for pp=1:length(EstimParamNames)
    if any(strcmp(OmitEstimParamsNames,EstimParamNames{pp}))
        % This parameter is under an omit-mask, so need to only use part of it
        tempparam=Parameters.(EstimParamNames{pp});
        tempomitparam=estimoptions.omitestimparam.(EstimParamNames{pp});
        % Make them both column vectors
        if size(tempparam,1)==1
            tempparam=tempparam';
        end
        if size(tempparam,1)==1
            tempomitparam=tempomitparam';
        end
        % If the omit and initial guess do not fit together, throw an error
        if ~all(tempomitparam(~isnan(tempomitparam))==tempparam(~isnan(tempomitparam)))
            fprintf('Following are the name, omit value, and initial value that related to following error (they should be the same in the non-NaN entries to be estimated) \n')
            EstimParamNames{pp}
            estimoptions.omitestimparam.(EstimParamNames{pp})
            Parameters.(EstimParamNames{pp})
            error('You have set an omitted estimated parameter, but the set values do not match the initial guess')
        end
        tempparam=tempparam(isnan(tempomitparam)); % only keep those which are NaN, not those with value for omitted
        % Keep the parts which should be estimated
        estimparamsvec0=[estimparamsvec0; tempparam]; % Note: it is already a column
        estimparamsvecindex(pp+1)=estimparamsvecindex(pp)+length(tempparam);
        % Store the whole thing
        estimomitparams_counter(pp)=1;
        estimomitparamsmatrix(:,sum(estimomitparams_counter))=tempomitparam;
    else
        % Get all the parameters
        if size(Parameters.(EstimParamNames{pp}),2)==1
            estimparamsvec0=[estimparamsvec0; Parameters.(EstimParamNames{pp})];
        else
            estimparamsvec0=[estimparamsvec0; Parameters.(EstimParamNames{pp})']; % transpose
        end
        estimparamsvecindex(pp+1)=estimparamsvecindex(pp)+length(Parameters.(EstimParamNames{pp}));
    end
end

% If the parameter is constrained in some way then we need to transform it
[estimparamsvec0,estimoptions]=ParameterConstraints_TransformParamsToUnconstrained(estimparamsvec0,estimparamsvecindex,EstimParamNames,estimoptions,1);
% Also converts the constraints info in estimoptions to be a vector rather than by name.


%% Setup for which moments are being targeted
% Only calculate each of AllStats and LifeCycleProfiles when being used (so as faster when not using both)
[targetmomentvec,usingallstats,usinglcp, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, acsmomentnames, acscummomentsizes, ACStats_whichstats]=SetupTargetMoments(TargetMoments,0);


%% Now, a bunch of things to avoid redoing them every parameter vector we want to try
% Note: I avoid doing this for ReturnFnParamNames because they are so
% dependent on the setup. Same for FnsToEvaluateParamNames
ReturnFnParamNames=[];
FnsToEvaluateParamNames=[];

estimoptions.calibrateshocks=0; % set to one if need to redo shocks for each new calib parameter vector
if isfield(vfoptions,'ExogShockFn')
    temp=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    if ~isempty(intersect(temp,CalibParamNames))
        estimoptions.calibrateshocks=1;
    end
elseif isfield(vfoptions,'EiidShockFn')
    estimoptions.calibrateshocks=1;
    temp=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    if ~isempty(intersect(temp,CalibParamNames))
        estimoptions.calibrateshocks=1;
    end
end
if estimoptions.calibrateshocks==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
    vfoptions.alreadygridvals=1;
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
end


%%
if all(size(WeightingMatrix)==[sum(~isnan(targetmomentvec)),sum(~isnan(targetmomentvec))])
    estimoptions.weights=WeightingMatrix;
else
    fprintf('Following two lines relate to the error below \n')
    fprintf('size(WeightingMatrix)=%i-by-%i \n',size(WeightingMatrix,1),size(WeightingMatrix,2))
    fprintf('you are targeting %i moments (this is number of elements that are not NaN, total number of elements is %i) \n', sum(~isnan(targetmomentvec)), length(targetmomentvec))
    error('size(WeightingMatrix) should be a square matrix with number of rows (and number of columns) equal to the number of moments to be estimated')
end

%% 
% estimoptions.logmoments can be specified by names
if isstruct(estimoptions.logmoments)
    logmomentnames=estimoptions.logmoments;
    % replace estimoptions.logmoments with a vector as this is what gets used internally
    estimoptions.logmoments=zeros(length(targetmomentvec),1);
    if any(strcmp(fieldnames(logmomentnames),'AllStats'))
        estimoptions.logmoments(1:allstatcummomentsizes(1))=logmomentnames.AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2})*ones(allstatcummomentsizes(1),1);
        for ii=2:size(allstatmomentnames,1)
            estimoptions.logmoments(allstatcummomentsizes(ii-1)+1:allstatcummomentsizes(ii))=logmomentnames.AllStats.(allstatmomentnames{ii,1}).(allstatmomentnames{ii,2})*ones(allstatcummomentsizes(ii)-allstatcummomentsizes(ii-1),1);
        end
    end
    if any(strcmp(fieldnames(logmomentnames),'AgeConditionalStats'))
        estimoptions.logmoments(1:acscummomentsizes(1))=logmomentnames.AgeConditionalStats.(acsmomentnames{1,1}).(acsmomentnames{1,2})*ones(acscummomentsizes(1),1);
        for ii=2:size(acsmomentnames,1)
            estimoptions.logmoments(acscummomentsizes(ii-1)+1:acscummomentsizes(ii))=logmomentnames.AgeConditionalStats.(acsmomentnames{ii,1}).(acsmomentnames{ii,2})*ones(acscummomentsizes(ii)-acscummomentsizes(ii-1),1);
        end
    end

% If estimoptions.logmoments is not a structure, then...
% estimoptions.logmoments will either be scalar, or a vector of zeros and ones
%    [scalar of zero is interpreted as vector of zeros, scalar of one is interpreted as vector of ones]
elseif any(estimoptions.logmoments>0) % =1 means log of moments (can be set up as vector, zeros(length(EstimParamNames),1)
   % If set this up, and then set up 
   if isscalar(estimoptions.logmoments)
       estimoptions.logmoments=ones(length(targetmomentvec),1); % log all of them
   else
        if length(estimoptions.logmoments)==(length(acsmomentnames)+length(allstatmomentnames))
            % Covert estimoptions.logmoments from being about EstimParamNames
            temp=estimoptions.logmoments;
            estimoptions.logmoments=zeros(length(targetmomentvec),1);
            cumsofar=1;
            for mm=1:length(temp)
                if mm<=allstatmomentsizes
                    estimoptions.logmoments(cumsofar:cumsofar+allstatmomentsizes(mm))=temp(mm);
                    cumsofar=cumsofar+allstatmomentsizes(mm);
                else
                    estimoptions.logmoments(cumsofar:cumsofar+acsmomentsizes(mm))=temp(mm);
                    cumsofar=cumsofar+acsmomentsizes(mm);
                end
            end
        elseif length(estimoptions.logmoments)==length(targetmomentvec)
            % This is fine (already in the appropriate form)
        else
            fprintf('Relevant to following error: length(estimoptions.logmoments)=%i \n', length(estimoptions.logmoments))
            fprintf('Relevant to following error: length(acsmomentnames)=%i, length(allstatmomentnames)=%i \n', length(acsmomentnames), length(allstatmomentnames))
            error('You are using estimoptions.logmoments, but length(estimoptions.logmoments) does not match number of moments to estimate [they should be equal]')
        end
   end
   % log of targetmoments [no need to do this as inputs should already be log()]
   % targetmomentvec=(1-estimoptions.logmoments).*targetmomentvec + estimoptions.logmoments.*log(targetmomentvec.*estimoptions.logmoments+(1-estimoptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Set up the objective function and the initial calibration parameter vector
% Note: _objectivefn is shared between Method of Moments Estimation and Calibration
if estimoptions.fminalgo~=8
    EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);
elseif estimoptions.fminalgo==8
    estimoptions.vectoroutput=2;
    estimoptions.weights=chol(estimoptions.weights,'upper'); % To use a weighting matrix in lsqnonlin(), we work with the upper-cholesky decomposition
    EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);
    estimoptions.weights=WeightingMatrix; % change it back now that we have set up CalibrateLifeCycleModel_objectivefn()
end

% estimparamsvec0 is our initial guess for estimparamsvec
% estimparamsvec0 is in 'unconstrained' parameters form


%% Choosing algorithm for the optimization problem
if estimoptions.skipestimation==0
    % https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
    minoptions = optimset('TolX',estimoptions.toleranceparams,'TolFun',estimoptions.toleranceobjective);
    if estimoptions.fminalgo==0
        error('cannot use fminalgo=0 for estimation (as fzero() is a multi-objective method)')
    elseif estimoptions.fminalgo==1
        [estimparamsvec,fval]=fminsearch(EstimateMoMObjectiveFn,estimparamsvec0,minoptions);
    elseif estimoptions.fminalgo==2
        % Use the optimization toolbox so as to take advantage of automatic differentiation
        z=optimvar('z',length(estimparamsvec0));
        optimfun=fcn2optimexpr(EstimateMoMObjectiveFn, z);
        prob = optimproblem("Objective",optimfun);
        z0.z=estimparamsvec0;
        [sol,fval]=solve(prob,z0);
        estimparamsvec=sol.z;
        % Note, doesn't really work as automatic differentiation is only for
        % supported functions, and the objective here is not a supported function
    elseif estimoptions.fminalgo==3
        goal=zeros(length(estimparamsvec0),1);
        weight=ones(length(estimparamsvec0),1); % I already implement weights via caliboptions
        [estimparamsvec,calibsummaryVec] = fgoalattain(EstimateMoMObjectiveFn,estimparamsvec0,goal,weight);
        fval=sum(abs(calibsummaryVec));
    elseif estimoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
        % https://en.wikipedia.org/wiki/CMA-ES
        % https://cma-es.github.io/
        % Code is cmaes.m from: https://cma-es.github.io/cmaes_sourcecode_page.html#matlab
        if ~isfield(estimoptions,'insigma')
            % insigma: initial coordinate wise standard deviation(s)
            estimoptions.insigma=0.3*abs(estimparamsvec0)+0.1*(estimparamsvec0==0); % Set standard deviation to 30% of the initial parameter value itself (cannot input zero, so add 0.1 to any zeros)
        end
        if ~isfield(estimoptions,'inopts')
            % inopts: options struct, see defopts below
            estimoptions.inopts=[];
        end
        % varargin (unused): arguments passed to objective function
        if estimoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
        end
    	% This is a minor edit of cmaes, because I want to use 'CalibrationObjectiveFn' as a function_handle, but the original cmaes code only allows for 'CalibrationObjectiveFn' as a string
        [estimparamsvec,fval,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(EstimateMoMObjectiveFn,estimparamsvec0,estimoptions.insigma,estimoptions.inopts); % ,varargin);

        estimoptions.cmaesoutputs.counteval=counteval;
        estimoptions.cmaesoutputs.stopflag=stopflag;
        estimoptions.cmaesoutputs.out=out;
        estimoptions.cmaesoutputs.bestever=bestever;
    elseif estimoptions.fminalgo==5
        % Update based on rules in caliboptions.fminalgo5.howtoupdate
        error('fminalgo=5 is not possible with model calibration/estimation')
    elseif estimoptions.fminalgo==6
        if ~isfield(estimoptions,'lb') || ~isfield(estimoptions,'ub')
            error('When using constrained optimization (caliboptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using caliboptions.lb and caliboptions.ub')
        end
        [estimparamsvec,fval]=fmincon(EstimateMoMObjectiveFn,estimparamsvec0,[],[],[],[],estimoptions.lb,estimoptions.ub,[],minoptions);
    elseif estimoptions.fminalgo==7 % fsolve()
        error('cannot use fminalgo=7 for estimation (as fsolve() is a multi-objective method)')
    elseif estimoptions.fminalgo==8 % lsqnonlin()
        minoptions = optimoptions('lsqnonlin','FiniteDifferenceStepSize',1e-2,'TolX',estimoptions.toleranceparams,'TolFun',estimoptions.toleranceobjective);
        [estimparamsvec,fval]=lsqnonlin(EstimateMoMObjectiveFn,estimparamsvec0,[],[],[],[],[],[],[],minoptions);
    end

else % estimoptions.skipestimation==1    
    warning('Skipping the estimation step (you have set estimoptions.skipestimation=1 in EstimateLifeCycleModel_MethodOfMoments() [Nothing wrong with this, just warning as want to be sure you did this on purpose]')
    % The values in Parameters are taken as the estimated values for EstimParams
    % Note that we already got these as estimparamsvec0, so we can just set
    estimparamsvec=estimparamsvec0;
end

%% estimparamsvec contains the (transformed) unconstrained parameters, not the original (constrained) parameter values.
[estimparamsvec,~]=ParameterConstraints_TransformParamsToOriginal(estimparamsvec,estimparamsvecindex,EstimParamNames,estimoptions);
% estimparamsvec is now the original (constrained) parameter values.


%% Two-iteration efficient GMM (actually, n-iteration, but just uses this recursively)
if estimoptions.iterateGMM>1 && estimoptions.skipestimation==0
    error('HAVE NOT YET IMPLEMENTED ITERATED GMM (you have estimoptions.iterateGMM>1)')
end


%% Bootstrap standard errors
if estimoptions.bootstrapStdErrors==1
    error('HAVE NOT YET IMPLEMENTED BOOTSTRAP STANDARD ERRORS (you have estimoptions.bootstrapStdErrors=1)')
end





%% Compute the standard deviation of the estimated parameters
% Part of the standard deviations is to compute J (the jacobian matrix of derivatives of model moments to the estimated parameters).
% To make it easier to compute the derivatives by finite-difference, I turn off the parameter constraints and just use the model 
% parameter values (rather than the internal-transformed-parameters) directly. Just makes it easier to follow what is going on (at least in my head).
% To faciliate this I use estimoptionsJacobian=estimoptions, but with modifications.
% Later I did some searching, and it seems there are no precise answers online, but some people (Python 'optimagic' on github) made same decision I did, of taking 
% derivatives based on 'external' parameters rather than 'internal' (transformed) parameters.
% Other open issue, what do you do when the resulting standard deviations mean confidence intervals reach outside your contraints?
if estimoptions.bootstrapStdErrors==0
    % First, need the Jacobian matrix, which involves computing all the
    % derivatives of the individual moments with respect to the estimated parameters
    estimoptionsJacobian=estimoptions;
    estimoptionsJacobian.constrainpositive=zeros(length(EstimParamNames),1); % eliminate constraints for Jacobian
    estimoptionsJacobian.constrain0to1=zeros(length(EstimParamNames),1); % eliminate constraints for Jacobian
    estimoptionsJacobian.constrainAtoB=zeros(length(EstimParamNames),1); % eliminate constraints for Jacobian
    % Note: idea is that we don't want to apply constraints inside CalibrateLifeCycleModel_objectivefn() while computing finite-differences
    estimoptionsJacobian.vectoroutput=1; % Was set to zero to get point estimates, now set to one as part of computing std deviations.
    estimoptionsJacobian.verbose=0; % otherwise looks a bit weird

    % According to https://en.wikipedia.org/wiki/Numerical_differentiation#Step_size
    % A good step size to compute the derivative of f(x) is epsilon*x with
    epsilonraw=sqrt(2.2)*10^(-8); % Note: this is sqrt(eps(1.d0)), the eps() is Matlab command that gives floating point precision
    % I am going to compute the upper and lower first differences
    % I then use the smallest of the two (as that gives the larger/more conservative, standard deviations)

    % Decided to actually do four different values of epsilon, then report J
    % for all so user can see how they look (are the derivatives sensitive to epsilon)
    epsilonmodvec=[1,10^2,10^4,10^6];
    % Default value of epsilon
    eedefault=estimoptions.eedefault; % Default epsilon value is epsilonraw*epsilonmodvec(eedefault)

    % For parameters of size 10^(-2) or less, use alternative epsilon values
    epsilonalt=[10^(-2),10^(-2),10^(-1),10^(-1)]; % Note: this must be same length as epsilonmodvec (default follows eedefault)
    
    %% We want to calculate derivatives from epsilon changes in the model parameters
    % I want to do epsilon change in the model parameter, but here I have the unconstrained parameters. So I create an epsilonparamup and
    % epsilonparamdown, which contain the unconstrained values that correspond to epsilon changes in the constrained parameters.
    % I do this in a separate loop, which is a loss of runtime, but this is minor and is much easier to read so whatever
    epsilonparamup=zeros(length(estimparamsvec),length(epsilonmodvec));
    epsilonparamdown=zeros(length(estimparamsvec),length(epsilonmodvec));
    modelestimparamsvec=estimparamsvec;
    modelestimparamsvecup=zeros(size(modelestimparamsvec));
    modelestimparamsvecdown=zeros(size(modelestimparamsvec));
    violateconstrainttop=zeros(size(modelestimparamsvec)); %=1 means use a one-sided (down) finite-difference because 'adding epsilon' would lead to a parameter value that violates the constraint
    violateconstraintbottom=zeros(size(modelestimparamsvec)); %=1 means use a one-sided (up) finite-difference because 'subtracting epsilon' would lead to a parameter value that violates the constraint
    % Switch modelestimparamsvec to the constrained (model) parameters
    for pp=1:length(EstimParamNames)
        if estimoptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
            % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
            modelestimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=exp(modelestimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        elseif estimoptions.constrain0to1(pp)==1
            % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
            modelestimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=1/(1+exp(-modelestimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))));
        end
        % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
        if estimoptions.constrainAtoB(pp)==1
            % Constrain parameter to be A to B
            modelestimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=estimoptions.constrainAtoBlimits(pp,1)+(estimoptions.constrainAtoBlimits(pp,2)-estimoptions.constrainAtoBlimits(pp,1))*modelestimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
            % Note, this parameter will have first been converted to 0 to 1 already, so just need to further make it A to B
            % y=A+(B-A)*x, converts 0-to-1 x, into A-to-B y
        end
    end
    % Now, multiply by (1+-epsilon)
    for ee=1:length(epsilonmodvec)
        epsilon=epsilonmodvec(ee)*epsilonraw;
        for pp=1:length(EstimParamNames)
            % 'Add/subtract' epsilon
            if floor(log(abs(modelestimparamsvec(pp)))/log(10))>-2 % order of magnitude is greater than 10^(-2)
                modelestimparamsvecup(pp)=(1+epsilon)*modelestimparamsvec(pp); % add epsilon*x to the pp-th parameter
                modelestimparamsvecdown(pp)=(1-epsilon)*modelestimparamsvec(pp); % subtract epsilon*x from the pp-th parameter
            elseif floor(log(abs(modelestimparamsvec(pp)))/log(10))<-4 % parameter is so small that actually just add/subtract epsilon to/from x [have to do this for x=0, and this seems a reasonable cutoff]
                modelestimparamsvecup(pp)=epsilon+modelestimparamsvec(pp); % add epsilon to the pp-th parameter
                modelestimparamsvecdown(pp)=-epsilon+modelestimparamsvec(pp); % subtract epsilon from the pp-th parameter
            else % is the modelestimparamsvec itself is small, use alternative values of epsilon
                modelestimparamsvecup(pp)=(1+epsilonalt(ee))*modelestimparamsvec(pp); % add epsilonalt*x to the pp-th parameter
                modelestimparamsvecdown(pp)=(1-epsilonalt(ee))*modelestimparamsvec(pp); % subtract epsilonalt*x from the pp-th parameter
            end

            % Enforce that we do not violate the constraints
            if estimoptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
                if modelestimparamsvecdown(pp)<=0
                    violateconstraintbottom(pp)=1;
                end
            elseif estimoptions.constrainAtoB(pp)==1 % Constrain A to B
                if modelestimparamsvecdown(pp)<=estimoptions.constrainAtoBlimits(pp,1) % less than A
                    violateconstraintbottom(pp)=1;
                elseif modelestimparamsvecup(pp)>=estimoptions.constrainAtoBlimits(pp,2) % greater than B
                    violateconstrainttop(pp)=1;
                end
            elseif estimoptions.constrain0to1(pp)==1 % Constrain 0 to 1 (but not as part of A to B)
                if modelestimparamsvecdown(pp)<=0
                    violateconstraintbottom(pp)=1;
                elseif modelestimparamsvecup(pp)>=1
                    violateconstrainttop(pp)=1;
                end
            end
        end
        % Store the epsilon parameters
        epsilonparamup(:,ee)=modelestimparamsvecup;
        epsilonparamdown(:,ee)=modelestimparamsvecdown;
    end
    
    
    %% Can now calculate derivatives to the epsilon change in parameters as the finite-difference
    for ee=1:length(epsilonmodvec)
        % ObjValue is used to compute f(x+h), f(x), and f(x-h), and then then can be used to evaluate the finite-differences
        ObjValue_upwind=zeros(sum(~isnan(targetmomentvec)),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'
        ObjValue_downwind=zeros(sum(~isnan(targetmomentvec)),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'
        
        % Note: estimoptions.vectoroutput=1, so ObjValue is a vector 
        epsilonparamvec=modelestimparamsvec; % and using estimoptionsJacobian, so using the actual parameters, rather than the transformed parameters
        ObjValue=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions);
        for pp=1:length(estimparamsvec)
            epsilonparamvec=modelestimparamsvec;
            if violateconstrainttop(pp)==0 % if ==1, we will just use down
                epsilonparamvec(pp)=epsilonparamup(pp,ee); % add epsilon*x to the pp-th parameter
                ObjValue_upwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions);
            end
            if violateconstraintbottom(pp)==0 % if ==1, we will just use up
                epsilonparamvec(pp)=epsilonparamdown(pp,ee); % subtract epsilon*x from the pp-th parameter
                ObjValue_downwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions);
            end
        end
        
        % Use finite-difference to compute the derivatives
        J_up=(ObjValue_upwind-ObjValue)./((epsilonparamup(:,ee)-modelestimparamsvec)');
        J_down=(ObjValue-ObjValue_downwind)./((modelestimparamsvec-epsilonparamdown(:,ee))');
        J_centered=(ObjValue_upwind-ObjValue_downwind)./((epsilonparamup(:,ee)-epsilonparamdown(:,ee))');
        % Jacobian matix of derivatives of model moments with respect to parameters, evaluated at the parameter point estimates
        
        % J is nmonents-by-nparams
        J_full=J_centered;
        % If epsilon changes pushed us outside the parameter constraints, then we just use the one-sided finite-differences
        for pp=1:length(estimparamsvec)
            if violateconstraintbottom(pp)==1 % 'subtracting epsilon' violates lower bound on parameter value, so just use J_up
                J_full(pp,:)=J_up(pp,:);
            elseif violateconstrainttop(pp)==1 % 'adding epsilon' violates upper bound on parameter value, so just use J_down
                J_full(pp,:)=J_down(pp,:);
            end
        end
        
        % Double-checks: reports various steps around computing the derivatives and standard deviations, and does so for various epsilon sizes
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J=J_full;
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J_centered=J_centered;
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J_up=J_up;
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J_down=J_down;
        if estimoptions.efficientW==0
            % This is standard formula for the asymptotic variance of method of moments estimator
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma=((J_full'*WeightingMatrix*J_full)^(-1)) * J_full'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_full * ((J_full'*WeightingMatrix*J_full)^(-1));
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_centered=((J_centered'*WeightingMatrix*J_centered)^(-1)) * J_centered'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_centered * ((J_centered'*WeightingMatrix*J_centered)^(-1));
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_up=((J_up'*WeightingMatrix*J_up)^(-1)) * J_up'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_up * ((J_up'*WeightingMatrix*J_up)^(-1));
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_down=((J_down'*WeightingMatrix*J_down)^(-1)) * J_down'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_down * ((J_down'*WeightingMatrix*J_down)^(-1));
        elseif estimoptions.efficientW==1
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma=(J_full'*WeightingMatrix*J_full)^(-1);
            % When using the efficient weighting matrix W=Omega^(-1), the asymptotic variance of the method of moments estimator simplifies to
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_centered=(J_centered'*WeightingMatrix*J_centered)^(-1);
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_up=(J_up'*WeightingMatrix*J_up)^(-1);
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_down=(J_down'*WeightingMatrix*J_down)^(-1);
        end
        tempestimparamscovarmatrix_diag=diag(estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma); % Just the diagonal of the covar matrix of the parameter vector
        for pp=1:length(EstimParamNames)
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).EstimParamsStdDev.(EstimParamNames{pp})=sqrt(tempestimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        end
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvec=modelestimparamsvec; % Is actually independent of ee anyway
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvecup=epsilonparamup(:,ee);
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvecdown=epsilonparamdown(:,ee);
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).violateconstraintbottom=violateconstraintbottom;
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).violateconstrainttop=violateconstrainttop;
        if ee==eedefault
            J=J_full; % This is the one used to report Sigma (parameter std deviations) [corresponds to epsilon=sqrt(2.2)*10^(-4)]

            disp('temp print')
            dbstack
            ee
            J_up
            J_down
            J_centered

            modelestimparamsvec
            epsilonparamup(:,ee)
            epsilonparamdown(:,ee)

            ObjValue
            ObjValue_upwind
            ObjValue_downwind
        end
    end
    
    % For later
    epsilon=epsilonmodvec(eedefault)*epsilonraw; % sqrt(2.2)*10^(-6)
    % What I have here as default uses epsilon of the order 10^(-6)
    % Grey Gordon's numerical derivative code used 10^(-6)
    
    if estimoptions.efficientW==0
        estimparamscovarmatrix=((J'*WeightingMatrix*J)^(-1)) * J'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J * ((J'*WeightingMatrix*J)^(-1));
        % This is standard formula for the asymptotic variance of method of moments estimator
        % See, e.g., Kirkby - "Classical (not Simulated!) Method of Moments Estimation of Life-Cycle Models"
    elseif estimoptions.efficientW==1
        % When using the efficient weighting matrix W=Omega^(-1), the asymptotic variance of the method of moments estimator simplifies to
        estimparamscovarmatrix=(J'*WeightingMatrix*J)^(-1);
    end


    % While we are here, if you do skip estimation, compute the objective function and output this (is useful for checking out alternative estimates)
    if estimoptions.skipestimation==1
        estimoptionsJacobian.vectoroutput=0; % using estimoptionsJacobian, so using the actual parameters, rather than the transformed parameters
        ObjValue=CalibrateLifeCycleModel_objectivefn(modelestimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions);
        fval=ObjValue;
        clear estimoptionsJacobian
    end
end




%% Local identification
if estimoptions.bootstrapStdErrors==0 % Depends on derivatives, so cannot do when bootstapping the standard errors
    % The estimate is locally identified if the matrix J is full rank
    estsummary.localidentification.rankJ=rank(J); % If this is greater or equal to number of parameters, then locally identified
    estsummary.localidentification.yesidentified=logical(rank(J)>=length(estimparamsvec));
    estsummary.notes.localidentification='If the Jacobian matrix (derivatives of model moments with respect to parameter vector) is full rank then the model is locally identified [so rank(J) should be greater than or equal to number of parameters being estimated]';
end


%% Some additional outputs
% Mainly, the Sensitivity matrix
if estimoptions.bootstrapStdErrors==0 % Depends on derivatives, so cannot do when bootstapping the standard errors
    % Sensitivity of estimated parameters to the target moments
    % Sensitivity matrix, Lambda, of Andrews, Gentzkow & Shapiro (2017) - Measuring the Sensitivity of Parameter Estimates to Estimation Moments
    SensitivityMatrix=(-(J'*WeightingMatrix*J)^(-1))*(J'*WeightingMatrix);
    estsummary.sensitivitymatrix=SensitivityMatrix;
    
    % Sensitivity of estimated parameters to the pre-calibrated parameters
    % If you have set estimoptions.CalibParamNames; Jorgensen (2023) - Sensitivity to Calibrated Parameters
    % Requires calculating derivatives of the objective vector to the calibrated parameters
    if isfield(estimoptions,'CalibParamsNames')
        calibparamvec=zeros(length(estimoptions.CalibParamsNames),1);
        ObjValue_upwind=zeros(sum(~isnan(targetmomentvec)),length(estimoptions.CalibParamsNames)); % Jacobian matrix of 'derivative of model moments with respect to pre-calibrated parameters, evaluated at estimated parameter point estimates'
        ObjValue_downwind=zeros(sum(~isnan(targetmomentvec)),length(estimoptions.CalibParamsNames)); % Jacobian matrix of 'derivative of model moments with respect to  pre-calibrated parameters, evaluated at estimated parameter point estimates'

        CalibParams=struct();
        for pp=1:length(estimoptions.CalibParamsNames)
            CalibParams.(estimoptions.CalibParamsNames{pp})=Parameters.(estimoptions.CalibParamsNames{pp});
            calibparamvec(pp)=Parameters.(estimoptions.CalibParamsNames{pp});
        end

        % Note: estimoptions.vectoroutput=1, so ObjValue is a vector
        % Can now calculate derivatives to the epsilon change in parameters as the finite-difference
        % ObjValue is used to compute f(x+h), f(x), and f(x-h), and then then can be used to evaluate the finite-differences
        for pp=1:length(estimoptions.CalibParamsNames)
            % 'Add' epsilon
            if floor(log(abs(modelestimparamsvec(pp)))/log(10))>-2 % order of magnitude is greater than 10^(-2)
                Parameters.(estimoptions.CalibParamsNames{pp})=(1+epsilon)*CalibParams.(estimoptions.CalibParamsNames{pp}); % add epsilon*x to the pp-th parameter
            elseif floor(log(abs(modelestimparamsvec(pp)))/log(10))<-4 % parameter is so small that actually just add/subtract epsilon to/from x [have to do this for x=0, and this seems a reasonable cutoff]
                Parameters.(estimoptions.CalibParamsNames{pp})=epsilon+CalibParams.(estimoptions.CalibParamsNames{pp}); % add epsilon to the pp-th parameter
            else % is the modelestimparamsvec itself is small, use alternative values of epsilon
                Parameters.(estimoptions.CalibParamsNames{pp})=(1+epsilonalt(eedefault))*CalibParams.(estimoptions.CalibParamsNames{pp});  % add epsilonalt*x to the pp-th parameter
            end
            if violateconstrainttop(pp)==0 % if ==1, we will just use down
                ObjValue_upwind(:,pp)=CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions); % use estimoptionsJacobian
            end
            % 'Subtract' epsilon
            if floor(log(abs(modelestimparamsvec(pp)))/log(10))>-2 % order of magnitude is greater than 10^(-2)
                Parameters.(estimoptions.CalibParamsNames{pp})=(1+epsilon)*CalibParams.(estimoptions.CalibParamsNames{pp}); % subtract epsilon*x from the pp-th parameter
            elseif floor(log(abs(modelestimparamsvec(pp)))/log(10))<-4 % parameter is so small that actually just add/subtract epsilon to/from x [have to do this for x=0, and this seems a reasonable cutoff]
                Parameters.(estimoptions.CalibParamsNames{pp})=epsilon+CalibParams.(estimoptions.CalibParamsNames{pp}); % subtract epsilon from the pp-th parameter
            else % is the modelestimparamsvec itself is small, use alternative values of epsilon
                Parameters.(estimoptions.CalibParamsNames{pp})=(1+epsilonalt(eedefault))*CalibParams.(estimoptions.CalibParamsNames{pp});  % subtract epsilonalt*x from the pp-th parameter
            end
            if violateconstraintbottom(pp)==0 % if ==1, we will just use up
                ObjValue_downwind(:,pp)=CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions); % use estimoptionsJacobian
            end
            % restore calib param
            Parameters.(estimoptions.CalibParamsNames{pp})=CalibParams.(estimoptions.CalibParamsNames{pp});
        end

        % Use finite-difference to compute the derivatives
        Jcalib_up=(ObjValue_upwind-ObjValue)./(epsilon*calibparamvec');
        Jcalib_down=(ObjValue-ObjValue_downwind)./(epsilon*calibparamvec');
        Jcalib_centered=(ObjValue_upwind-ObjValue_downwind)./(2*epsilon*calibparamvec');
        % Jacobian matix of derivatives of model moments with respect to parameters, evaluated at the parameter point estimates

        % Sensitivity matrix of Jorgensen (2023) - Sensitivity to Calibrated Parameters
        estsummary.sensitivitytocalibrationmatrix=SensitivityMatrix*Jcalib_centered; % This is the formula in Corollary 1 of Jorgensen (2023)

        estsummary.doublechecks.Jcalib=Jcalib_centered;
        % also, just so user can see them
        estsummary.doublechecks.Jcalib_up=Jcalib_up;
        estsummary.doublechecks.Jcalib_down=Jcalib_down;        
    end

end


%% Clean up the first two outputs
for pp=1:length(EstimParamNames)
    if estimoptions.skipestimation==0
        % If parameter is constrained, switch it back to the unconstrained value
        if estimoptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
            % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
            estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=exp(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        elseif estimoptions.constrain0to1(pp)==1
            % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
            estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=1/(1+exp(-estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))));
        end
        % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
        if estimoptions.constrainAtoB(pp)==1
            % Constrain parameter to be A to B
            estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=estimoptions.constrainAtoBlimits(pp,1)+(estimoptions.constrainAtoBlimits(pp,2)-estimoptions.constrainAtoBlimits(pp,1))*estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
            % Note, this parameter will have first been converted to 0 to 1 already, so just need to further make it A to B
            % y=A+(B-A)*x, converts 0-to-1 x, into A-to-B y
        end
   
        % Now store the unconstrained values
        if estimomitparams_counter(pp)>0
            currparamraw=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
            currparamraw(isnan(currparamraw))=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
            EstimParams.(EstimParamNames{pp})=currparamraw;
        else
            EstimParams.(EstimParamNames{pp})=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
        end
    else
        EstimParams.(EstimParamNames{pp})=Parameters.(EstimParamNames{pp}); % When skipping estimation, just returns the same parameters as you input
    end
    
    if estimoptions.bootstrapStdErrors==0
        % Note: J and Sigma where calculated on 'external' parameters
        estimparamscovarmatrix_diag=diag(estimparamscovarmatrix); % Just the diagonal of the covar matrix of the parameter vector
        estsummary.EstimParamsStdDev.(EstimParamNames{pp})=sqrt(estimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        % if estimoptions.constrainpositive(pp)==0 % THIS IS NO LONGER NEEDED AS J and Sigma TO CALCULATE FROM EXTERNAL PARAMETERS
        %     estsummary.EstimParamsStdDev.(EstimParamNames{pp})=sqrt(estimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        % elseif estimoptions.constrainpositive(pp)==1
        %     % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        %     estsummary.EstimParamsStdDev.(EstimParamNames{pp})=exp(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))+estimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))-exp(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        % end
        % If bootstrap std errors, then replace the std dev with the bootstrap distribuiton
    elseif estimoptions.bootstrapStdErrors==1
        estsummary.EstimParamsStdDev=EstimParamsBootStrapDist;
        estsummary.notes.bootstrap=['Standard errors report distribution of parameter estimates based on ',num2str(estimoptions.numbootstrapsims),' bootstraps, each had ',num2str(estimoptions.numberinvidualsperbootstrapsim),' agents for ',num2str(N_j),' periods (so some ',num2str(N_j*estimoptions.numberinvidualsperbootstrapsim),' observations)' ];
    end
end
clear estimparamsvec % I modified it, so want to make sure I don't accidently use it again later

if estimoptions.confidenceintervals==68
    criticalvalue_normaldist_z_alphadiv2=1;
elseif estimoptions.confidenceintervals==80
    criticalvalue_normaldist_z_alphadiv2=1.282;
elseif estimoptions.confidenceintervals==85
    criticalvalue_normaldist_z_alphadiv2=1.440;
elseif estimoptions.confidenceintervals==90
    criticalvalue_normaldist_z_alphadiv2=1.645;
elseif estimoptions.confidenceintervals==95
    criticalvalue_normaldist_z_alphadiv2=1.96;
elseif estimoptions.confidenceintervals==98
    criticalvalue_normaldist_z_alphadiv2=2.33;
elseif estimoptions.confidenceintervals==99
    criticalvalue_normaldist_z_alphadiv2=2.575;
else
    error('Currently only 68, 80, 85, 90, 95, 98 and 99 are possible values for estimoptions.confidenceintervals (default is 90=')
end

% By executive decision, I decided that confidence intervals are the 'main'
% output, rather than the standard deviations of the estimated parameters.
% This avoids people focusing on statistical significance and the 'star wars'.
% Instead they will hopefully focus on what is likely and plausible.
EstimParamsConfInts.notes='These are 90-percent confidence intervals';
for pp=1:length(EstimParamNames)
    EstimParamsConfInts.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp}) + [-1,1]*criticalvalue_normaldist_z_alphadiv2*estsummary.EstimParamsStdDev.(EstimParamNames{pp});
end

% Give lots of alternative confidence intervals in the estsummary
confintvec=[68,80,85,90,95,98,99];
criticalvalue_normaldist_z_alphadiv2_vec=[1,1.282,1.440,1.645, 1.96, 2.33, 2.575];
for ii=1:length(confintvec)
    confint=confintvec(ii);
    critval=criticalvalue_normaldist_z_alphadiv2_vec(ii);
    for pp=1:length(EstimParamNames)
        estsummary.confidenceintervals.(['confint',num2str(confint)]).EstimParamsConfInts.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp}) + [-1,1]*critval*estsummary.EstimParamsStdDev.(EstimParamNames{pp});
    end
end


%% Give various outputs
estsummary.variousmatrices.W=WeightingMatrix; % This is just a duplicate of the input, but I figure it is handy to keep in same place as the rest of estimation results

estsummary.objectivefnval=fval;
estsummary.notes.objectivefnval='The objective function value is the value of (M_d-M_m)W(M_d-M_m).';
if estimoptions.skipestimation==1
    estsummary.warningskipestimation='Warning: this estimation used estimoptions.skipestimation=1 (all good, just reminding you as you need to be careful when using skipestimation=1 :)';
end

if estimoptions.bootstrapStdErrors==0 % Depends on derivatives, so cannot do when bootstapping the standard errors
    estsummary.variousmatrices.J=J;
    if estimoptions.efficientW==0
        estsummary.variousmatrices.Omega=CoVarMatrixDataMoments; % Covariance matrix of the data moments
    elseif estimoptions.efficientW==1
        estsummary.variousmatrices.Omega=[]; % Two-iteration efficient GMM does not use the covariance matrix of data moments
        estsummary.notes.iterateGMM='When using iterated GMM we do not use the covariance matrix of the data moments (Omega) and hence it is empty [the model implied one will be W^(-1)]';
    end
    estsummary.variousmatrices.Sigma=estimparamscovarmatrix; % Asymptotic covariance matrix of estimated parameters
    estsummary.notes.variousmatrices='J is the jacobian of derivatives of model moments with respect to the parameter vector, W is the weighting matix (just duplicates what was input), Omega is the covariance matrix of data moments (just duplicates what was input).';
    estsummary.notes.sensitivitymatrix='Sensitivity matrix of Andrews, Gentzkow & Shapiro (2017), which they denote Lambda. Measures the change in parameter (rows index parameters) given a change in moments (columns index moments).';
    if isfield(estimoptions,'CalibParamsNames')
        estsummary.notes.sensitivitytocalibrationmatrix='Sensitivity matrix to calibrated parameters, of Jorgensen (2023) - Sensitivity to Calibrated Parameters (his Corrollary 1, =Lambda*Jcalib). Measures the change in estimated parameters (rows index estim params) given change in calibrated parameters (columns index calib params)';
    end
    estsummary.doublechecks.note='Double-checks are various outputs relating to computing J by finite differences, including using different epsilon in df/dx=f(x+epsilon)/epsilon.';
end

if estimoptions.previousiterations.niters>0  % If there were any previous iterations (using estimoptions.iterateGMM) then get that output (I hid them in estimoptions)
    for ii=1:estimoptions.previousiterations.niters
        estsummary.iterateGMM.(['iteration',num2str(ii)]).estimparams=estimoptions.(['storeiter',num2str(ii)]).estimparams;
        estsummary.iterateGMM.(['iteration',num2str(ii)]).CoVarMatrixSimMoments=estimoptions.(['storeiter',num2str(ii)]).CoVarMatrixSimMoments;
    end
end








end
