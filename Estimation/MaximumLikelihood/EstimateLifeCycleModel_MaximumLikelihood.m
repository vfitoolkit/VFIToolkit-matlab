function [EstimParams, EstimParamsConfInts, estsummary]=EstimateLifeCycleModel_MaximumLikelihood(EstimParamNames,Data,MeasurementError,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions)
% Maximum likelihood estimation of a life-cycle (finite horizon) model from panel data.
%
% Maximizes the log-likelihood of the data, log L(theta)=sum_i log L_i(theta), over the parameters named in EstimParamNames. The
% likelihood is evaluated exactly on the model grid by LogLikelihood_FHorz (the model as a hidden Markov model, the forward
% algorithm; see there for Data and MeasurementError). Each evaluation solves the model (ValueFnIter_Case1_FHorz) at the current
% parameters and then evaluates the likelihood. No simulation is involved.
%
% The parameters of the measurement error (MeasurementError.(name).sigma etc. are names of parameters in Parameters) are estimated
% by including them in EstimParamNames, or held fixed by leaving them out.
%
% Inputs: as for EstimateLifeCycleModel_MethodOfMoments, with Data and MeasurementError in place of the target moments and weighting
% matrices (and no AgeWeightParamNames, which the likelihood does not use).
%
% estimoptions (all optional)
%   .fminalgo: 4 (default) CMA-ES; 1 fminsearch; 6 fmincon (needs estimoptions.lb and .ub). The likelihood is a step function of the
%              parameters (the policy is chosen on a grid; the grid interpolation layer makes the steps much smaller but does not
%              remove them), which is why the default is a derivative-free global method.
%   .stderrors: which asymptotic covariance gives EstimParamsConfInts and estsummary.EstimParamsStdDev: 'OPG' (default; outer
%               product of the individual scores, BHHH), 'sandwich' (H^(-1)*OPG*H^(-1), robust to misspecification), or 'Hessian'
%               (inverse of minus the Hessian of log L). All three are always reported in estsummary.
%   .confidenceintervals: 68, 80, 85, 90 (default), 95, 98 or 99
%   .eedefault: 'auto' (default), 1, 2, 3 or 4: the finite-difference step is sqrt(2.2)*10^(-8)*[1,10^2,10^4,10^6](eedefault)
%               relative to the parameter (the scores are reported for all four steps in estsummary.doublechecks). 'auto' uses 3
%               (about 1.5e-4) for each parameter whose up and down one-sided scores agree there; for the rest, a step estimated
%               from 3 and 4 (between them, two more evaluations each) or else 4 (about 1.5e-2): the likelihood is a step function
%               of the parameters, and a step inside its jumps gives too-small standard errors (see estsummary.stepchoice)
%   .constrainpositive, .constrain0to1, .constrainAtoB (+.constrainAtoBlimits), .constrainpositivemethod: as for method of moments
%   .omitestimparam: as for method of moments (estimate only some entries of a vector-valued parameter)
%   .skipestimation: =1 takes the values in Parameters as the estimate and just computes the standard errors etc.
%   .initialcondition, .nindividualsbatch: passed to LogLikelihood_FHorz
%   .refinecheck_ngridinterp: a vector of values of ngridinterp; the log-likelihood at the estimate is recomputed with each (a check
%               of how much the grid matters), reported in estsummary.refinecheck. Needs the grid interpolation layer.
%   .verbose: =1 (default) prints the parameters and log-likelihood at each evaluation
%   .toleranceparams, .toleranceobjective: for fminsearch and fmincon
%   .insigma, .inopts: for CMA-ES
%   .calibrateshocks, .ParametrizeParamsFn_InputNames, .ParametrizeParamsFn_OutputNames: as for method of moments
%
% Outputs
%   EstimParams: the estimated parameters
%   EstimParamsConfInts: confidence intervals, [lower, upper], one row per entry of each parameter
%   estsummary: LogL and LogLi at the estimate, AIC and BIC, the covariance matrices (OPG, Hessian, sandwich), the scores, the
%               measurement error diagnostics, local identification, and double-checks of the finite differences

%% Setup estimoptions
if ~isfield(estimoptions,'verbose')
    estimoptions.verbose=1;
end
if ~isfield(estimoptions,'constrainpositive')
    estimoptions.constrainpositive={};
end
if ~isfield(estimoptions,'constrainpositivemethod')
    estimoptions.constrainpositivemethod='softplus';
end
if ~isfield(estimoptions,'constrain0to1')
    estimoptions.constrain0to1={};
end
if ~isfield(estimoptions,'constrainAtoB')
    estimoptions.constrainAtoB={};
else
    if ~isfield(estimoptions,'constrainAtoBlimits')
        error('You have used estimoptions.constrainAtoB, but are missing estimoptions.constrainAtoBlimits')
    end
end
if ~isfield(estimoptions,'confidenceintervals')
    estimoptions.confidenceintervals=90;
end
if ~isfield(estimoptions,'eedefault')
    estimoptions.eedefault='auto';
end
if ~isfield(estimoptions,'toleranceparams')
    estimoptions.toleranceparams=10^(-4);
end
if ~isfield(estimoptions,'toleranceobjective')
    estimoptions.toleranceobjective=10^(-6);
end
if ~isfield(estimoptions,'fminalgo')
    estimoptions.fminalgo=4; % CMA-ES
end
if ~any(estimoptions.fminalgo==[1,4,6])
    error('EstimateLifeCycleModel_MaximumLikelihood: estimoptions.fminalgo must be 4 (CMA-ES, default), 1 (fminsearch) or 6 (fmincon)')
end
if ~isfield(estimoptions,'stderrors')
    estimoptions.stderrors='OPG';
end
if ~any(strcmp(estimoptions.stderrors,{'OPG','sandwich','Hessian'}))
    error('EstimateLifeCycleModel_MaximumLikelihood: estimoptions.stderrors must be ''OPG'', ''sandwich'' or ''Hessian''')
end
if ~isfield(estimoptions,'skipestimation')
    estimoptions.skipestimation=0;
end
if ~isfield(estimoptions,'refinecheck_ngridinterp')
    estimoptions.refinecheck_ngridinterp=[];
end
if ~isfield(estimoptions,'ParametrizeParamsFn_InputNames')
    estimoptions.ParametrizeParamsFn_InputNames={};
end
if ~isfield(estimoptions,'ParametrizeParamsFn_OutputNames')
    estimoptions.ParametrizeParamsFn_OutputNames={};
end
if ~isfield(estimoptions,'warnARgrid')
    estimoptions.warnARgrid=1; % =1 warn (once, at the estimate) when the iid s.d. of a measurement error with an AR(1) component is smaller than the spacing of its AR(1) grid
end
warnARgrid=estimoptions.warnARgrid;
% Following are estimoptions used internally
estimoptions.warnARgrid=0; % LogLikelihood_FHorz does not warn at every evaluation; the check is done once at the estimate (below)
estimoptions.vectoroutput=0; % =0: the objective is minus the log-likelihood (for the optimizer); =1: the vector of individual log-likelihoods LogLi (for the scores)

%% The grid interpolation layer
if ~isfield(vfoptions,'gridinterplayer')
    vfoptions.gridinterplayer=0;
end
if ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0;
end
if vfoptions.gridinterplayer~=simoptions.gridinterplayer
    error('EstimateLifeCycleModel_MaximumLikelihood: vfoptions.gridinterplayer and simoptions.gridinterplayer must be the same')
end
if vfoptions.gridinterplayer==1
    if ~isfield(vfoptions,'ngridinterp') || ~isfield(simoptions,'ngridinterp') || vfoptions.ngridinterp~=simoptions.ngridinterp
        error('EstimateLifeCycleModel_MaximumLikelihood: vfoptions.ngridinterp and simoptions.ngridinterp must both be set, and be the same')
    end
else
    warning('EstimateLifeCycleModel_MaximumLikelihood: without the grid interpolation layer (vfoptions.gridinterplayer=1) the likelihood is a step function of the parameters with large steps; the estimate and especially the finite-difference standard errors can be unreliable')
end
if estimoptions.fminalgo==6
    warning('EstimateLifeCycleModel_MaximumLikelihood: fmincon (estimoptions.fminalgo=6) uses derivatives, but the likelihood is a step function of the parameters (the policy is chosen on a grid); a derivative-free method (fminalgo=4, CMA-ES, the default) is more reliable')
end
if ~isempty(estimoptions.refinecheck_ngridinterp) && vfoptions.gridinterplayer==0
    error('EstimateLifeCycleModel_MaximumLikelihood: estimoptions.refinecheck_ngridinterp needs the grid interpolation layer')
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
estimomitparams_counter=zeros(length(EstimParamNames),1); % column vector: estimomitparamsvec allows omitting the parameter for certain ages
estimomitparamsmatrix=zeros(N_j,1); % Each row is of size N_j-by-1 and holds the omitted values of a parameter
for pp=1:length(EstimParamNames)
    if any(strcmp(OmitEstimParamsNames,EstimParamNames{pp}))
        % This parameter is under an omit-mask, so need to only use part of it
        tempparam=Parameters.(EstimParamNames{pp});
        tempomitparam=estimoptions.omitestimparam.(EstimParamNames{pp});
        % Make them both column vectors
        tempparam=tempparam(:);
        tempomitparam=tempomitparam(:);
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


%% Now, a bunch of things to avoid redoing them every parameter vector we want to try
ReturnFnParamNames=[];

% Do the shocks depend on the estimated parameters (directly, or via ParametrizeParamsFn)? As for method of moments.
effectiveEstimParamNames=EstimParamNames;
parametrizefailed=0;
if ~isempty(ParametrizeParamsFn)
    if isempty(estimoptions.ParametrizeParamsFn_InputNames) || isempty(estimoptions.ParametrizeParamsFn_OutputNames)
        [parsedInputNames,parsedOutputNames,parametrizefailed]=ParseParametrizeParamsFn(ParametrizeParamsFn,Parameters);
        if parametrizefailed==0
            if isempty(estimoptions.ParametrizeParamsFn_InputNames)
                estimoptions.ParametrizeParamsFn_InputNames=parsedInputNames;
            end
            if isempty(estimoptions.ParametrizeParamsFn_OutputNames) % user-given names are never overwritten
                estimoptions.ParametrizeParamsFn_OutputNames=parsedOutputNames;
            end
        end
    end
    effectiveEstimParamNames=union(EstimParamNames,estimoptions.ParametrizeParamsFn_OutputNames);
end
usersetcalibrateshocks=isfield(estimoptions,'calibrateshocks');
autocalibrateshocks=0;
if isfield(vfoptions,'ExogShockFn')
    temp=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    if ~isempty(intersect(temp,effectiveEstimParamNames)) || parametrizefailed==1
        autocalibrateshocks=1;
    end
end
if isfield(vfoptions,'EiidShockFn')
    temp=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    if ~isempty(intersect(temp,effectiveEstimParamNames)) || parametrizefailed==1
        autocalibrateshocks=1;
    end
end
if usersetcalibrateshocks==1
    if estimoptions.calibrateshocks==0 && autocalibrateshocks==1
        warning('You set estimoptions.calibrateshocks=0, but the shock processes appear to depend on the estimated parameters (directly, or via ParametrizeParamsFn): trusting your setting, the shocks will NOT be rebuilt during the estimation')
    end
else
    estimoptions.calibrateshocks=autocalibrateshocks;
end
if ~isfield(simoptions,'jequaloneDist_usergrids')
    simoptions.jequaloneDist_usergrids=1;
end
if estimoptions.calibrateshocks==0
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3,simoptions.jequaloneDist_usergrids);
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
    if simoptions.jequaloneDist_usergrids==1
        simoptions.user_z_grid=vfoptions.user_z_grid;
        simoptions.user_pi_z=vfoptions.user_pi_z;
    end
else
    % The shock grids depend on a parameter being estimated, so they are rebuilt inside the objective function every evaluation
    z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end
vfoptions.alreadygridvals=1;
simoptions.alreadygridvals=1;


%% Set up the objective function
EstimateMLObjectiveFn=@(estimparamsvec) EstimateLifeCycleModel_MaximumLikelihood_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist, ParametrizeParamsFn, FnsToEvaluate, Data, MeasurementError, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);
EstimateMLObjectiveFn_Jac=@(estimparamsvec,estimoptions_temp,vfoptions_temp,simoptions_temp) EstimateLifeCycleModel_MaximumLikelihood_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist, ParametrizeParamsFn, FnsToEvaluate, Data, MeasurementError, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptions_temp, vfoptions_temp,simoptions_temp);


%% Maximize the log-likelihood (minimize minus the log-likelihood)
if estimoptions.skipestimation==0
    minoptions = optimset('TolX',estimoptions.toleranceparams,'TolFun',estimoptions.toleranceobjective);
    if estimoptions.fminalgo==1
        estimparamsvec=fminsearch(EstimateMLObjectiveFn,estimparamsvec0,minoptions);
    elseif estimoptions.fminalgo==4 % CMA-ES algorithm (Covariance-Matrix adaptation - Evolutionary Stategy)
        if ~isfield(estimoptions,'insigma')
            estimoptions.insigma=0.3*abs(estimparamsvec0)+0.1*(estimparamsvec0==0); % 30% of the initial parameter value (0.1 for any zeros)
        end
        if ~isfield(estimoptions,'inopts')
            estimoptions.inopts=[];
        end
        if estimoptions.verbose==1
            disp('VFI Toolkit is using the CMA-ES algorithm, consider giving a cite to: Hansen, N. and S. Kern (2004). Evaluating the CMA Evolution Strategy on Multimodal Test Functions' )
        end
        [estimparamsvec,~,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(EstimateMLObjectiveFn,estimparamsvec0,estimoptions.insigma,estimoptions.inopts);
        estsummary.cmaesoutputs.counteval=counteval;
        estsummary.cmaesoutputs.stopflag=stopflag;
        estsummary.cmaesoutputs.out=out;
        estsummary.cmaesoutputs.bestever=bestever;
    elseif estimoptions.fminalgo==6
        if ~isfield(estimoptions,'lb') || ~isfield(estimoptions,'ub')
            error('When using constrained optimization (estimoptions.fminalgo=6) you must set the lower and upper bounds of the estimated parameters using estimoptions.lb and estimoptions.ub')
        end
        estimparamsvec=fmincon(EstimateMLObjectiveFn,estimparamsvec0,[],[],[],[],estimoptions.lb,estimoptions.ub,[],minoptions);
    end
else % estimoptions.skipestimation==1
    warning('Skipping the estimation step (you have set estimoptions.skipestimation=1 in EstimateLifeCycleModel_MaximumLikelihood() [Nothing wrong with this, just warning as want to be sure you did this on purpose]')
    estimparamsvec=estimparamsvec0;
end

% estimparamsvec contains the (transformed) unconstrained parameters, switch back to the original (constrained) parameter values
[estimparamsvec,~]=ParameterConstraints_TransformParamsToOriginal(estimparamsvec,estimparamsvecindex,EstimParamNames,estimoptions);


%% The scores (derivatives of each individual's log-likelihood with respect to the parameters), by finite differences
% As for the Jacobian in method of moments: the derivatives are taken in the model parameters (not the internal unconstrained ones), so
% the constraints are switched off in estimoptionsJacobian, and a step that would leave a constraint is replaced by a one-sided difference.
estimoptionsJacobian=estimoptions;
estimoptionsJacobian.constrainpositive=zeros(length(EstimParamNames),1);
estimoptionsJacobian.constrain0to1=zeros(length(EstimParamNames),1);
estimoptionsJacobian.constrainAtoB=zeros(length(EstimParamNames),1);
estimoptionsJacobian.vectoroutput=1; % the objective returns LogLi, the N-by-1 vector of individual log-likelihoods
estimoptionsJacobian.verbose=0;

epsilonraw=sqrt(2.2)*10^(-8); % sqrt of double precision
epsilonmodvec=[1,10^2,10^4,10^6];
epsilonalt=[10^(-2),10^(-2),10^(-1),10^(-1)]; % for parameters of size 10^(-2) to 10^(-4)
nparams=length(estimparamsvec);

epsilonparamup=zeros(nparams,length(epsilonmodvec));
epsilonparamdown=zeros(nparams,length(epsilonmodvec));
modelestimparamsvec=estimparamsvec;
modelestimparamsvecup=zeros(size(modelestimparamsvec));
modelestimparamsvecdown=zeros(size(modelestimparamsvec));
violateconstrainttop=zeros(size(modelestimparamsvec)); % =1 means use a one-sided (down) finite-difference
violateconstraintbottom=zeros(size(modelestimparamsvec)); % =1 means use a one-sided (up) finite-difference
for ee=1:length(epsilonmodvec)
    epsilon=epsilonmodvec(ee)*epsilonraw;
    for pp=1:nparams
        pname=find(estimparamsvecindex(2:end)>=pp,1,'first'); % the parameter this element belongs to (the constraints are by parameter)
        if floor(log(abs(modelestimparamsvec(pp)))/log(10))>-2 % order of magnitude is greater than 10^(-2)
            modelestimparamsvecup(pp)=(1+epsilon)*modelestimparamsvec(pp);
            modelestimparamsvecdown(pp)=(1-epsilon)*modelestimparamsvec(pp);
        elseif floor(log(abs(modelestimparamsvec(pp)))/log(10))<-4 % so small (or zero) that epsilon is added/subtracted outright
            modelestimparamsvecup(pp)=epsilon+modelestimparamsvec(pp);
            modelestimparamsvecdown(pp)=-epsilon+modelestimparamsvec(pp);
        else
            modelestimparamsvecup(pp)=(1+epsilonalt(ee))*modelestimparamsvec(pp);
            modelestimparamsvecdown(pp)=(1-epsilonalt(ee))*modelestimparamsvec(pp);
        end
        % Enforce that we do not violate the constraints
        if estimoptions.constrainpositive(pname)==1
            if modelestimparamsvecdown(pp)<=0
                violateconstraintbottom(pp)=1;
            end
        elseif estimoptions.constrainAtoB(pname)==1
            if modelestimparamsvecdown(pp)<=estimoptions.constrainAtoBlimits(pname,1)
                violateconstraintbottom(pp)=1;
            elseif modelestimparamsvecup(pp)>=estimoptions.constrainAtoBlimits(pname,2)
                violateconstrainttop(pp)=1;
            end
        elseif estimoptions.constrain0to1(pname)==1
            if modelestimparamsvecdown(pp)<=0
                violateconstraintbottom(pp)=1;
            elseif modelestimparamsvecup(pp)>=1
                violateconstrainttop(pp)=1;
            end
        end
    end
    epsilonparamup(:,ee)=modelestimparamsvecup;
    epsilonparamdown(:,ee)=modelestimparamsvecdown;
end

LogLi0=gather(EstimateMLObjectiveFn_Jac(modelestimparamsvec,estimoptionsJacobian,vfoptions,simoptions)); % N-by-1, at the estimate
N=length(LogLi0);
onesideddisagreement=zeros(nparams,length(epsilonmodvec));
Scoresall=zeros(N,nparams,length(epsilonmodvec));
LogLi_upall=zeros(N,nparams,length(epsilonmodvec));
LogLi_downall=zeros(N,nparams,length(epsilonmodvec));
for ee=1:length(epsilonmodvec)
    LogLi_up=zeros(N,nparams);
    LogLi_down=zeros(N,nparams);
    for pp=1:nparams
        epsilonparamvec=modelestimparamsvec;
        if violateconstrainttop(pp)==0
            epsilonparamvec(pp)=epsilonparamup(pp,ee);
            LogLi_up(:,pp)=gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions));
        end
        epsilonparamvec=modelestimparamsvec;
        if violateconstraintbottom(pp)==0
            epsilonparamvec(pp)=epsilonparamdown(pp,ee);
            LogLi_down(:,pp)=gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions));
        end
    end
    S_up=(LogLi_up-LogLi0)./((epsilonparamup(:,ee)-modelestimparamsvec)');
    S_down=(LogLi0-LogLi_down)./((modelestimparamsvec-epsilonparamdown(:,ee))');
    S_centered=(LogLi_up-LogLi_down)./((epsilonparamup(:,ee)-epsilonparamdown(:,ee))');
    S=S_centered; % N-by-nparams: the score of each individual
    for pp=1:nparams
        if violateconstraintbottom(pp)==1
            S(:,pp)=S_up(:,pp);
        elseif violateconstrainttop(pp)==1
            S(:,pp)=S_down(:,pp);
        end
    end
    tempname=['epsilon',num2str(epsilonmodvec(ee))];
    estsummary.doublechecks.(tempname).scores=S;
    estsummary.doublechecks.(tempname).OPG=S'*S;
    estsummary.doublechecks.(tempname).Sigma_OPG=(S'*S)^(-1);
    estsummary.doublechecks.(tempname).LogLi_up=LogLi_up;
    estsummary.doublechecks.(tempname).LogLi_down=LogLi_down;
    estsummary.doublechecks.(tempname).estimparamsvec=modelestimparamsvec;
    estsummary.doublechecks.(tempname).estimparamsvecup=epsilonparamup(:,ee);
    estsummary.doublechecks.(tempname).estimparamsvecdown=epsilonparamdown(:,ee);
    estsummary.doublechecks.(tempname).violateconstraintbottom=violateconstraintbottom;
    estsummary.doublechecks.(tempname).violateconstrainttop=violateconstrainttop;
    % One-sided disagreement of each parameter's scores, ||S_up-S_down||/||S_up+S_down|| over the individuals (NaN when one-sided)
    onesideddisagreement(:,ee)=sqrt(sum((S_up-S_down).^2,1)./sum((S_up+S_down).^2,1))';
    onesideddisagreement(violateconstraintbottom==1 | violateconstrainttop==1,ee)=NaN;
    Scoresall(:,:,ee)=S;
    LogLi_upall(:,:,ee)=LogLi_up;
    LogLi_downall(:,:,ee)=LogLi_down;
end
estsummary.doublechecks.LogLi=LogLi0;
estsummary.doublechecks.note='Double-checks are the scores (by finite differences) for four sizes of the step epsilon, with the parameter vectors at which the individual log-likelihoods were evaluated.';

%% Which step each parameter uses
% The likelihood is a step function of the parameters (the policy is on a grid). A step that is small relative to the jumps gives
% scores that are zero for most individuals and a jump divided by the step for a few, so the OPG is inflated (roughly like 1/step)
% and the standard errors are too small (or Inf when no individual has a jump inside the step). Where the step is wide enough the up
% and down one-sided scores agree, while inside the jumps they do not (the jumps either side are different events). So
% eedefault='auto' uses epsilon10000 (eedefault=3) for each parameter whose one-sided disagreement there is at most 0.1, and
% epsilon1000000 (eedefault=4) otherwise. (With independent jumps either side, a disagreement r inflates the OPG by 1/(1-r^2), so
% 0.1 is about 1 percent.) A parameter at a constraint (one-sided differences) compares its scores at this step and the next instead.
for ee=1:length(epsilonmodvec)-1
    for pp=1:nparams
        if violateconstraintbottom(pp)==1 || violateconstrainttop(pp)==1
            onesideddisagreement(pp,ee)=sqrt(sum((Scoresall(:,pp,ee)-Scoresall(:,pp,ee+1)).^2)/sum((Scoresall(:,pp,ee)+Scoresall(:,pp,ee+1)).^2));
        end
    end
end
% A parameter that fails at epsilon10000 gets an estimated step, hstar, in place of epsilon1000000 where that is smaller. Over the
% individuals, the mean squared difference of the up and down one-sided scores is about D(h)^2=a*h^2+b/h: a*h^2 from the curvature
% (S_up-S_down is about h times the second derivative) and b/h from the jumps (a jump inside the step, divided by the step, with a
% chance proportional to the step). a and b solve this at epsilon10000 and epsilon1000000. (Not the smaller steps: the jumps are not
% one process across scales, e.g. the many small jumps of the grid interpolation layer and the few large ones of the coarse grid, so
% at a much smaller step D^2 is far below the b/h seen between epsilon10000 and epsilon1000000, and drags b down.) The jumps add about b/(4h) to the mean
% squared centered score, so they inflate the OPG by 1 percent (the 0.1 above) at hstar=25*b/V, V the mean squared centered score
% at epsilon1000000 less its jump part. hstar is used when it lies between the two steps, the jumps at epsilon10000 are not so sparse
% that b is noise (the share of individuals whose up and down scores are both exactly zero is at most 0.5 above that at
% epsilon1000000), and the scores at hstar (two more evaluations of the likelihood) have D^2 at most twice the fit; otherwise
% epsilon1000000. eechosen=5 means hstar.
hstar=NaN(nparams,1);
hstarfit=NaN(nparams,5); % [a, b, V, D^2 predicted at hstar, D^2 measured at hstar]
hstarfitsteps=false(nparams,length(epsilonmodvec)); % the steps the fit used
hstarLogLi_up=NaN(N,nparams);
hstarLogLi_down=NaN(N,nparams);
epsilonparamup(:,length(epsilonmodvec)+1)=NaN;
epsilonparamdown(:,length(epsilonmodvec)+1)=NaN;
if strcmp(estimoptions.eedefault,'auto')
    eechosen=3+(~(onesideddisagreement(:,3)<=0.1)); % NaN (every score zero at epsilon10000) also gives 4
    for pp=1:nparams
        if eechosen(pp)==4 && violateconstraintbottom(pp)==0 && violateconstrainttop(pp)==0 && abs(epsilonparamup(pp,4)-modelestimparamsvec(pp))>abs(epsilonparamup(pp,3)-modelestimparamsvec(pp)) % (for a parameter of size 1e-4 to 1e-2 the two steps are the same)
            hh=zeros(length(epsilonmodvec),1);
            D2=zeros(length(epsilonmodvec),1);
            zeroshare=zeros(length(epsilonmodvec),1);
            for ee=1:length(epsilonmodvec)
                hh(ee)=abs(epsilonparamup(pp,ee)-modelestimparamsvec(pp));
                temp_up=(LogLi_upall(:,pp,ee)-LogLi0)/(epsilonparamup(pp,ee)-modelestimparamsvec(pp));
                temp_down=(LogLi0-LogLi_downall(:,pp,ee))/(modelestimparamsvec(pp)-epsilonparamdown(pp,ee));
                D2(ee)=mean((temp_up-temp_down).^2);
                zeroshare(ee)=mean(temp_up==0 & temp_down==0);
            end
            % the fit uses epsilon10000 and epsilon1000000 (relative to each step's own D^2)
            hstarfitsteps(pp,3:4)=(D2(3:4)'>0);
            if sum(hstarfitsteps(pp,:))>=2
                ab=([hh(hstarfitsteps(pp,:)).^2, 1./hh(hstarfitsteps(pp,:))]./D2(hstarfitsteps(pp,:)))\ones(sum(hstarfitsteps(pp,:)),1);
            else
                ab=[0;0]; % (no fit, so no hstar)
            end
            h3=hh(3);
            h4=hh(4);
            a=max(ab(1),0);
            b=max(ab(2),0);
            V=mean(Scoresall(:,pp,4).^2)-b/(4*h4);
            hstarfit(pp,1:3)=[a,b,V];
            if V>0 && 25*b/V>h3 && 25*b/V<h4 && zeroshare(3)-zeroshare(4)<=0.5
                hsign=sign(epsilonparamup(pp,4)-modelestimparamsvec(pp));
                epsilonparamvec=modelestimparamsvec;
                epsilonparamvec(pp)=modelestimparamsvec(pp)+hsign*25*b/V;
                hstarLogLi_up(:,pp)=gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions));
                epsilonparamup(pp,length(epsilonmodvec)+1)=epsilonparamvec(pp);
                epsilonparamvec(pp)=modelestimparamsvec(pp)-hsign*25*b/V;
                hstarLogLi_down(:,pp)=gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions));
                epsilonparamdown(pp,length(epsilonmodvec)+1)=epsilonparamvec(pp);
                temp_up=(hstarLogLi_up(:,pp)-LogLi0)/(epsilonparamup(pp,length(epsilonmodvec)+1)-modelestimparamsvec(pp));
                temp_down=(LogLi0-hstarLogLi_down(:,pp))/(modelestimparamsvec(pp)-epsilonparamdown(pp,length(epsilonmodvec)+1));
                hstarfit(pp,4:5)=[a*(25*b/V)^2+b/(25*b/V), mean((temp_up-temp_down).^2)];
                if hstarfit(pp,5)<=2*hstarfit(pp,4)
                    hstar(pp)=25*b/V;
                    eechosen(pp)=length(epsilonmodvec)+1;
                    Scoresall(:,pp,length(epsilonmodvec)+1)=(hstarLogLi_up(:,pp)-hstarLogLi_down(:,pp))/(epsilonparamup(pp,length(epsilonmodvec)+1)-epsilonparamdown(pp,length(epsilonmodvec)+1));
                    LogLi_upall(:,pp,length(epsilonmodvec)+1)=hstarLogLi_up(:,pp);
                    LogLi_downall(:,pp,length(epsilonmodvec)+1)=hstarLogLi_down(:,pp);
                end
            end
        end
    end
else
    eechosen=estimoptions.eedefault*ones(nparams,1);
end
estsummary.doublechecks.hstar.LogLi_up=hstarLogLi_up; % NaN for a parameter at which hstar was not tried
estsummary.doublechecks.hstar.LogLi_down=hstarLogLi_down;
estsummary.doublechecks.hstar.estimparamsvecup=epsilonparamup(:,length(epsilonmodvec)+1);
estsummary.doublechecks.hstar.estimparamsvecdown=epsilonparamdown(:,length(epsilonmodvec)+1);
Scores=zeros(N,nparams);
LogLi_updefault=zeros(N,nparams);
LogLi_downdefault=zeros(N,nparams);
paramup=zeros(nparams,1);
paramdown=zeros(nparams,1);
for pp=1:nparams
    Scores(:,pp)=Scoresall(:,pp,eechosen(pp));
    LogLi_updefault(:,pp)=LogLi_upall(:,pp,eechosen(pp));
    LogLi_downdefault(:,pp)=LogLi_downall(:,pp,eechosen(pp));
    paramup(pp)=epsilonparamup(pp,eechosen(pp));
    paramdown(pp)=epsilonparamdown(pp,eechosen(pp));
end
stderr_OPG=zeros(nparams,length(epsilonmodvec));
for ee=1:length(epsilonmodvec)
    stderr_OPG(:,ee)=sqrt(diag(estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_OPG));
end
estsummary.stepchoice.eechosen=eechosen;
estsummary.stepchoice.onesideddisagreement=onesideddisagreement;
estsummary.stepchoice.stderr_OPG=stderr_OPG;
estsummary.stepchoice.hstar=hstar;
estsummary.stepchoice.hstarfit=hstarfit;
estsummary.stepchoice.hstarfitsteps=hstarfitsteps;
estsummary.notes.stepchoice=['eechosen is the step each parameter''s scores and Hessian use (1-4, as eedefault: epsilon1, epsilon100, epsilon10000, epsilon1000000 in doublechecks; 5, the estimated step hstar, see below). ' ...
    'onesideddisagreement (nparams-by-4) is ||S_up-S_down||/||S_up+S_down|| over the individuals at each step (for a parameter at a constraint, its scores at this step vs the next): ' ...
    'near zero means the step is wide relative to the jumps of the likelihood (a step function of the parameters, as the policy is on a grid); near one means the scores are mostly jumps and the OPG standard errors are too small. ' ...
    'eedefault=''auto'' uses 3 where this is at most 0.1; otherwise the estimated step hstar (the absolute step at which the jumps inflate the OPG by 1 percent, from fitting D(h)^2=a*h^2+b/h, the mean squared difference of the one-sided scores, at 3 and 4, hstarfitsteps: hstarfit is [a, b, V, D^2 predicted at hstar, D^2 measured there]) when it lies between steps 3 and 4 and its D^2 is at most twice the fit, and 4 otherwise (hstar NaN). stderr_OPG (nparams-by-4) are the OPG standard errors at each step. Only the value at epsilon10000 is a test for jumps: the one-sided differences also differ by about the step times the second derivative, so at epsilon1000000 a large disagreement can be curvature rather than jumps (e.g. a discount factor near one, where the step is a sizeable share of 1-beta). To judge the grid, compare the standard errors across grids (e.g. ngridinterp).'];

%% The Hessian of the log-likelihood (at each parameter's step), by finite differences
% Diagonal: (L(+h_j)-2L+L(-h_j))/h_j^2. Off-diagonal: (L(+h_j+h_k)-L(+h_j)-L(+h_k)+2L-L(-h_j)-L(-h_k)+L(-h_j-h_k))/(2*h_j*h_k).
% Needs both sides of every parameter, so if a step would leave a constraint the Hessian (and the sandwich) is NaN, with a warning.
LogL0=sum(LogLi0);
LogL_up=sum(LogLi_updefault,1)';
LogL_down=sum(LogLi_downdefault,1)';
hup=paramup-modelestimparamsvec;
hdown=modelestimparamsvec-paramdown;
LogL_upup=NaN(nparams,nparams);
LogL_downdown=NaN(nparams,nparams);
if any(violateconstraintbottom) || any(violateconstrainttop)
    warning('EstimateLifeCycleModel_MaximumLikelihood: a finite-difference step would leave a parameter constraint, so the Hessian (and the sandwich standard errors) are not computed')
    Hessian=NaN(nparams,nparams);
else
    Hessian=zeros(nparams,nparams);
    for jj=1:nparams
        Hessian(jj,jj)=(LogL_up(jj)-2*LogL0+LogL_down(jj))/(hup(jj)*hdown(jj));
        for kk=jj+1:nparams
            epsilonparamvec=modelestimparamsvec;
            epsilonparamvec(jj)=paramup(jj);
            epsilonparamvec(kk)=paramup(kk);
            LogL_upup(jj,kk)=sum(gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions)));
            epsilonparamvec=modelestimparamsvec;
            epsilonparamvec(jj)=paramdown(jj);
            epsilonparamvec(kk)=paramdown(kk);
            LogL_downdown(jj,kk)=sum(gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions)));
            Hessian(jj,kk)=(LogL_upup(jj,kk)-LogL_up(jj)-LogL_up(kk)+2*LogL0-LogL_down(jj)-LogL_down(kk)+LogL_downdown(jj,kk))/(2*hup(jj)*hup(kk));
            Hessian(kk,jj)=Hessian(jj,kk);
        end
    end
end
estsummary.doublechecks.hessian.LogL=LogL0;
estsummary.doublechecks.hessian.LogL_up=LogL_up;
estsummary.doublechecks.hessian.LogL_down=LogL_down;
estsummary.doublechecks.hessian.LogL_upup=LogL_upup; % (j,k), j<k: both parameters stepped up
estsummary.doublechecks.hessian.LogL_downdown=LogL_downdown; % (j,k), j<k: both parameters stepped down
estsummary.doublechecks.hessian.hup=hup;
estsummary.doublechecks.hessian.hdown=hdown;

%% Covariance matrices of the estimated parameters
OPG=Scores'*Scores;
Sigma_OPG=OPG^(-1);
Sigma_Hessian=(-Hessian)^(-1);
Sigma_sandwich=Sigma_Hessian*OPG*Sigma_Hessian;
if strcmp(estimoptions.stderrors,'OPG')
    estimparamscovarmatrix=Sigma_OPG;
elseif strcmp(estimoptions.stderrors,'sandwich')
    estimparamscovarmatrix=Sigma_sandwich;
else
    estimparamscovarmatrix=Sigma_Hessian;
end
estsummary.variousmatrices.Scores=Scores;
estsummary.variousmatrices.OPG=OPG;
estsummary.variousmatrices.Hessian=Hessian;
estsummary.variousmatrices.Sigma_OPG=Sigma_OPG;
estsummary.variousmatrices.Sigma_Hessian=Sigma_Hessian;
estsummary.variousmatrices.Sigma_sandwich=Sigma_sandwich;
estsummary.variousmatrices.Sigma=estimparamscovarmatrix;
estsummary.notes.variousmatrices=['Scores is N-by-nparams, the derivative of each individual''s log-likelihood with respect to the parameters. OPG=Scores''*Scores (outer product of gradients). Hessian is that of the log-likelihood. Sigma_OPG=inv(OPG), Sigma_Hessian=inv(-Hessian), Sigma_sandwich=Sigma_Hessian*OPG*Sigma_Hessian. Sigma is the one used for the standard errors and confidence intervals (estimoptions.stderrors=''',estimoptions.stderrors,''').'];
estsummary.gradient=sum(Scores,1)';
estsummary.notes.gradient='The gradient of the log-likelihood at the estimate (the sum of the scores). Should be near zero at an interior maximum; as the likelihood is a step function of the parameters it is only approximately so.';

%% Local identification
if all(isfinite(OPG(:)))
    rankOPGtol=sqrt(eps)*norm(OPG);
    estsummary.localidentification.rankOPG=rank(OPG,rankOPGtol);
    estsummary.localidentification.ranktolerance=rankOPGtol;
else
    warning('EstimateLifeCycleModel_MaximumLikelihood: the scores have non-finite entries, so the local identification check cannot be done')
    estsummary.localidentification.rankOPG=NaN;
    estsummary.localidentification.ranktolerance=NaN;
end
estsummary.localidentification.yesidentified=logical(estsummary.localidentification.rankOPG>=nparams);
estsummary.notes.localidentification='If the OPG matrix (the outer product of the scores, an estimate of the information matrix) is full rank then the model is locally identified [rank(OPG) should equal the number of parameters being estimated]';

%% The log-likelihood at the estimate, information criteria
nobs=0;
ObsNames=fieldnames(Data);
ObsNames=ObsNames(~strcmp(ObsNames,'age') & ~strcmp(ObsNames,'agej0'));
for vv=1:length(ObsNames)
    nobs=nobs+sum(~isnan(Data.(ObsNames{vv})(:)));
end
estsummary.LogL=LogL0;
estsummary.LogLi=LogLi0;
estsummary.nindividuals=N;
estsummary.nobservations=nobs;
estsummary.AIC=2*nparams-2*LogL0;
estsummary.BIC=nparams*log(N)-2*LogL0;
estsummary.notes.informationcriteria='AIC=2*nparams-2*LogL, BIC=nparams*log(N)-2*LogL, with N the number of individuals.';

%% Clean up the first two outputs
estimparamscovarmatrix_diag=diag(estimparamscovarmatrix);
stderrnames={'OPG','Hessian','sandwich'};
stderrmatrices={Sigma_OPG,Sigma_Hessian,Sigma_sandwich};
for pp=1:length(EstimParamNames)
    if estimoptions.skipestimation==0
        if estimomitparams_counter(pp)>0
            currparamraw=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
            currparamraw(isnan(currparamraw))=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
        else
            currparamraw=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
        end
        EstimParams.(EstimParamNames{pp})=currparamraw;
    else
        EstimParams.(EstimParamNames{pp})=Parameters.(EstimParamNames{pp}); % When skipping estimation, just returns the same parameters as you input
    end
    for ss=0:length(stderrnames)
        if ss==0
            tempdiag=estimparamscovarmatrix_diag;
        else
            tempdiag=diag(stderrmatrices{ss});
        end
        sdblock=sqrt(tempdiag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        if estimomitparams_counter(pp)>0
            mask=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
            sdfull=nan(size(mask));
            sdfull(isnan(mask))=sdblock;
        else
            sdfull=sdblock;
        end
        if ss==0
            estsummary.EstimParamsStdDev.(EstimParamNames{pp})=reshape(sdfull,size(EstimParams.(EstimParamNames{pp})));
        else
            estsummary.(['EstimParamsStdDev_',stderrnames{ss}]).(EstimParamNames{pp})=reshape(sdfull,size(EstimParams.(EstimParamNames{pp})));
        end
    end
end
clear estimparamsvec

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
    error('Currently only 68, 80, 85, 90, 95, 98 and 99 are possible values for estimoptions.confidenceintervals (default is 90)')
end
EstimParamsConfInts.notes=['These are ',num2str(estimoptions.confidenceintervals),'-percent confidence intervals: [lower, upper], one row per entry of the parameter (standard errors: ',estimoptions.stderrors,')'];
confintvec=[68,80,85,90,95,98,99];
criticalvalue_normaldist_z_alphadiv2_vec=[1,1.282,1.440,1.645, 1.96, 2.33, 2.575];
for pp=1:length(EstimParamNames)
    v=EstimParams.(EstimParamNames{pp}); v=v(:);
    sd=estsummary.EstimParamsStdDev.(EstimParamNames{pp}); sd=sd(:);
    EstimParamsConfInts.(EstimParamNames{pp})=[v-criticalvalue_normaldist_z_alphadiv2*sd, v+criticalvalue_normaldist_z_alphadiv2*sd];
    for ii=1:length(confintvec)
        estsummary.confidenceintervals.(['confint',num2str(confintvec(ii))]).EstimParamsConfInts.(EstimParamNames{pp})=[v-criticalvalue_normaldist_z_alphadiv2_vec(ii)*sd, v+criticalvalue_normaldist_z_alphadiv2_vec(ii)*sd];
    end
end

%% Measurement error diagnostics
% For each variable observed with normal or t measurement error: the measurement error standard deviation at the estimate, and the
% share of the variance of the data it accounts for (sigma^2 over the variance of the observations; for type='log', of the log
% observations). A large share often signals that the model is misspecified for that variable, rather than large reporting error.
ParamsEst=Parameters;
for pp=1:length(EstimParamNames)
    ParamsEst.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp});
end
if ~isempty(ParametrizeParamsFn)
    ParamsEst=ParametrizeParamsFn(ParamsEst);
end
for vv=1:length(ObsNames)
    if isfield(MeasurementError,ObsNames{vv})
        ME=MeasurementError.(ObsNames{vv});
        if ~isfield(ME,'distribution')
            ME.distribution='normal';
        end
        if ~isfield(ME,'type')
            ME.type='additive';
        end
        estsummary.measurementerror.(ObsNames{vv}).distribution=ME.distribution;
        if any(strcmp(ME.distribution,{'normal','t'}))
            sigma=ParamsEst.(ME.sigma);
            tempdata=Data.(ObsNames{vv})(:);
            tempdata=tempdata(~isnan(tempdata));
            if strcmp(ME.type,'log')
                tempdata=log(tempdata);
            end
            estsummary.measurementerror.(ObsNames{vv}).sigma=sigma;
            if isfield(ME,'rho') && isnumeric(sigma) && isnumeric(ParamsEst.(ME.rho)) && isnumeric(ParamsEst.(ME.sigmaAR))
                % the AR(1) component: the spacing of its grid, and the iid s.d. relative to it (should be well above one)
                sd_u=ParamsEst.(ME.sigmaAR)/sqrt(1-ParamsEst.(ME.rho)^2);
                ARgridspacing=2*sqrt(estimoptions.nARpoints-1)*sd_u/(estimoptions.nARpoints-1);
                estsummary.measurementerror.(ObsNames{vv}).rho=ParamsEst.(ME.rho);
                estsummary.measurementerror.(ObsNames{vv}).sigmaAR=ParamsEst.(ME.sigmaAR);
                estsummary.measurementerror.(ObsNames{vv}).ARgridspacing=ARgridspacing;
                estsummary.measurementerror.(ObsNames{vv}).sigmaovergridspacing=min(sigma(:))/ARgridspacing;
                if warnARgrid==1 && min(sigma(:))<ARgridspacing
                    warning('At the estimate, for %s the s.d. of the iid measurement error (%.4g) is smaller than the spacing of the AR(1) grid (%.4g): the measurement error density is a lumpy mixture of nARpoints bumps, and the estimates are likely biased. Re-estimate with a larger estimoptions.nARpoints.',ObsNames{vv},min(sigma(:)),ARgridspacing)
                end
            end
            estsummary.measurementerror.(ObsNames{vv}).noiseshare=sigma.^2/var(tempdata);
            if estimoptions.verbose==1
                fprintf('Measurement error, %s: sigma=%s, share of the variance of the data=%s \n',ObsNames{vv},mat2str(sigma,4),mat2str(sigma.^2/var(tempdata),4))
            end
        elseif strcmp(ME.distribution,'misclassified')
            estsummary.measurementerror.(ObsNames{vv}).misclassprob=ParamsEst.(ME.misclassprob);
        end
    end
end
estsummary.notes.measurementerror='For normal and t measurement error: sigma at the estimate, and noiseshare=sigma^2/var(data) (of log data for type=''log''), the share of the variance of the data attributed to measurement error. A large share often signals misspecification rather than reporting error. The measurement error s.d. should also be large relative to the grid spacing of the model values. With an AR(1) component: rho, sigmaAR, ARgridspacing=2*sqrt(nARpoints-1)*sd_u/(nARpoints-1), and sigmaovergridspacing (the iid s.d. over ARgridspacing), which should be well above one, else the density is lumpy and the estimates biased (increase estimoptions.nARpoints).';

%% Check of the grid: the log-likelihood at the estimate with other values of ngridinterp
if ~isempty(estimoptions.refinecheck_ngridinterp)
    estimoptionsRefine=estimoptionsJacobian;
    estimoptionsRefine.vectoroutput=1;
    estsummary.refinecheck.ngridinterp=[vfoptions.ngridinterp, estimoptions.refinecheck_ngridinterp(:)'];
    estsummary.refinecheck.LogL=zeros(1,length(estsummary.refinecheck.ngridinterp));
    estsummary.refinecheck.LogL(1)=LogL0;
    for ii=1:length(estimoptions.refinecheck_ngridinterp)
        vfoptionsRefine=vfoptions; vfoptionsRefine.ngridinterp=estimoptions.refinecheck_ngridinterp(ii);
        simoptionsRefine=simoptions; simoptionsRefine.ngridinterp=estimoptions.refinecheck_ngridinterp(ii);
        estsummary.refinecheck.LogL(ii+1)=sum(gather(EstimateMLObjectiveFn_Jac(modelestimparamsvec,estimoptionsRefine,vfoptionsRefine,simoptionsRefine)));
    end
    estsummary.notes.refinecheck='The log-likelihood at the estimate with the ngridinterp used (first) and with each of estimoptions.refinecheck_ngridinterp. Large changes mean the grid matters for the estimate.';
end

if estimoptions.skipestimation==1
    estsummary.warningskipestimation='Warning: this estimation used estimoptions.skipestimation=1 (all good, just reminding you as you need to be careful when using skipestimation=1 :)';
end

end
