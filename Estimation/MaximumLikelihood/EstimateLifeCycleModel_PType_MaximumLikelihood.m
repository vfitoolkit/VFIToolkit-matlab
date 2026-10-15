function [EstimParams, EstimParamsConfInts, estsummary]=EstimateLifeCycleModel_PType_MaximumLikelihood(EstimParamNames,Data,MeasurementError,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions)
% Maximum likelihood estimation of a life-cycle (finite horizon) model with permanent types, from panel data.
%
% As EstimateLifeCycleModel_MaximumLikelihood, but the model has permanent types and the likelihood is the finite mixture over them
% (LogLikelihood_FHorz_PType): each individual's likelihood is sum_k w_k L_i^k, with w_k the masses of the permanent types
% (PTypeDistParamNames) and L_i^k their likelihood if they are of type k. Data.ptype can give the type of the individuals whose type
% is observed (see LogLikelihood_FHorz_PType).
%
% Inputs: as for EstimateLifeCycleModel_PType_MethodOfMoments, with Data and MeasurementError in place of the target moments and
% weighting matrices (and no AgeWeightParamNames). Parameters that differ by permanent type are given as structures over the types or
% with a dimension of length N_i, as for the other PType commands, and are estimated type by type (one block per type). The masses of
% the permanent types can be estimated (include PTypeDistParamNames{1} in EstimParamNames; it is then up to you to keep them summing to
% one, e.g. by estimating a parameter that ParametrizeParamsFn maps into the masses).
%
% estimoptions: as for EstimateLifeCycleModel_MaximumLikelihood (fminalgo, stderrors, confidenceintervals, eedefault, the constraints,
% omitestimparam, skipestimation, initialcondition, nindividualsbatch, refinecheck_ngridinterp, verbose, toleranceparams,
% toleranceobjective, insigma, inopts).
%
% Outputs: as for EstimateLifeCycleModel_MaximumLikelihood (a per-type parameter as a structure over the types, or as a matrix when it
% was given that way), plus estsummary.PTypePosterior (N-by-N_i, the posterior probability of each type for each individual at the
% estimate) and estsummary.ptypemasses (the masses of the types at the estimate).

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
    estimoptions.eedefault=3;
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
    error('EstimateLifeCycleModel_PType_MaximumLikelihood: estimoptions.fminalgo must be 4 (CMA-ES, default), 1 (fminsearch) or 6 (fmincon)')
end
if ~isfield(estimoptions,'stderrors')
    estimoptions.stderrors='OPG';
end
if ~any(strcmp(estimoptions.stderrors,{'OPG','sandwich','Hessian'}))
    error('EstimateLifeCycleModel_PType_MaximumLikelihood: estimoptions.stderrors must be ''OPG'', ''sandwich'' or ''Hessian''')
end
if ~isfield(estimoptions,'skipestimation')
    estimoptions.skipestimation=0;
end
if ~isfield(estimoptions,'refinecheck_ngridinterp')
    estimoptions.refinecheck_ngridinterp=[];
end
% Following are estimoptions used internally
estimoptions.vectoroutput=0; % =0: the objective is minus the log-likelihood (for the optimizer); =1: the vector of individual log-likelihoods LogLi (for the scores); =2: [LogLi, PTypePosterior]

%% The grid interpolation layer (must be the same for every permanent type)
if ~isfield(vfoptions,'gridinterplayer')
    vfoptions.gridinterplayer=0;
end
if ~isfield(simoptions,'gridinterplayer')
    simoptions.gridinterplayer=0;
end
if isstruct(vfoptions.gridinterplayer) || isstruct(simoptions.gridinterplayer)
    error('EstimateLifeCycleModel_PType_MaximumLikelihood: vfoptions.gridinterplayer and simoptions.gridinterplayer must be the same for every permanent type')
end
if vfoptions.gridinterplayer~=simoptions.gridinterplayer
    error('EstimateLifeCycleModel_PType_MaximumLikelihood: vfoptions.gridinterplayer and simoptions.gridinterplayer must be the same')
end
if vfoptions.gridinterplayer==1
    if ~isfield(vfoptions,'ngridinterp') || ~isfield(simoptions,'ngridinterp') || vfoptions.ngridinterp~=simoptions.ngridinterp
        error('EstimateLifeCycleModel_PType_MaximumLikelihood: vfoptions.ngridinterp and simoptions.ngridinterp must both be set, and be the same')
    end
else
    warning('EstimateLifeCycleModel_PType_MaximumLikelihood: without the grid interpolation layer (vfoptions.gridinterplayer=1) the likelihood is a step function of the parameters with large steps; the estimate and especially the finite-difference standard errors can be unreliable')
end
if estimoptions.fminalgo==6
    warning('EstimateLifeCycleModel_PType_MaximumLikelihood: fmincon (estimoptions.fminalgo=6) uses derivatives, but the likelihood is a step function of the parameters (the policy is chosen on a grid); a derivative-free method (fminalgo=4, CMA-ES, the default) is more reliable')
end
if ~isempty(estimoptions.refinecheck_ngridinterp) && vfoptions.gridinterplayer==0
    error('EstimateLifeCycleModel_PType_MaximumLikelihood: estimoptions.refinecheck_ngridinterp needs the grid interpolation layer')
end

%% Set up Names_i and N_i
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

%% Setup for which parameters are being estimated (as for PType method of moments)
% First figure out how many parameters there are (tricky as they can be dependent on ptype)
nEstimParams=0;
nEstimParamsFinder=[]; % rows are the nEstimParams, first column is pp, second column is ii
nEstimParams_PTypeMatrix=[]; % records which ptype parameters are set up as matrix, only use is in setting up the final output, =1 means N_i is first dim, =2 means N_i is second dim
for pp=1:length(EstimParamNames)
    if isstruct(Parameters.(EstimParamNames{pp}))
        nEstimParams_PTypeMatrix(pp,1)=0;
        for ii=1:N_i
            if isfield(Parameters.(EstimParamNames{pp}),Names_i{ii})
                nEstimParams=nEstimParams+1;
                nEstimParamsFinder(nEstimParams,1)=pp;
                nEstimParamsFinder(nEstimParams,2)=ii;
            end
        end
    else
        if any(size(Parameters.(EstimParamNames{pp}))==N_i) && ~any(strcmp(EstimParamNames{pp},PTypeDistParamNames)) % parameter depends on ptype, as matrix. Convert it to struct (the ptype masses themselves are one block)
            temp=Parameters.(EstimParamNames{pp});
            if size(temp,1)==N_i
                temp=temp';
                nEstimParams_PTypeMatrix(pp,1)=1;
            else
                nEstimParams_PTypeMatrix(pp,1)=2;
            end
            Parameters=rmfield(Parameters,(EstimParamNames{pp}));
            for ii=1:N_i
                nEstimParams=nEstimParams+1;
                nEstimParamsFinder(nEstimParams,1)=pp;
                nEstimParamsFinder(nEstimParams,2)=ii;
                Parameters.(EstimParamNames{pp}).(Names_i{ii})=temp(:,ii);
            end
        else % parameter does not depend on ptype
            nEstimParams_PTypeMatrix(pp,1)=0;
            nEstimParams=nEstimParams+1;
            nEstimParamsFinder(nEstimParams,1)=pp;
            nEstimParamsFinder(nEstimParams,2)=0;
        end
    end
end

% Sometimes we want to omit parameters
if isfield(estimoptions,'omitestimparam')
    OmitEstimParamsNames=fieldnames(estimoptions.omitestimparam);
else
    OmitEstimParamsNames={''};
end
estimparamsvec0=[]; % column vector
estimparamsvecindex=zeros(nEstimParams+1,1); % Note, first element remains zero
estimparamssizes=zeros(nEstimParams,1); % with PType, some parameters may be matrices (depend on both j and i)
estimomitparams_counter=zeros(nEstimParams,1); % column vector: estimomitparamsvec allows omitting the parameter for certain ages
estimomitparamsmatrix=zeros(N_j,1); % Each row is of size N_j-by-1 and holds the omitted values of a parameter
for pp=1:nEstimParams
    if nEstimParamsFinder(pp,2)==0 % Doesn't depend on ptype
        currentparameter=Parameters.(EstimParamNames{nEstimParamsFinder(pp,1)});
    else % depends on ptype
        currentparameter=Parameters.(EstimParamNames{nEstimParamsFinder(pp,1)}).(Names_i{nEstimParamsFinder(pp,2)});
    end
    estimparamssizes(pp,1:2)=size(currentparameter);
    if any(strcmp(OmitEstimParamsNames,EstimParamNames{nEstimParamsFinder(pp,1)})) % Omitting part of parameters cannot differ across permanent types
        tempparam=currentparameter;
        tempomitparam=estimoptions.omitestimparam.(EstimParamNames{nEstimParamsFinder(pp,1)});
        tempparam=tempparam(:);
        tempomitparam=tempomitparam(:);
        if ~all(tempomitparam(~isnan(tempomitparam))==tempparam(~isnan(tempomitparam)))
            fprintf('Following are the name, omit value, and initial value that related to following error (they should be the same in the non-NaN entries to be estimated) \n')
            EstimParamNames{nEstimParamsFinder(pp,1)}
            estimoptions.omitestimparam.(EstimParamNames{nEstimParamsFinder(pp,1)})
            currentparameter
            error('You have set an omitted estimated parameter, but the set values do not match the initial guess')
        end
        tempparam=tempparam(isnan(tempomitparam)); % only keep those which are NaN, not those with value for omitted
        estimparamsvec0=[estimparamsvec0; tempparam];
        estimparamsvecindex(pp+1)=estimparamsvecindex(pp)+length(tempparam);
        estimomitparams_counter(pp)=1;
        estimomitparamsmatrix(:,sum(estimomitparams_counter))=tempomitparam;
    else
        if size(currentparameter,2)==1
            estimparamsvec0=[estimparamsvec0; currentparameter];
        else
            estimparamsvec0=[estimparamsvec0; currentparameter']; % transpose
        end
        estimparamsvecindex(pp+1)=estimparamsvecindex(pp)+length(currentparameter);
    end
end

% If the parameter is constrained in some way then we need to transform it
[estimparamsvec0,estimoptions]=ParameterConstraints_PType_TransformParamsToUnconstrained(estimparamsvec0,estimparamsvecindex,EstimParamNames,nEstimParamsFinder,estimoptions,1);
% Also converts the constraints info in estimoptions to be a vector (one entry per block) rather than by name.


%% Set up the objective function
% The shock grids are not precomputed: ValueFnIter_Case1_FHorz_PType and LogLikelihood_FHorz_PType set them up at every evaluation,
% so shocks that depend on the estimated parameters (directly or via ParametrizeParamsFn) are always current.
EstimateMLObjectiveFn=@(estimparamsvec) EstimateLifeCycleModel_PType_MaximumLikelihood_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, Data, MeasurementError, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);
EstimateMLObjectiveFn_Jac=@(estimparamsvec,estimoptions_temp,vfoptions_temp,simoptions_temp) EstimateLifeCycleModel_PType_MaximumLikelihood_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, Data, MeasurementError, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions_temp, vfoptions_temp,simoptions_temp);


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
    warning('Skipping the estimation step (you have set estimoptions.skipestimation=1 in EstimateLifeCycleModel_PType_MaximumLikelihood() [Nothing wrong with this, just warning as want to be sure you did this on purpose]')
    estimparamsvec=estimparamsvec0;
end

% estimparamsvec contains the (transformed) unconstrained parameters, switch back to the original (constrained) parameter values
[estimparamsvec,~]=ParameterConstraints_TransformParamsToOriginal(estimparamsvec,estimparamsvecindex,EstimParamNames,estimoptions);


%% The scores (derivatives of each individual's log-likelihood with respect to the parameters), by finite differences
% As for EstimateLifeCycleModel_MaximumLikelihood. The constraints are by block (a parameter, or a parameter of one type).
estimoptionsJacobian=estimoptions;
estimoptionsJacobian.constrainpositive=zeros(nEstimParams,1);
estimoptionsJacobian.constrain0to1=zeros(nEstimParams,1);
estimoptionsJacobian.constrainAtoB=zeros(nEstimParams,1);
estimoptionsJacobian.vectoroutput=1; % the objective returns LogLi, the N-by-1 vector of individual log-likelihoods
estimoptionsJacobian.verbose=0;

epsilonraw=sqrt(2.2)*10^(-8); % sqrt of double precision
epsilonmodvec=[1,10^2,10^4,10^6];
eedefault=estimoptions.eedefault;
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
        pname=find(estimparamsvecindex(2:end)>=pp,1,'first'); % the block this element belongs to (the constraints are by block)
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

estimoptionsPosterior=estimoptionsJacobian;
estimoptionsPosterior.vectoroutput=2;
temp=gather(EstimateMLObjectiveFn_Jac(modelestimparamsvec,estimoptionsPosterior,vfoptions,simoptions)); % [LogLi, PTypePosterior] at the estimate
LogLi0=temp(:,1);
estsummary.PTypePosterior=temp(:,2:end);
N=length(LogLi0);
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
    if ee==eedefault
        Scores=S;
        LogLi_updefault=LogLi_up;
        LogLi_downdefault=LogLi_down;
    end
end
estsummary.doublechecks.LogLi=LogLi0;
estsummary.doublechecks.note='Double-checks are the scores (by finite differences) for four sizes of the step epsilon, with the parameter vectors at which the individual log-likelihoods were evaluated. The parameter vector is ordered by block (a parameter, or a parameter of one type), as nEstimParamsFinder.';
estsummary.doublechecks.nEstimParamsFinder=nEstimParamsFinder;

%% The Hessian of the log-likelihood (at the default step), by finite differences (as EstimateLifeCycleModel_MaximumLikelihood)
LogL0=sum(LogLi0);
LogL_up=sum(LogLi_updefault,1)';
LogL_down=sum(LogLi_downdefault,1)';
hup=epsilonparamup(:,eedefault)-modelestimparamsvec;
hdown=modelestimparamsvec-epsilonparamdown(:,eedefault);
LogL_upup=NaN(nparams,nparams);
LogL_downdown=NaN(nparams,nparams);
if any(violateconstraintbottom) || any(violateconstrainttop)
    warning('EstimateLifeCycleModel_PType_MaximumLikelihood: a finite-difference step would leave a parameter constraint, so the Hessian (and the sandwich standard errors) are not computed')
    Hessian=NaN(nparams,nparams);
else
    Hessian=zeros(nparams,nparams);
    for jj=1:nparams
        Hessian(jj,jj)=(LogL_up(jj)-2*LogL0+LogL_down(jj))/(hup(jj)*hdown(jj));
        for kk=jj+1:nparams
            epsilonparamvec=modelestimparamsvec;
            epsilonparamvec(jj)=epsilonparamup(jj,eedefault);
            epsilonparamvec(kk)=epsilonparamup(kk,eedefault);
            LogL_upup(jj,kk)=sum(gather(EstimateMLObjectiveFn_Jac(epsilonparamvec,estimoptionsJacobian,vfoptions,simoptions)));
            epsilonparamvec=modelestimparamsvec;
            epsilonparamvec(jj)=epsilonparamdown(jj,eedefault);
            epsilonparamvec(kk)=epsilonparamdown(kk,eedefault);
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
estsummary.notes.variousmatrices=['Scores is N-by-nparams, the derivative of each individual''s log-likelihood with respect to the parameters (ordered by block, see estsummary.doublechecks.nEstimParamsFinder). OPG=Scores''*Scores (outer product of gradients). Hessian is that of the log-likelihood. Sigma_OPG=inv(OPG), Sigma_Hessian=inv(-Hessian), Sigma_sandwich=Sigma_Hessian*OPG*Sigma_Hessian. Sigma is the one used for the standard errors and confidence intervals (estimoptions.stderrors=''',estimoptions.stderrors,''').'];
estsummary.gradient=sum(Scores,1)';
estsummary.notes.gradient='The gradient of the log-likelihood at the estimate (the sum of the scores). Should be near zero at an interior maximum; as the likelihood is a step function of the parameters it is only approximately so.';

%% Local identification
if all(isfinite(OPG(:)))
    rankOPGtol=sqrt(eps)*norm(OPG);
    estsummary.localidentification.rankOPG=rank(OPG,rankOPGtol);
    estsummary.localidentification.ranktolerance=rankOPGtol;
else
    warning('EstimateLifeCycleModel_PType_MaximumLikelihood: the scores have non-finite entries, so the local identification check cannot be done')
    estsummary.localidentification.rankOPG=NaN;
    estsummary.localidentification.ranktolerance=NaN;
end
estsummary.localidentification.yesidentified=logical(estsummary.localidentification.rankOPG>=nparams);
estsummary.notes.localidentification='If the OPG matrix (the outer product of the scores, an estimate of the information matrix) is full rank then the model is locally identified [rank(OPG) should equal the number of parameters being estimated]';

%% The log-likelihood at the estimate, information criteria
nobs=0;
ObsNames=fieldnames(Data);
ObsNames=ObsNames(~strcmp(ObsNames,'age') & ~strcmp(ObsNames,'agej0') & ~strcmp(ObsNames,'ptype'));
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
estsummary.notes.PTypePosterior='PTypePosterior is N-by-N_i: the posterior probability of each permanent type (columns, in the order of Names_i) for each individual (rows), given their data, at the estimate.';

%% Clean up the first two outputs
% EstimParams: a per-type parameter as a structure over the types, or as a matrix with N_i as its first or second dimension when it was
% given that way (as PType method of moments). The standard deviations (estsummary.EstimParamsStdDev, and _OPG, _Hessian, _sandwich) in
% the same layout, NaN at the fixed entries of an omitted parameter. EstimParamsConfInts: [lower, upper], one row per entry; a structure
% over the types for a per-type parameter.
stderrnames={'','_OPG','_Hessian','_sandwich'};
stderrmatrices={estimparamscovarmatrix,Sigma_OPG,Sigma_Hessian,Sigma_sandwich};
EstimParams=struct();
EstimParamsConfInts=struct();
blockval=cell(nEstimParams,1); % the value of each block, as a column
blocksd=cell(nEstimParams,1); % and its standard deviations (of the main covariance matrix)
for pp=1:nEstimParams
    pname=EstimParamNames{nEstimParamsFinder(pp,1)};
    ii=nEstimParamsFinder(pp,2); % the type (0: common to the types)
    if estimoptions.skipestimation==0
        if estimomitparams_counter(pp)>0
            currparamraw=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
            currparamraw(isnan(currparamraw))=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
        else
            currparamraw=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
        end
    else % When skipping estimation, just returns the same parameters as you input
        if ii==0
            currparamraw=Parameters.(pname);
        else
            currparamraw=Parameters.(pname).(Names_i{ii});
        end
    end
    currparamraw=currparamraw(:);
    blockval{pp}=currparamraw;
    if ii==0 % does not depend on ptype
        EstimParams.(pname)=currparamraw;
    elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==0 % a structure over the types
        EstimParams.(pname).(Names_i{ii})=currparamraw;
    elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==1 % N_i as first dim
        if isfield(EstimParams,pname)
            temp=EstimParams.(pname);
        else
            temp=zeros(N_i,length(currparamraw));
        end
        temp(ii,:)=currparamraw';
        EstimParams.(pname)=temp;
    elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==2 % N_i as second dim
        if isfield(EstimParams,pname)
            temp=EstimParams.(pname);
        else
            temp=zeros(length(currparamraw),N_i);
        end
        temp(:,ii)=currparamraw;
        EstimParams.(pname)=temp;
    end
    for ss=1:length(stderrnames)
        tempdiag=diag(stderrmatrices{ss});
        sdblock=sqrt(tempdiag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        if estimomitparams_counter(pp)>0
            mask=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
            sdfull=nan(size(mask));
            sdfull(isnan(mask))=sdblock;
        else
            sdfull=sdblock;
        end
        sdfull=sdfull(:);
        if ss==1
            blocksd{pp}=sdfull;
        end
        sdname=['EstimParamsStdDev',stderrnames{ss}];
        if ii==0
            estsummary.(sdname).(pname)=sdfull;
        elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==0
            estsummary.(sdname).(pname).(Names_i{ii})=sdfull;
        elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==1
            if isfield(estsummary,sdname) && isfield(estsummary.(sdname),pname)
                temp=estsummary.(sdname).(pname);
            else
                temp=zeros(N_i,length(sdfull));
            end
            temp(ii,:)=sdfull';
            estsummary.(sdname).(pname)=temp;
        elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==2
            if isfield(estsummary,sdname) && isfield(estsummary.(sdname),pname)
                temp=estsummary.(sdname).(pname);
            else
                temp=zeros(length(sdfull),N_i);
            end
            temp(:,ii)=sdfull;
            estsummary.(sdname).(pname)=temp;
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
EstimParamsConfInts.notes=['These are ',num2str(estimoptions.confidenceintervals),'-percent confidence intervals: [lower, upper], one row per entry of the parameter, a structure over the types for a per-type parameter (standard errors: ',estimoptions.stderrors,')'];
confintvec=[68,80,85,90,95,98,99];
criticalvalue_normaldist_z_alphadiv2_vec=[1,1.282,1.440,1.645, 1.96, 2.33, 2.575];
for pp=1:nEstimParams
    pname=EstimParamNames{nEstimParamsFinder(pp,1)};
    ii=nEstimParamsFinder(pp,2);
    v=blockval{pp}; sd=blocksd{pp};
    if ii==0 % does not depend on ptype
        EstimParamsConfInts.(pname)=[v-criticalvalue_normaldist_z_alphadiv2*sd, v+criticalvalue_normaldist_z_alphadiv2*sd];
        for cc=1:length(confintvec)
            estsummary.confidenceintervals.(['confint',num2str(confintvec(cc))]).EstimParamsConfInts.(pname)=[v-criticalvalue_normaldist_z_alphadiv2_vec(cc)*sd, v+criticalvalue_normaldist_z_alphadiv2_vec(cc)*sd];
        end
    else
        EstimParamsConfInts.(pname).(Names_i{ii})=[v-criticalvalue_normaldist_z_alphadiv2*sd, v+criticalvalue_normaldist_z_alphadiv2*sd];
        for cc=1:length(confintvec)
            estsummary.confidenceintervals.(['confint',num2str(confintvec(cc))]).EstimParamsConfInts.(pname).(Names_i{ii})=[v-criticalvalue_normaldist_z_alphadiv2_vec(cc)*sd, v+criticalvalue_normaldist_z_alphadiv2_vec(cc)*sd];
        end
    end
end

%% The masses of the permanent types, and the measurement error diagnostics, at the estimate
ParamsEst=Parameters;
for pp=1:length(EstimParamNames)
    ParamsEst.(EstimParamNames{pp})=EstimParams.(EstimParamNames{pp});
end
if ~isempty(ParametrizeParamsFn)
    ParamsEst=ParametrizeParamsFn(ParamsEst);
end
estsummary.ptypemasses=reshape(ParamsEst.(PTypeDistParamNames{1}),[],1);
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
            sigma=ParamsEst.(ME.sigma); % a structure over the types if it differs by type
            estsummary.measurementerror.(ObsNames{vv}).sigma=sigma;
            if isnumeric(sigma)
                tempdata=Data.(ObsNames{vv})(:);
                tempdata=tempdata(~isnan(tempdata));
                if strcmp(ME.type,'log')
                    tempdata=log(tempdata);
                end
                estsummary.measurementerror.(ObsNames{vv}).noiseshare=sigma.^2/var(tempdata);
                if estimoptions.verbose==1
                    fprintf('Measurement error, %s: sigma=%s, share of the variance of the data=%s \n',ObsNames{vv},mat2str(sigma,4),mat2str(sigma.^2/var(tempdata),4))
                end
            end
        elseif strcmp(ME.distribution,'misclassified')
            estsummary.measurementerror.(ObsNames{vv}).misclassprob=ParamsEst.(ME.misclassprob);
        end
    end
end
estsummary.notes.measurementerror='For normal and t measurement error: sigma at the estimate, and (when sigma is common to the types) noiseshare=sigma^2/var(data) (of log data for type=''log''), the share of the variance of the data attributed to measurement error. A large share often signals misspecification rather than reporting error.';

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
