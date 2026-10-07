function [EstimParams, EstimParamsConfInts,estsummary]=EstimateLifeCycleModel_PType_MethodOfMoments(EstimParamNames,TargetMoments,WeightingMatrix,CoVarMatrixDataMoments,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames,PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions)
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
if ~isfield(estimoptions,'constrainpositivemethod')
    estimoptions.constrainpositivemethod='softplus'; % 'log' (uparam=log(cparam)) or 'softplus' (cparam=log(1+exp(uparam))); see ParameterConstraints_TransformParamsToUnconstrained for which suits which parameter
end
if ~isfield(estimoptions,'constrain0to1')
    estimoptions.constrain0to1={}; % names of parameters to be constrained to 0 to 1 (gets converted to binary-valued vector below)
    % Handle 0 to 1 constraints by using log-odds function to switch parameter p into unconstrained x, so x=log(p/(1-p))
    % Then use the logistic-sigmoid p=1/(1+exp(-x)) when evaluating model.
end
if ~isfield(estimoptions,'constrainAtoB')
    estimoptions.constrainAtoB={}; % names of parameters to be constrained to interval A to B (gets converted to binary-valued vector below)
    % Handle A to B constraints by converting y=(p-A)/(B-A) which is 0 to 1, and then treating as constrained 0 to 1 y (so convert to unconstrained x using log-odds function)
    % Once we have the 0 to 1 y (by converting unconstrained x with the logistic sigmoid function), we convert to p=A+(B-A)*y
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
    % When doing two-step GMM, or bootstrapping Standard Errors
    estimoptions.numbootstrapsims=100; % Number of simulations
end
if ~isfield(estimoptions,'efficientW')
    estimoptions.efficientW=0; % =1, Calculates std error of parameters under assumption that the weighting matrix is efficient (that the weighting matrix is the inverse of the covariance matrix of the data moments)
end
if ~isfield(estimoptions,'skipestimation')
    estimoptions.skipestimation=0; % =1, skips the estimation, is here so you can do estimation, and then rerun later to bootstrap the standard errors without reestimating the whole model
end
if ~isfield(estimoptions,'cohortagejshifter')
    estimoptions.cohortagejshifter=0; % =0 standard, jequaloneDist is the age j=1 distribution
end
if ~isfield(simoptions,'agemass_withCohort')
    simoptions.agemass_withCohort=[]; % ncohorts-by-N_j age weights by cohort, or a structure of these by ptype (only with cohortagejshifter); [] means every cohort gets the AgeWeightParamNames values (the population age weights) from its entry age on (with a warning)
end
% Following are estimoptions used internally, but which the user won't want to set themselves
estimoptions.vectoroutput=0; % Set to zero to get point estimates, then later set to one as part of computing Jacobian matrix J (needed for Sigma, among other things).
% estimoptions.rngindex will be set below if you have estimoptions.simulatemoments=1 to bootstrap standard errors
estimoptions.metric='MethodOfMoments';
% estimoptions.weights=WeightingMatrix; is set below, after check it is correct size
if ~isfield(estimoptions,'previousiterations')
    estimoptions.previousiterations.niters=0; % gets incremented for each iteration when using estimoptions.iterateGMM
end

if ~isfield(estimoptions,'whichcombos')
    estimoptions.whichcombos=1; % =1: the stats commands compute only the targeted (function, statistic, restriction, age, ptype or grouped) combinations, through the selectors SetupTargetMoments_FHorz builds (as CalibrateLifeCycleModel_PType); =0: every statistic of every targeted function. The moments are identical either way.
end
estimoptions.useCustomModelStats=0;
if isfield(estimoptions,'CustomModelStats')
    estimoptions.useCustomModelStats=1;
    if ~isfield(estimoptions,'CustomModelStats_usergrids')
        estimoptions.CustomModelStats_usergrids=0; % =0: pass internal z_gridvals_J & pi_z_J; =1: pass exactly the z_grid & pi_z the user input
    end
    % Stash some of the inputs so they can be passed to CustomModelStats later (only things we otherwise override).
    % So that user gets exactly what they input, not any internally reworked things
    if estimoptions.CustomModelStats_usergrids==1
        estimoptions.CustomModelStatsInputs.z_grid=z_grid;
        estimoptions.CustomModelStatsInputs.pi_z=pi_z;
    end
    % Need the following two as otherwise they would contain alreadygridvals=1
    estimoptions.CustomModelStatsInputs.vfoptions=vfoptions;
    estimoptions.CustomModelStatsInputs.simoptions=simoptions;
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

%% Setup for which parameters are being estimated
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
        if any(size(Parameters.(EstimParamNames{pp}))==N_i) % parameter depends on ptype, as matrix. Convert it to struct
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
    % Get all the parameters
    if any(strcmp(OmitEstimParamsNames,EstimParamNames{nEstimParamsFinder(pp,1)})) % Omitting part of parameters cannot differ across permanent types
        % This parameter is under an omit-mask, so need to only use part of it
        tempparam=currentparameter;
        tempomitparam=estimoptions.omitestimparam.(EstimParamNames{nEstimParamsFinder(pp,1)});
        % Make them both column vectors
        tempparam=tempparam(:);
        tempomitparam=tempomitparam(:);
        % If the omit and initial guess do not fit together, throw an error
        if ~all(tempomitparam(~isnan(tempomitparam))==tempparam(~isnan(tempomitparam)))
            fprintf('Following are the name, omit value, and initial value that related to following error (they should be the same in the non-NaN entries to be estimated) \n')
            EstimParamNames{nEstimParamsFinder(pp,1)}
            estimoptions.omitestimparam.(EstimParamNames{nEstimParamsFinder(pp,1)})
            currentparameter
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
% Also converts the constraints info in estimoptions to be a vector rather than by name.



%% Cohorts entering at different ages (estimoptions.cohortagejshifter)
if ~(isscalar(estimoptions.cohortagejshifter) && estimoptions.cohortagejshifter==0)
    if isstruct(N_j)
        error('estimoptions.cohortagejshifter is not implemented together with N_j that differs by permanent type')
    end
    if isfield(vfoptions,'n_e')
        if isstruct(vfoptions.n_e)
            error('estimoptions.cohortagejshifter is not implemented together with n_e that differs by permanent type')
        end
        N_e=prod(vfoptions.n_e);
        n_e=vfoptions.n_e;
    else
        N_e=1;
        n_e=0;
    end
    if isstruct(n_a) || isstruct(n_z)
        error('estimoptions.cohortagejshifter is not implemented together with n_a or n_z that differ by permanent type')
    end
    N_a=prod(n_a); N_z=prod(n_z);
    % jequaloneDist must have an extra (last) dimension of size N_j: slice j is the cohort entering at age j.
    % Either a structure over ptypes (mass one per ptype), or a single array used for every ptype.
    if ~isstruct(jequaloneDist)
        if numel(jequaloneDist)~=N_a*N_z*N_e*N_j
            error('estimoptions.cohortagejshifter is being used, so jequaloneDist must have an extra (last) dimension of size N_j: [n_a,n_z,(n_e),N_j] (or a structure of these by ptype)')
        end
        temp=jequaloneDist;
        jequaloneDist=struct();
        for ii=1:N_i
            jequaloneDist.(Names_i{ii})=temp;
        end
    end
    cohortmasses=zeros(N_i,N_j); % slice masses by ptype
    for ii=1:N_i
        if ~isfield(jequaloneDist,Names_i{ii})
            error(['You must input a jequaloneDist for permanent type ', Names_i{ii}])
        end
        if numel(jequaloneDist.(Names_i{ii}))~=N_a*N_z*N_e*N_j
            error(['estimoptions.cohortagejshifter is being used, so jequaloneDist.',Names_i{ii},' must have an extra (last) dimension of size N_j: [n_a,n_z,(n_e),N_j]'])
        end
        if abs(sum(jequaloneDist.(Names_i{ii})(:))-1)>10^(-9)
            error(['jequaloneDist.',Names_i{ii},' must have mass one in total (summing across all the cohort slices)'])
        end
        jequaloneDist.(Names_i{ii})=reshape(jequaloneDist.(Names_i{ii}),[N_a*N_z*N_e,N_j]);
        cohortmasses(ii,:)=gather(sum(jequaloneDist.(Names_i{ii}),1));
    end
    anymass=(sum(cohortmasses,1)>0); % ages at which some ptype has mass
    if isscalar(estimoptions.cohortagejshifter) % =1: entry ages are the slices with positive mass (for any ptype)
        estimoptions.cohortagejshifter=find(anymass);
    else % vector of entry ages
        estimoptions.cohortagejshifter=estimoptions.cohortagejshifter(:)';
        if any(anymass(setdiff(1:N_j,estimoptions.cohortagejshifter)))
            error('estimoptions.cohortagejshifter gives the entry ages, but jequaloneDist has mass at an age that is not an entry age')
        end
        if any(~anymass(estimoptions.cohortagejshifter))
            error('estimoptions.cohortagejshifter gives an entry age at which jequaloneDist has zero mass for every permanent type')
        end
    end
    estimoptions.ncohorts=length(estimoptions.cohortagejshifter);
    estimoptions.cohortmasses=cohortmasses(:,estimoptions.cohortagejshifter); % N_i-by-ncohorts
    % jequaloneDist becomes a cell over cohorts, each a structure over ptypes of normalized slices in the shape the
    % agent distribution command expects. A ptype with no mass in a cohort gets a uniform placeholder and (in the
    % objective fn) zero weight in that cohort.
    temp=jequaloneDist;
    jequaloneDist=cell(estimoptions.ncohorts,1);
    for cc=1:estimoptions.ncohorts
        jj=estimoptions.cohortagejshifter(cc);
        jequaloneDist{cc}=struct();
        for ii=1:N_i
            if cohortmasses(ii,jj)>0
                tempslice=temp.(Names_i{ii})(:,jj)/cohortmasses(ii,jj);
            else
                tempslice=ones(N_a*N_z*N_e,1)/(N_a*N_z*N_e);
            end
            if N_e==1
                jequaloneDist{cc}.(Names_i{ii})=reshape(tempslice,[n_a,n_z]);
            else
                jequaloneDist{cc}.(Names_i{ii})=reshape(tempslice,[n_a,n_z,n_e]);
            end
        end
    end
    % age weights by cohort
    if ~isempty(simoptions.agemass_withCohort)
        if isstruct(simoptions.agemass_withCohort)
            for ii=1:N_i
                if ~isfield(simoptions.agemass_withCohort,Names_i{ii})
                    error(['simoptions.agemass_withCohort is a structure but is missing permanent type ',Names_i{ii}])
                end
                if ~all(size(simoptions.agemass_withCohort.(Names_i{ii}))==[estimoptions.ncohorts,N_j])
                    error(['simoptions.agemass_withCohort.',Names_i{ii},' must be ncohorts-by-N_j'])
                end
            end
        elseif ~all(size(simoptions.agemass_withCohort)==[estimoptions.ncohorts,N_j])
            error('simoptions.agemass_withCohort must be ncohorts-by-N_j (or a structure of these by ptype)')
        end
    else
        warning('estimoptions.cohortagejshifter is being used but simoptions.agemass_withCohort is not set: every cohort gets the population age weights (AgeWeightParamNames, by ptype if that parameter is a structure) from its entry age on, so its mass falls with age as the population''s does. This only matters for pooled (AllStats, AutoCorr) cohort targets, not for age-conditional ones. Set simoptions.agemass_withCohort (ncohorts-by-N_j, or a structure of these by ptype) to choose the age weights of each cohort yourself.')
    end
    if isstruct(AgeWeightParamNames)
        error('estimoptions.cohortagejshifter is not implemented together with AgeWeightParamNames that differ by permanent type (the parameter itself can differ by ptype)')
    end
    if estimoptions.bootstrapStdErrors==1
        error('estimoptions.bootstrapStdErrors=1 is not implemented together with estimoptions.cohortagejshifter')
    end
    if isfield(TargetMoments,'CustomModelStats')
        error('TargetMoments.CustomModelStats is not implemented together with estimoptions.cohortagejshifter (if you want this, ask on the forum, discourse.vfitoolkit.com)')
    end
    if estimoptions.verbose==1
        fprintf('Cohorts: %i cohorts entering at ages %s \n',estimoptions.ncohorts,mat2str(estimoptions.cohortagejshifter))
        fprintf('Cohort masses by ptype (rows) and cohort (columns): \n')
        disp(estimoptions.cohortmasses)
    end
end

%% Setup for which moments are being targeted
if estimoptions.cohortagejshifter==0
    % Only calculate each of AllStats and LifeCycleProfiles when being used (so as faster when not using both)
    [targetmomentvec,usingallstats,usinglcp,usingcustomstats, allstatmomentnames,allstatcummomentsizes,AllStats_whichstats, FnsToEvaluate_AllStats, acsmomentnames, acscummomentsizes, ACStats_whichstats, FnsToEvaluate_ACStats,cmsmomentnames, cmscummomentsizes,selectors, usingautocorr,autocorrmomentnames,autocorrcummomentsizes,FnsToEvaluate_AutoCorr,autocorrtimehorizons, usingcrosssec,crosssecmomentnames,crossseccummomentsizes,FnsToEvaluate_CrossSec, usingagecrosssec,agecrosssecmomentnames,agecrossseccummomentsizes,FnsToEvaluate_AgeCrossSec]=SetupTargetMoments_FHorz(TargetMoments,FnsToEvaluate,1,N_j,simoptions,Names_i);
    estimoptions.selectors=selectors; % the per-combination whichcombos/whichstats of the two stats commands, with a trailing type dimension (used as estimoptions.whichcombos=1)
else
    [targetmomentvec,cohortmoments]=SetupTargetMoments_FHorz_withCohorts(TargetMoments,FnsToEvaluate,N_j,estimoptions.cohortagejshifter,1);
    if isfield(TargetMoments,'AutoCorrTransProbs') || isfield(TargetMoments,'CrossSectionCovarCorr') || isfield(TargetMoments,'AgeConditionalCrossSectionCovarCorr')
        error('TargetMoments.AutoCorrTransProbs, .CrossSectionCovarCorr and .AgeConditionalCrossSectionCovarCorr are not implemented together with estimoptions.cohortagejshifter')
    end
    % (the names tables and sizes the logmoments block below reads: none of the kinds are in use on this path, which only takes a scalar or a per-entry logmoments)
    usingallstats=0; usinglcp=0; usingcustomstats=0; usingautocorr=0; usingcrosssec=0; usingagecrosssec=0;
    allstatcummomentsizes=0; acscummomentsizes=0; autocorrcummomentsizes=0; crossseccummomentsizes=0; agecrossseccummomentsizes=0; cmscummomentsizes=0;
end


%% Now, a bunch of things to avoid redoing them every parameter vector we want to try
% Note: I avoid doing this for ReturnFnParamNames because they are so
% dependent on the setup. Same for FnsToEvaluateParamNames
ReturnFnParamNames=[];
FnsToEvaluateParamNames=[];

estimoptions.calibrateshocks=0; % set to one if need to redo shocks for each new estim parameter vector
if isfield(vfoptions,'ExogShockFn')
    if isstruct(vfoptions.ExogShockFn) % can depend on permanent type (before 2026-10-07 a structure errored here)
        temp=[];
        shockfnnames=fieldnames(vfoptions.ExogShockFn);
        for ii=1:length(shockfnnames)
            temp=[temp,getAnonymousFnInputNames(vfoptions.ExogShockFn.(shockfnnames{ii}))];
        end
    else
        temp=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    end
    % can just leave action space in here as we only use it to see if EstimParamNames is part of it
    if ~isempty(intersect(temp,EstimParamNames))
        estimoptions.calibrateshocks=1;
    end
end
if isfield(vfoptions,'EiidShockFn') % note: not elseif, can have both and either alone should trigger redoing the shocks
    if isstruct(vfoptions.EiidShockFn) % can depend on permanent type
        temp=[];
        shockfnnames=fieldnames(vfoptions.EiidShockFn);
        for ii=1:length(shockfnnames)
            temp=[temp,getAnonymousFnInputNames(vfoptions.EiidShockFn.(shockfnnames{ii}))];
        end
    else
        temp=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    end
    % can just leave action space in here as we only use it to see if EstimParamNames is part of it
    if ~isempty(intersect(temp,EstimParamNames))
        estimoptions.calibrateshocks=1;
    end
end
if estimoptions.calibrateshocks==0
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    % The user's own grids are needed if CustomModelStats is given them
    KeepOriginalGrid=((estimoptions.useCustomModelStats==1 && estimoptions.CustomModelStats_usergrids==1));
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz_PType(n_z,z_grid,pi_z,N_j,Names_i,Parameters,vfoptions,3,KeepOriginalGrid);
    if KeepOriginalGrid==1 && isfield(vfoptions,'user_z_grid')
        % ExogShockFn builds the user's own grid internally, so take it from there rather than from the z_grid input (which is then just a placeholder)
        estimoptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
        estimoptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
    end
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
else
    % The shock grids depend on a parameter being estimated, so they are rebuilt inside the objective function every evaluation. The
    % z_grid and pi_z inputs are passed through: with an ExogShockFn they are only placeholders, but with only an EiidShockFn (the iid
    % shock estimated, z as the user gave it) they are the z grids themselves, and empty placeholders broke the setup (2026-10-07).
    z_gridvals_J=z_grid;
    pi_z_J=pi_z;
end
% Regardless of whether they are done here or in _objectivefn, they will be
% precomputed by the time we get to the value fn, stationary dist, etc. So
vfoptions.alreadygridvals=1;
simoptions.alreadygridvals=1;

% Same for semi-exogenous shocks
estimoptions.calibsemiexo=0; % use =0 to also cover models without semi-exogenous shocks
if isfield(vfoptions,'n_semiz')
    if prod(vfoptions.n_semiz)>0
        if isfield(vfoptions,'SemiExoStateFn')
            if isstruct(vfoptions.SemiExoStateFn) % can depend on permanent type
                temp=[];
                semiexofnnames=fieldnames(vfoptions.SemiExoStateFn);
                for ii=1:length(semiexofnnames)
                    temp=[temp,getAnonymousFnInputNames(vfoptions.SemiExoStateFn.(semiexofnnames{ii}))];
                end
            else
                temp=getAnonymousFnInputNames(vfoptions.SemiExoStateFn);
            end
            % can just leave action space in here as we only use it to see if EstimParamNames is part of it
            if ~isempty(intersect(temp,EstimParamNames))
                estimoptions.calibsemiexo=1;
            end
        end
        vfoptions=SemiExogShockSetup_FHorz_PType(n_d,N_j,Names_i,d_grid,Parameters,vfoptions,3);
        simoptions.semiz_gridvals_J=vfoptions.semiz_gridvals_J;
        simoptions.pi_semiz_J=vfoptions.pi_semiz_J;
        % Regardless of whether they are done here or in _objectivefn, they will be precomputed by the time we get to the value fn, stationary dist, etc. So
        vfoptions.alreadygridvals_semiexo=1;
        simoptions.alreadygridvals_semiexo=1;
    end
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
if isstruct(estimoptions.logmoments) && ~(isscalar(estimoptions.cohortagejshifter) && estimoptions.cohortagejshifter==0)
    error('estimoptions.logmoments by name is not implemented together with estimoptions.cohortagejshifter (use a scalar, or a vector with one entry per target entry)')
end
% estimoptions.logmoments: which moments to take logs of (the targets must then already be log(moments); same for any covariance matrix of the data moments).
% Four forms: a scalar 0 (none) or 1 (all); a vector with one entry per target (same length as targetmomentvec); a vector with one entry
% per TARGET NAME (one per row of allstatmomentnames, then acsmomentnames, autocorrmomentnames, crosssecmomentnames, agecrosssecmomentnames, then cmsmomentnames, in that order), expanded over the
% entries of each; or by name with the same nesting as the targets, e.g. estimoptions.logmoments.AgeConditionalStats.earnings.low.Mean=1
% (names not mentioned are 0). Internally it becomes a vector with one entry per target.
momentrowsizes=[]; % the number of entries of each target name, in the order the names enter targetmomentvec
allstatsizes=diff([0,allstatcummomentsizes]); acssizes=diff([0,acscummomentsizes]); autocorrsizes=diff([0,autocorrcummomentsizes]); crosssecsizes=diff([0,crossseccummomentsizes]); agecrosssecsizes=diff([0,agecrossseccummomentsizes]); cmssizes=diff([0,cmscummomentsizes]);
if usingallstats==1
    momentrowsizes=[momentrowsizes, allstatsizes];
end
if usinglcp==1
    momentrowsizes=[momentrowsizes, acssizes];
end
if usingautocorr==1
    momentrowsizes=[momentrowsizes, autocorrsizes];
end
if usingcrosssec==1
    momentrowsizes=[momentrowsizes, crosssecsizes];
end
if usingagecrosssec==1
    momentrowsizes=[momentrowsizes, agecrosssecsizes];
end
if usingcustomstats==1
    momentrowsizes=[momentrowsizes, cmssizes];
end
if isstruct(estimoptions.logmoments)
    logmomentnames=estimoptions.logmoments;
    estimoptions.logmoments=zeros(length(targetmomentvec),1);
    sofar=0;
    if usingallstats==1
        for ii=1:size(allstatmomentnames,1)
            flag=0; % walk the (two to four) names of this target into logmomentnames.AllStats; the flag is the number at the end of the walk, if it is all there
            if isfield(logmomentnames,'AllStats')
                temp=logmomentnames.AllStats;
                found=1;
                for kk=1:size(allstatmomentnames,2)
                    if ~isempty(allstatmomentnames{ii,kk})
                        if isstruct(temp) && isfield(temp,allstatmomentnames{ii,kk})
                            temp=temp.(allstatmomentnames{ii,kk});
                        else
                            found=0;
                        end
                    end
                end
                if found==1 && isnumeric(temp) && isscalar(temp)
                    flag=temp;
                end
            end
            estimoptions.logmoments(sofar+1:sofar+allstatsizes(ii))=flag;
            sofar=sofar+allstatsizes(ii);
        end
    end
    if usinglcp==1
        for ii=1:size(acsmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'AgeConditionalStats')
                temp=logmomentnames.AgeConditionalStats;
                found=1;
                for kk=1:size(acsmomentnames,2)
                    if ~isempty(acsmomentnames{ii,kk})
                        if isstruct(temp) && isfield(temp,acsmomentnames{ii,kk})
                            temp=temp.(acsmomentnames{ii,kk});
                        else
                            found=0;
                        end
                    end
                end
                if found==1 && isnumeric(temp) && isscalar(temp)
                    flag=temp;
                end
            end
            estimoptions.logmoments(sofar+1:sofar+acssizes(ii))=flag;
            sofar=sofar+acssizes(ii);
        end
    end
    if usingautocorr==1
        for ii=1:size(autocorrmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'AutoCorrTransProbs')
                temp=logmomentnames.AutoCorrTransProbs;
                found=1;
                for kk=1:size(autocorrmomentnames,2)
                    if ~isempty(autocorrmomentnames{ii,kk})
                        if isstruct(temp) && isfield(temp,autocorrmomentnames{ii,kk})
                            temp=temp.(autocorrmomentnames{ii,kk});
                        else
                            found=0;
                        end
                    end
                end
                if found==1 && isnumeric(temp) && isscalar(temp)
                    flag=temp;
                end
            end
            estimoptions.logmoments(sofar+1:sofar+autocorrsizes(ii))=flag;
            sofar=sofar+autocorrsizes(ii);
        end
    end
    if usingcrosssec==1
        for ii=1:size(crosssecmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'CrossSectionCovarCorr')
                temp=logmomentnames.CrossSectionCovarCorr;
                found=1;
                for kk=1:size(crosssecmomentnames,2)
                    if ~isempty(crosssecmomentnames{ii,kk})
                        if isstruct(temp) && isfield(temp,crosssecmomentnames{ii,kk})
                            temp=temp.(crosssecmomentnames{ii,kk});
                        else
                            found=0;
                        end
                    end
                end
                if found==1 && isnumeric(temp) && isscalar(temp)
                    flag=temp;
                end
            end
            estimoptions.logmoments(sofar+1:sofar+crosssecsizes(ii))=flag;
            sofar=sofar+crosssecsizes(ii);
        end
    end
    if usingagecrosssec==1
        for ii=1:size(agecrosssecmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'AgeConditionalCrossSectionCovarCorr')
                temp=logmomentnames.AgeConditionalCrossSectionCovarCorr;
                found=1;
                for kk=1:size(agecrosssecmomentnames,2)
                    if ~isempty(agecrosssecmomentnames{ii,kk})
                        if isstruct(temp) && isfield(temp,agecrosssecmomentnames{ii,kk})
                            temp=temp.(agecrosssecmomentnames{ii,kk});
                        else
                            found=0;
                        end
                    end
                end
                if found==1 && isnumeric(temp) && isscalar(temp)
                    flag=temp;
                end
            end
            estimoptions.logmoments(sofar+1:sofar+agecrosssecsizes(ii))=flag;
            sofar=sofar+agecrosssecsizes(ii);
        end
    end
    if usingcustomstats==1
        for ii=1:size(cmsmomentnames,1)
            flag=0;
            if isfield(logmomentnames,'CustomModelStats') && isfield(logmomentnames.CustomModelStats,cmsmomentnames{ii,1})
                flag=logmomentnames.CustomModelStats.(cmsmomentnames{ii,1});
            end
            estimoptions.logmoments(sofar+1:sofar+cmssizes(ii))=flag;
            sofar=sofar+cmssizes(ii);
        end
    end
elseif any(estimoptions.logmoments>0)
    if isscalar(estimoptions.logmoments)
        estimoptions.logmoments=ones(length(targetmomentvec),1); % log all of them
    elseif length(estimoptions.logmoments)==length(targetmomentvec)
        estimoptions.logmoments=reshape(estimoptions.logmoments,[length(targetmomentvec),1]); % already one entry per target
    elseif length(estimoptions.logmoments)==length(momentrowsizes)
        estimoptions.logmoments=repelem(reshape(estimoptions.logmoments,[],1),reshape(momentrowsizes,[],1)); % one entry per target name, expanded over the entries of each
    else
        fprintf('Relevant to following error: length(estimoptions.logmoments)=%i \n', length(estimoptions.logmoments))
        fprintf('Relevant to following error: number of target names=%i, number of target entries=%i \n', length(momentrowsizes), length(targetmomentvec))
        error('You are using estimoptions.logmoments, but length(estimoptions.logmoments) matches neither the number of target names nor the number of target entries')
    end
else
    estimoptions.logmoments=zeros(length(targetmomentvec),1);
end


%% Turn off some warnings that would normally be given (as they are otherwise repeated ad infinitum)
if ~isfield(simoptions,'warnjequaloneptypeasdim')
    simoptions.warnjequaloneptypeasdim=0;
end

%% Set up the objective function and the initial calibration parameter vector
% Note: _objectivefn is shared between Method of Moments Estimation and Calibration
if estimoptions.fminalgo==8
    estimoptions.vectoroutput=2;
    estimoptions.weights=chol(estimoptions.weights,'upper'); % To use a weighting matrix in lsqnonlin(), we work with the upper-cholesky decomposition
end
% EstimateMoMObjectiveFn is used by the minimization. EstimateMoMObjectiveFn_Jac is the same objective but taking the
% Parameters and estimoptions as inputs, used below for the Jacobian (estimoptionsJacobian) and the sensitivity to CalibParamsNames
if estimoptions.cohortagejshifter==0
    EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_PType_objectivefn(estimparamsvec, EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, usingallstats,usinglcp,usingcustomstats, targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes, AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, usingautocorr, autocorrmomentnames, autocorrcummomentsizes, FnsToEvaluate_AutoCorr, autocorrtimehorizons, usingcrosssec, crosssecmomentnames, crossseccummomentsizes, FnsToEvaluate_CrossSec, usingagecrosssec, agecrosssecmomentnames, agecrossseccummomentsizes, FnsToEvaluate_AgeCrossSec, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);
    EstimateMoMObjectiveFn_Jac=@(estimparamsvec,Parameters_temp,estimoptions_temp) CalibrateLifeCycleModel_PType_objectivefn(estimparamsvec, EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters_temp, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, usingallstats,usinglcp,usingcustomstats, targetmomentvec, allstatmomentnames, acsmomentnames, cmsmomentnames,allstatcummomentsizes, acscummomentsizes,cmscummomentsizes, AllStats_whichstats, ACStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_ACStats, usingautocorr, autocorrmomentnames, autocorrcummomentsizes, FnsToEvaluate_AutoCorr, autocorrtimehorizons, usingcrosssec, crosssecmomentnames, crossseccummomentsizes, FnsToEvaluate_CrossSec, usingagecrosssec, agecrosssecmomentnames, agecrossseccummomentsizes, FnsToEvaluate_AgeCrossSec, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions_temp, vfoptions,simoptions);
else
    EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_PType_withCohorts_objectivefn(estimparamsvec, EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, targetmomentvec, cohortmoments, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);
    EstimateMoMObjectiveFn_Jac=@(estimparamsvec,Parameters_temp,estimoptions_temp) CalibrateLifeCycleModel_PType_withCohorts_objectivefn(estimparamsvec, EstimParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters_temp, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, targetmomentvec, cohortmoments, nEstimParams, nEstimParamsFinder, estimparamsvecindex, estimparamssizes, estimomitparams_counter, estimomitparamsmatrix, estimoptions_temp, vfoptions,simoptions);
end
if estimoptions.fminalgo==8
    estimoptions.weights=WeightingMatrix; % change it back now that we have set up CalibrateLifeCycleModel_objectivefn()
end
% estimparamsvec0 is our initial guess for estimparamsvec


%% Choosing algorithm for the optimization problem
if estimoptions.skipestimation==0
    % https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
    minoptions = optimset('TolX',estimoptions.toleranceparams,'TolFun',estimoptions.toleranceobjective);
    if estimoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
        estimoptions.multiGEcriterion=0;
        [estimparamsvec,fval]=fzero(EstimateMoMObjectiveFn,estimparamsvec0,minoptions);
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
        weight=ones(length(estimparamsvec0),1); % I already implement weights via estimoptions
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
        % Update based on rules in estimoptions.fminalgo5.howtoupdate
        error('fminalgo=5 is not possible with model calibration/estimation')
    elseif estimoptions.fminalgo==6
        if ~isfield(estimoptions,'lb') || ~isfield(estimoptions,'ub')
            error('When using constrained optimization (estimoptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using estimoptions.lb and estimoptions.ub')
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


%% Compute the standard deviation of the estimated parameters
if estimoptions.bootstrapStdErrors==0
    % First, need the Jacobian matrix, which involves computing all the
    % derivatives of the individual moments with respect to the estimated parameters
    estimoptionsJacobian=estimoptions;
    estimoptionsJacobian.constrainpositive=zeros(nEstimParams,1); % eliminate constraints for Jacobian
    estimoptionsJacobian.constrain0to1=zeros(nEstimParams,1); % eliminate constraints for Jacobian
    estimoptionsJacobian.constrainAtoB=zeros(nEstimParams,1); % eliminate constraints for Jacobian
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
    % I want to do epsilon change in the model parameter, but here I have
    % the unconstrained parameters. So I create an epsilonparamup and
    % epsilonparamdown, which contain the unconstrained values the
    % correspond to epsilon changes in the constrained parameters
    % I do this in a separate loop, which is a loss of runtime, but this is
    % minor and is much easier to read so whatever
    epsilonparamup=zeros(length(estimparamsvec),length(epsilonmodvec));
    epsilonparamdown=zeros(length(estimparamsvec),length(epsilonmodvec));
    modelestimparamsvec=estimparamsvec;
    modelestimparamsvecup=zeros(size(modelestimparamsvec));
    modelestimparamsvecdown=zeros(size(modelestimparamsvec));
    violateconstrainttop=zeros(size(modelestimparamsvec)); %=1 means use a one-sided (down) finite-difference because 'adding epsilon' would lead to a parameter value that violates the constraint
    violateconstraintbottom=zeros(size(modelestimparamsvec)); %=1 means use a one-sided (up) finite-difference because 'subtracting epsilon' would lead to a parameter value that violates the constraint
    % (estimparamsvec is already the constrained (original) parameters: it was transformed back right after the estimation step; before 2026-10-07 it was transformed a second time here, so J was taken at the wrong point whenever a constraint was in use)
    % Now, multiply by (1+-epsilon)
    for ee=1:length(epsilonmodvec)
        epsilon=epsilonmodvec(ee)*epsilonraw;
        for pp=1:length(estimparamsvec) % every element of the parameter vector
            pname=find(estimparamsvecindex(2:end)>=pp,1,'first'); % the block (parameter, or parameter of one type) this element belongs to: the constraints are by block (before 2026-10-07 they were indexed by the element)
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
            if estimoptions.constrainpositive(pname)==1 % Forcing this parameter to be positive
                if modelestimparamsvecdown(pp)<=0
                    violateconstraintbottom(pp)=1;
                end
            elseif estimoptions.constrainAtoB(pname)==1 % Constrain A to B
                if modelestimparamsvecdown(pp)<=estimoptions.constrainAtoBlimits(pname,1) % less than A
                    violateconstraintbottom(pp)=1;
                elseif modelestimparamsvecup(pp)>=estimoptions.constrainAtoBlimits(pname,2) % greater than B
                    violateconstrainttop(pp)=1;
                end
            elseif estimoptions.constrain0to1(pname)==1 % Constrain 0 to 1 (but not as part of A to B)
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
        ObjValue=EstimateMoMObjectiveFn_Jac(epsilonparamvec,Parameters,estimoptionsJacobian);
        for pp=1:length(estimparamsvec)
            epsilonparamvec=modelestimparamsvec;
            epsilonparamvec(pp)=epsilonparamup(pp,ee); % add epsilon*x to the pp-th parameter
            ObjValue_upwind(:,pp)=EstimateMoMObjectiveFn_Jac(epsilonparamvec,Parameters,estimoptionsJacobian);
            epsilonparamvec(pp)=epsilonparamdown(pp,ee); % subtract epsilon*x from the pp-th parameter
            ObjValue_downwind(:,pp)=EstimateMoMObjectiveFn_Jac(epsilonparamvec,Parameters,estimoptionsJacobian);
        end
        epsilonparamvec=modelestimparamsvec; % and using estimoptionsJacobian, so using the actual parameters, rather than the transformed parameters

        % Use finite-difference to compute the derivatives
        J_up=(ObjValue_upwind-ObjValue)./((epsilonparamup(:,ee)-epsilonparamvec)');
        J_down=(ObjValue-ObjValue_downwind)./((epsilonparamvec-epsilonparamdown(:,ee))');
        J_centered=(ObjValue_upwind-ObjValue_downwind)./((epsilonparamup(:,ee)-epsilonparamdown(:,ee))');
        % Jacobian matix of derivatives of model moments with respect to parameters, evaluated at the parameter point estimates

        % J is nmoments-by-nparams
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
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvec=epsilonparamvec; % Is actually independent of ee anyway
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvecup=epsilonparamup(:,ee);
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvecdown=epsilonparamdown(:,ee);
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).violateconstraintbottom=violateconstraintbottom;
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).violateconstrainttop=violateconstrainttop;
        if ee==eedefault
            J=J_full; % This is the one used to report Sigma (parameter std deviations) [corresponds to epsilon=sqrt(2.2)*10^(-4)]
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


    % The model moments at the estimate (ObjValue is the Jacobian block's evaluation there: the targeted entries, logged where
    % estimoptions.logmoments asks), the targets alongside, and the objective (M_d-M_m)'W(M_d-M_m) recomputed from them. Before 2026-10-07
    % objectivefnval was the optimiser's own value, which lsqnonlin reports as (M_d-M_m)'W(M_d-M_m) but fminsearch and skipestimation as
    % that divided by the number of parameters.
    estsummary.currentmomentvec=gather(ObjValue(:));
    estsummary.targetmomentvec=targetmomentvec(~isnan(targetmomentvec));
    fval=(estsummary.currentmomentvec-estsummary.targetmomentvec)'*WeightingMatrix*(estsummary.currentmomentvec-estsummary.targetmomentvec);
end




%% Bootstrap standard errors
if estimoptions.bootstrapStdErrors==1
    error('HAVE NOT YET IMPLEMENTED BOOTSTRAP STANDARD ERRORS (you have estimoptions.bootstrapStdErrors=1)')
end


%% Local identification
if estimoptions.bootstrapStdErrors==0 % Depends on derivatives, so cannot do when bootstrapping the standard errors
    % The estimate is locally identified if the matrix J is full rank
    % The estimate is locally identified if J has full column rank. J is finite differences, so a column that is zero in truth carries
    % noise of the order of eps/(relative step): rank() with its default tolerance read a 1e-12 column as rank (2026-10-07), so the
    % tolerance is sqrt(eps) times the norm of J. A non-finite entry (a moment undefined at the estimate, e.g. a conditional
    % restriction with no mass at an age) makes the check impossible, and rank() would error.
    if all(isfinite(J(:)))
        rankJtol=sqrt(eps)*norm(J);
        estsummary.localidentification.rankJ=rank(J,rankJtol); % If this is greater or equal to number of parameters, then locally identified
        estsummary.localidentification.ranktolerance=rankJtol;
    else
        warning('EstimateLifeCycleModel_PType_MethodOfMoments: J has non-finite entries (a model moment is NaN or Inf at the estimate, moment rows %s), so the local identification check cannot be done',mat2str(find(any(~isfinite(J),2))'))
        estsummary.localidentification.rankJ=NaN;
        estsummary.localidentification.ranktolerance=NaN;
    end
    estsummary.localidentification.yesidentified=logical(estsummary.localidentification.rankJ>=length(estimparamsvec));
    estsummary.notes.localidentification='If the Jacobian matrix (derivatives of model moments with respect to parameter vector) is full rank then the model is locally identified [so rank(J) should be greater than or equal to number of parameters being estimated]';
end

%% Some additional outputs
% Mainly, the Sensitivity matrix
if estimoptions.bootstrapStdErrors==0 % Depends on derivatives, so cannot do when bootstrapping the standard errors
    % Sensitivity of estimated parameters to the target moments
    % Sensitivity matrix, Lambda, of Andrews, Gentzkow & Shapiro (2017) - Measuring the Sensitivity of Parameter Estimates to Estimation Moments
    SensitivityMatrix=(-(J'*WeightingMatrix*J)^(-1))*(J'*WeightingMatrix);
    estsummary.sensitivitymatrix=SensitivityMatrix;

    % Sensitivity of estimated parameters to the pre-calibrated parameters
    % If you have set estimoptions.CalibParamNames; Jorgensen (2023) - Sensitivity to Calibrated Parameters
    % Requires calculating derivatives of the objective vector to the calibrated parameters
    if isfield(estimoptions,'CalibParamsNames')
        ncp=length(estimoptions.CalibParamsNames);
        calibparamvec=zeros(ncp,1);
        calibstepup=zeros(ncp,1);
        calibstepdown=zeros(ncp,1);
        ObjValue_upwind=zeros(sum(~isnan(targetmomentvec)),ncp); % derivatives of the model moments with respect to the pre-calibrated parameters, evaluated at the estimate
        ObjValue_downwind=zeros(sum(~isnan(targetmomentvec)),ncp);
        CalibParams=struct();
        for pp=1:ncp
            if ~isscalar(Parameters.(estimoptions.CalibParamsNames{pp}))
                error(['estimoptions.CalibParamsNames: ',estimoptions.CalibParamsNames{pp},' is not a scalar (the sensitivity to calibrated parameters is implemented for scalar parameters)'])
            end
            CalibParams.(estimoptions.CalibParamsNames{pp})=Parameters.(estimoptions.CalibParamsNames{pp});
            calibparamvec(pp)=Parameters.(estimoptions.CalibParamsNames{pp});
        end
        % Centered finite differences in each calibrated parameter (the step regime is that of the Jacobian above, by the size of the
        % parameter itself; the calibrated parameters carry no constraints, so both sides are always used). Before 2026-10-07 the 'subtract'
        % step added epsilon again and the regime was read off the estimated parameter of the same index.
        for pp=1:ncp
            cval=calibparamvec(pp);
            if floor(log(abs(cval))/log(10))>-2 % order of magnitude is greater than 10^(-2)
                calibstepup(pp)=(1+epsilon)*cval; calibstepdown(pp)=(1-epsilon)*cval;
            elseif floor(log(abs(cval))/log(10))<-4 % so small (or zero) that epsilon is added/subtracted outright
                calibstepup(pp)=cval+epsilon; calibstepdown(pp)=cval-epsilon;
            else
                calibstepup(pp)=(1+epsilonalt(eedefault))*cval; calibstepdown(pp)=(1-epsilonalt(eedefault))*cval;
            end
            Parameters.(estimoptions.CalibParamsNames{pp})=calibstepup(pp);
            ObjValue_upwind(:,pp)=EstimateMoMObjectiveFn_Jac(modelestimparamsvec,Parameters,estimoptionsJacobian);
            Parameters.(estimoptions.CalibParamsNames{pp})=calibstepdown(pp);
            ObjValue_downwind(:,pp)=EstimateMoMObjectiveFn_Jac(modelestimparamsvec,Parameters,estimoptionsJacobian);
            Parameters.(estimoptions.CalibParamsNames{pp})=CalibParams.(estimoptions.CalibParamsNames{pp}); % restore
        end
        Jcalib_up=(ObjValue_upwind-ObjValue)./((calibstepup-calibparamvec)');
        Jcalib_down=(ObjValue-ObjValue_downwind)./((calibparamvec-calibstepdown)');
        Jcalib_centered=(ObjValue_upwind-ObjValue_downwind)./((calibstepup-calibstepdown)');

        % Sensitivity matrix of Jorgensen (2023) - Sensitivity to Calibrated Parameters
        estsummary.sensitivitytocalibrationmatrix=SensitivityMatrix*Jcalib_centered; % This is the formula in Corollary 1 of Jorgensen (2023)

        estsummary.doublechecks.Jcalib=Jcalib_centered;
        % also, just so user can see them
        estsummary.doublechecks.Jcalib_up=Jcalib_up;
        estsummary.doublechecks.Jcalib_down=Jcalib_down;
        estsummary.doublechecks.calibparamsvec=calibparamvec;
        estsummary.doublechecks.calibparamsvecup=calibstepup;
        estsummary.doublechecks.calibparamsvecdown=calibstepdown;
    end

end


%% Clean up the first two outputs
% EstimParams: the estimated parameters (an omitted parameter in full, its fixed entries from the mask); a per-type parameter as a structure
% over the types, or as a matrix with N_i as its first or second dimension when it was given that way. estsummary.EstimParamsStdDev: the
% asymptotic standard deviations, in the same layout (NaN at the fixed entries of an omitted parameter). EstimParamsConfInts: [lower, upper],
% one row per entry of the parameter (so 1-by-2 for a scalar), estimate -/+ z times the standard deviation; for a per-type parameter a
% structure over the types whichever way the parameter was given. (Before 2026-10-07 the matrix form filled the wrong type's row and the
% confidence interval of a vector-valued parameter errored.)
if estimoptions.bootstrapStdErrors==0
    estimparamscovarmatrix_diag=diag(estimparamscovarmatrix); % Just the diagonal of the covar matrix of the parameter vector (J and Sigma are on the model parameters)
end
EstimParams=struct();
EstimParamsConfInts=struct();
blockval=cell(nEstimParams,1); % the value of each block (a parameter, or a parameter of one type), as a column
blocksd=cell(nEstimParams,1); % and its standard deviations, NaN at the fixed entries of an omitted parameter
for pp=1:nEstimParams
    pname=EstimParamNames{nEstimParamsFinder(pp,1)};
    ii=nEstimParamsFinder(pp,2); % the type (0: common to the types)
    if estimoptions.skipestimation==0
        % Note: estimparamsvec was already switched back to the original (constrained) values further above
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
    if estimoptions.bootstrapStdErrors==0
        sdblock=sqrt(estimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        if estimomitparams_counter(pp)>0
            mask=estimomitparamsmatrix(:,sum(estimomitparams_counter(1:pp)));
            sdfull=nan(size(mask));
            sdfull(isnan(mask))=sdblock;
        else
            sdfull=sdblock;
        end
        sdfull=sdfull(:);
        blocksd{pp}=sdfull;
        if ii==0
            estsummary.EstimParamsStdDev.(pname)=sdfull;
        elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==0
            estsummary.EstimParamsStdDev.(pname).(Names_i{ii})=sdfull;
        elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==1
            if isfield(estsummary,'EstimParamsStdDev') && isfield(estsummary.EstimParamsStdDev,pname)
                temp=estsummary.EstimParamsStdDev.(pname);
            else
                temp=zeros(N_i,length(sdfull));
            end
            temp(ii,:)=sdfull';
            estsummary.EstimParamsStdDev.(pname)=temp;
        elseif nEstimParams_PTypeMatrix(nEstimParamsFinder(pp,1))==2
            if isfield(estsummary,'EstimParamsStdDev') && isfield(estsummary.EstimParamsStdDev,pname)
                temp=estsummary.EstimParamsStdDev.(pname);
            else
                temp=zeros(length(sdfull),N_i);
            end
            temp(:,ii)=sdfull;
            estsummary.EstimParamsStdDev.(pname)=temp;
        end
    elseif estimoptions.bootstrapStdErrors==1
        estsummary.EstimParamsStdDev=EstimParamsBootStrapDist;
        estsummary.notes.bootstrap=['Standard errors report distribution of parameter estimates based on ',num2str(estimoptions.numbootstrapsims),' bootstraps, each had ',num2str(estimoptions.numberinvidualsperbootstrapsim),' individuals'];
    end
end

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
EstimParamsConfInts.notes=['These are ',num2str(estimoptions.confidenceintervals),'-percent confidence intervals: [lower, upper], one row per entry of the parameter (a structure over the types for a per-type parameter)'];
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

%%
clear estimparamsvec % I modified it, so want to make sure I don't accidently use it again later

%% Give various outputs
estsummary.variousmatrices.W=WeightingMatrix; % This is just a duplicate of the input, but I figure it is handy to keep in same place as the rest of estimation results

estsummary.objectivefnval=fval;
estsummary.notes.objectivefnval='The objective function value is the value of (M_d-M_m)''W(M_d-M_m) at the estimate (recomputed from currentmomentvec and targetmomentvec, so the same whichever algorithm found the estimate).';
if estimoptions.skipestimation==1
    estsummary.warningskipestimation='Warning: this estimation used estimoptions.skipestimation=1 (all good, just reminding you as you need to be careful when using skipestimation=1 :)';
end

if estimoptions.bootstrapStdErrors==0 % Depends on derivatives, so cannot do when bootstrapping the standard errors
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
end

if estimoptions.previousiterations.niters>0  % If there were any previous iterations (using estimoptions.iterateGMM) then get that output (I hid them in estimoptions)
    for ii=1:estimoptions.previousiterations.niters
        estsummary.iterateGMM.(['iteration',num2str(ii)]).estimparams=estimoptions.(['storeiter',num2str(ii)]).estimparams;
        estsummary.iterateGMM.(['iteration',num2str(ii)]).CoVarMatrixSimMoments=estimoptions.(['storeiter',num2str(ii)]).CoVarMatrixSimMoments;
    end
end








end