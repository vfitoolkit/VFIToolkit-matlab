function [EstimParams, EstimParamsConfInts,estsummary]=EstimateLifeCycleModel_MethodOfMoments(EstimParamNames,TargetMoments,WeightingMatrix,CoVarMatrixDataMoments,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, estimoptions, vfoptions,simoptions)
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
if ~isfield(estimoptions,'toleranceparams')
    estimoptions.toleranceparams=10^(-4); % tolerance accuracy of the calibrated parameters
end
if ~isfield(estimoptions,'toleranceobjective')
    estimoptions.toleranceobjective=10^(-6); % tolerance accuracy of the objective function
end
if ~isfield(estimoptions,'fminalgo')
    estimoptions.fminalgo=4; % CMA-ES by default, I tried fminsearch() by default but it regularly fails to converge to a decent solution
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



%% Setup for which parameters are being estimated

% Backup the parameter constraint names, so I can replace them with vectors
estimoptions.constrainpositivenames=estimoptions.constrainpositive;
estimoptions.constrainpositive=zeros(length(EstimParamNames),1); % if equal 1, then that parameter is constrained to be positive
estimoptions.constrain0to1names=estimoptions.constrain0to1;
estimoptions.constrain0to1=zeros(length(EstimParamNames),1); % if equal 1, then that parameter is constrained to be 0 to 1
estimoptions.constrainAtoBnames=estimoptions.constrainAtoB;
estimoptions.constrainAtoB=zeros(length(EstimParamNames),1); % if equal 1, then that parameter is constrained to be 0 to 1
if ~isempty(estimoptions.constrainAtoBnames)
    estimoptions.constrainAtoBlimitsnames=estimoptions.constrainAtoBlimits;
    estimoptions.constrainAtoBlimits=zeros(length(EstimParamNames),2); % rows are parameters, column is lower (A) and upper (B) bounds [row will be [0,0] is unconstrained]
end

% Sometimes we want to omit parameters
if isfield(estimoptions,'omitestimparam')
    OmitEstimParamsNames=fieldnames(estimoptions.omitestimparam);
else
    OmitEstimParamsNames={''};
end
estimparamsvec0=[]; % column vector
estimparamsvecindex=zeros(length(EstimParamNames)+1,1); % Note, first element remains zero
estimomitparams_counter=zeros(length(EstimParamNames)); % column vector: estimomitparamsvec allows omiting the parameter for certain ages
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

    % If the parameter is constrained in some way then we need to transform it

    % First, check the name, and convert it if relevant
    if any(strcmp(estimoptions.constrainpositivenames,EstimParamNames{pp}))
        estimoptions.constrainpositive(pp)=1;
    end
    if any(strcmp(estimoptions.constrain0to1names,EstimParamNames{pp}))
        estimoptions.constrain0to1(pp)=1;
    end
    if any(strcmp(estimoptions.constrainAtoBnames,EstimParamNames{pp}))
        % For parameters A to B, I convert via 0 to 1
        estimoptions.constrain0to1(pp)=1;
        estimoptions.constrainAtoB(pp)=1;
        estimoptions.constrainAtoBlimits(pp,:)=estimoptions.constrainAtoBlimitsnames.(EstimParamNames{pp});
    end

    if estimoptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=max(log(estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))),-49.99);
        % Note, the max() is because otherwise p=0 returns -Inf. [Matlab evaluates exp(-50) as about 10^-22, I overrule and use exp(-50) as zero, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if estimoptions.constrainAtoB(pp)==1
        % Constraint parameter to be A to B (by first converting to 0 to 1, and then treating it as contraint 0 to 1)
        estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=(estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))-estimoptions.constrainAtoBlimits(pp,1))/(estimoptions.constrainAtoBlimits(pp,2)-estimoptions.constrainAtoBlimits(pp,1));
        % x=(y-A)/(B-A), converts A-to-B y, into 0-to-1 x
        % And then the next if-statement converts this 0-to-1 into unconstrained
    end
    if estimoptions.constrain0to1(pp)==1
        % Constrain parameter to be 0 to 1 (be working with log(p/(1-p)), where p is parameter) then always take exp()/(1+exp()) before inputting to model
        estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=min(49.99,max(-49.99,  log(estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))/(1-estimparamsvec0(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))) ));
        % Note: the max() and min() are because otherwise p=0 or 1 returns -Inf or Inf [Matlab evaluates 1/(1+exp(-50)) as one, and 1/(1+exp(50)) as about 10^-22, so I overrule them as 1 and 0, so I set -49.99 here so solver can realise the boundary is there; not sure if this setting -49.99 instead of my -50 cutoff actually helps, but seems like it might so I have done it here].
    end
    if estimoptions.constrainpositive(pp)==1 && estimoptions.constrain0to1(pp)==1 % Double check of inputs
        fprinf(['Relating to following error message: Parameter ',num2str(pp),' of ',num2str(length(EstimParamNames))])
        error('You cannot constrain parameter twice (you are constraining one of the parameters using both estimoptions.constrainpositive and estimoptions.constrain0to1')
    end
end


%% Setup for which moments are being targeted
% Only calculate each of AllStats and LifeCycleProfiles when being used (so as faster when not using both)
if isfield(TargetMoments,'AllStats')
    usingallstats=1;
else
    usingallstats=0;
end
if isfield(TargetMoments,'AgeConditionalStats')
    usinglcp=1;
else
    usinglcp=0;
end

% NEED TO ADD A CHECK THAT THE INPUT TARGETS ARE THE CORRECT SIZES!!!


% Get all of the moments out of TargetMoments and make them into a vector
% Also, store all the names
targetmomentvec=[]; % Can't preallocate as have no idea how big this will be
% Ends up a colmumn vector

% First, do those in AllStats
if usingallstats==1
    allstatmomentnames={};
    allstatmomentcounter=0;
    allstatmomentsizes=0;
    if isfield(TargetMoments,'AllStats')
        a1vec=fieldnames(TargetMoments.AllStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AllStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                allstatmomentcounter=allstatmomentcounter+1;
                if size(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                    targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})]; % append to end
                else
                    targetmomentvec=[targetmomentvec; TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                end
                allstatmomentnames(allstatmomentcounter,:)={a1vec{a1},a2vec{a2}};
                allstatmomentsizes(allstatmomentcounter)=length(TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2}));
                % Note: PType will require an extra possible level of depth where this is still a structure (for ptype specific moments)
            end
        end
    end
    allstatcummomentsizes=cumsum(allstatmomentsizes); % Note: this is zero is AllStats is unused
    % To do AllStats faster, we use simoptions.whichstats so that we only compute the stats we want.
    AllStats_whichstats=zeros(7,1);
    if any(strcmp(allstatmomentnames(:,2),'Mean'))
        AllStats_whichstats(1)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'Median'))
        AllStats_whichstats(2)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'RatioMeanToMedian'))
        AllStats_whichstats(1)=1;
        AllStats_whichstats(2)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'Variance')) || any(strcmp(allstatmomentnames(:,2),'StdDeviation'))
        AllStats_whichstats(3)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'LorenzCurve')) || any(strcmp(allstatmomentnames(:,2),'Gini'))
        AllStats_whichstats(4)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'Maximum')) || any(strcmp(allstatmomentnames(:,2),'Minimum'))
        AllStats_whichstats(5)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'QuantileCutoffs')) || any(strcmp(allstatmomentnames(:,2),'QuantileMeans'))
        AllStats_whichstats(5)=1;
    end
    if any(strcmp(allstatmomentnames(:,2),'MoreInequality'))
        AllStats_whichstats(7)=1;
    end
else
    % Placeholders
    allstatmomentnames={};
    allstatcummomentsizes=0;
    AllStats_whichstats=zeros(7,1);
end


% Second, do those in AgeConditionalStats
if usinglcp==1
    acsmomentnames={};
    acsmomentcounter=0;
    acsmomentsizes=0;
    if isfield(TargetMoments,'AgeConditionalStats')
        a1vec=fieldnames(TargetMoments.AgeConditionalStats); % This will be the FnsToEvaluate names
        for a1=1:length(a1vec)
            a2vec=fieldnames(TargetMoments.AgeConditionalStats.(a1vec{a1}));% These will be Mean, etc
            for a2=1:length(a2vec)
                acsmomentcounter=acsmomentcounter+1;
                if size(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}),2)==1 % already column vector
                    targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})]; % append to end
                else
                    targetmomentvec=[targetmomentvec; TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})']; % transpose, then append to end
                end
                acsmomentnames(acsmomentcounter,:)={a1vec{a1},a2vec{a2}};
                acsmomentsizes(acsmomentcounter)=length(TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2}));
                % Note: PType will require an extra possible level of depth where this is still a structure (for ptype specific moments)
            end
        end
    end
    acscummomentsizes=cumsum(acsmomentsizes); % Note: this is zero is AgeConditionalStats is unused
    % To do AgeConditionalStats faster, we use simoptions.whichstats so that we only compute the stats we want.
    ACStats_whichstats=zeros(7,1);
    if any(strcmp(acsmomentnames(:,2),'Mean'))
        ACStats_whichstats(1)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'Median'))
        ACStats_whichstats(2)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'RatioMeanToMedian'))
        ACStats_whichstats(1)=1;
        ACStats_whichstats(2)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'Variance')) || any(strcmp(acsmomentnames(:,2),'StdDeviation'))
        ACStats_whichstats(3)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'LorenzCurve')) || any(strcmp(acsmomentnames(:,2),'Gini'))
        ACStats_whichstats(4)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'Maximum')) || any(strcmp(acsmomentnames(:,2),'Minimum'))
        ACStats_whichstats(5)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'QuantileCutoffs')) || any(strcmp(acsmomentnames(:,2),'QuantileMeans'))
        ACStats_whichstats(5)=1;
    end
    if any(strcmp(acsmomentnames(:,2),'MoreInequality'))
        ACStats_whichstats(7)=1;
    end
else
    % Placeholders
    acsmomentnames={};
    acscummomentsizes=0;
    ACStats_whichstats=zeros(7,1);
end



%% Now, a bunch of things to avoid redoing them every parameter vector we want to try
% Note: I avoid doing this for ReturnFnParamNames because they are so
% dependent on the setup. Same for FnsToEvaluateParamNames
ReturnFnParamNames=[];
FnsToEvaluateParamNames=[];


% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
pi_z_J=zeros(prod(n_z),prod(n_z),'gpuArray');
if isfield(vfoptions,'ExogShockFn')
    if ~isfield(vfoptions,'ExogShockFnParamNames')
        vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    end
    for jj=1:N_j
        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
        for ii=1:length(ExogShockFnParamsVec)
            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
        end
        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
        pi_z_J(:,:,jj)=gpuArray(pi_z);
        if all(size(z_grid)==[sum(n_z),1])
            z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
        else % already joint-grid
            z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
        end
    end
elseif prod(n_z)==0 % no z
    z_gridvals_J=[];
elseif ndims(z_grid)==3 % already an age-dependent joint-grid
    if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
        z_gridvals_J=z_grid;
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
    end
    pi_z_J=pi_z;
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
    z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
    z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
    pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
end

% If using e variable, do same for this
if isfield(vfoptions,'n_e')
    if prod(vfoptions.n_e)==0
        vfoptions=rmfield(vfoptions,'n_e');
    else
        if isfield(vfoptions,'e_grid_J')
            error('No longer use vfoptions.e_grid_J, instead just put the age-dependent grid in vfoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
        end
        if ~isfield(vfoptions,'e_grid') % && ~isfield(vfoptions,'e_grid_J')
            error('You are using an e (iid) variable, and so need to declare vfoptions.e_grid')
        elseif ~isfield(vfoptions,'pi_e')
            error('You are using an e (iid) variable, and so need to declare vfoptions.pi_e')
        end

        vfoptions.e_gridvals_J=zeros(prod(vfoptions.n_e),length(vfoptions.n_e),'gpuArray');
        vfoptions.pi_e_J=zeros(prod(vfoptions.n_e),prod(vfoptions.n_e),'gpuArray');
        if isfield(vfoptions,'EiidShockFn')
            if ~isfield(vfoptions,'EiidShockFnParamNames')
                vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
            end
            for jj=1:N_j
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                vfoptions.pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                    vfoptions.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                else % already joint-grid
                    vfoptions.e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                end
            end
        elseif ndims(vfoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e),N_j])
                vfoptions.e_gridvals_J=vfoptions.e_grid;
            end
            vfoptions.pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),N_j]) % age-dependent stacked-grid
            for jj=1:N_j
                vfoptions.e_gridvals_J(:,:,jj)=CreateGridvals(vfoptions.n_e,vfoptions.e_grid(:,jj),1);
            end
            vfoptions.pi_e_J=vfoptions.pi_e;
        elseif all(size(vfoptions.e_grid)==[prod(vfoptions.n_e),length(vfoptions.n_e)]) % joint grid
            vfoptions.e_gridvals_J=vfoptions.e_grid.*ones(1,1,N_j,'gpuArray');
            vfoptions.pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        elseif all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1]) % basic grid
            vfoptions.e_gridvals_J=CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
            vfoptions.pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        end
    end
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
EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptions, vfoptions,simoptions);


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
    end

else % estimoptions.skipestimation==1    
    warning('Skipping the estimation step (you have set estimoptions.skipestimation=1 in EstimateLifeCycleModel_MethodOfMoments() [Nothing wrong with this, just warning as want to be sure you did this on purpose]')
    % The values in Parameters are taken as the estimated values for EstimParams
    % Note that we already got these as estimparamsvec0, so we can just set
    estimparamsvec=estimparamsvec0;
end



%% Two-iteration efficient GMM (actually, n-iteration, but just uses this recursively)
if estimoptions.iterateGMM>1 && estimoptions.skipestimation==0
    error('HAVE NOT YET IMPLEMENTED ITERATED GMM (you have estimoptions.iterateGMM>1)')
end


%% Compute the standard deviation of the estimated parameters
if estimoptions.bootstrapStdErrors==0
    % First, need the Jacobian matrix, which involves computing all the
    % derivatives of the individual moments with respect to the estimated parameters

    estimoptions.vectoroutput=1; % Was set to zero to get point estimates, now set to one as part of computing std deviations.
    % To change the estimoptions, we have to reset EstimateMoMObjectiveFn
    EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);

    % According to https://en.wikipedia.org/wiki/Numerical_differentiation#Step_size
    % A good step size to compute the derivative of f(x) is epsilon*x with
    epsilonraw=sqrt(2.2)*10^(-8); % Note: this is sqrt(eps(1.d0)), the eps() is Matlab command that gives floating point precision
    % I am going to compute the upper and lower first differences
    % I then use the smallest of the two (as that gives the larger/more conservative, standard deviations)

    % Decided to actually do four different values of epsilon, then report J
    % for all so user can see how they look (are the derivatives sensitive to epsilon)
    epsilonmodvec=[1,10^2,10^4,10^6];
    % Default value of epsilon
    eedefault=3; % Default epsilon value is epsilonraw*epsilonmodvec(eedefault)

    %% We want to calculate derivatives from epsilon changes in the model parameters
    % I want to do epsilon change in the model parameter, but here I have
    % the unconstrained parameters. So I create an epsilonparamup and
    % epsilonparamdown, which contain the unconstrained values the
    % correspond to espilon changes in the constrained parameters
    % I do this in a separate loop, which is a loss of runtime, but this is
    % minor and is much easier to read so whatever
    epsilonparamup=zeros(length(estimparamsvec),length(epsilonmodvec));
    epsilonparamdown=zeros(length(estimparamsvec),length(epsilonmodvec));
    modelestimparamsvec=estimparamsvec;
    modelestimparamsvecup=zeros(size(modelestimparamsvec));
    modelestimparamsvecdown=zeros(size(modelestimparamsvec));
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
    % Now, multiply by (1+-epsilon) and then convert back to unconstrained parameter value
    for ee=1:length(epsilonmodvec)
        epsilon=epsilonmodvec(ee)*epsilonraw;
        for pp=1:length(EstimParamNames)
            if modelestimparamsvec(pp)>10*epsilon
                modelestimparamsvecup(pp)=(1+epsilon)*modelestimparamsvec(pp); % add epsilon*x to the pp-th parameter
                modelestimparamsvecdown(pp)=(1-epsilon)*modelestimparamsvec(pp); % subtract epsilon*x from the pp-th parameter
            else % is the modelestimparamsvec itself is tiny, then actually just add/subtract epsilon to/from x [have to do this for x=0, and this seems a reasonable cutoff]
                modelestimparamsvecup(pp)=epsilon+modelestimparamsvec(pp); % add epsilon to the pp-th parameter
                modelestimparamsvecdown(pp)=-epsilon+modelestimparamsvec(pp); % subtract epsilon from the pp-th parameter
            end
        end
        % Switch to the unconstrained
        if estimoptions.constrainpositive(pp)==1
            % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
            modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=max(log(modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))),-10^3);
            modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=max(log(modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))),-10^3);
            % Note, the max() is because otherwise p=0 returns -Inf. [Matlab evaluates exp(-10^3) as zero]
        end
        if estimoptions.constrainAtoB(pp)==1
            % Constraint parameter to be A to B (by first converting to 0 to 1, and then treating it as contraint 0 to 1)
            modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=(modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))-estimoptions.constrainAtoBlimits(pp,1))/(estimoptions.constrainAtoBlimits(pp,2)-estimoptions.constrainAtoBlimits(pp,1));
            modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=(modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))-estimoptions.constrainAtoBlimits(pp,1))/(estimoptions.constrainAtoBlimits(pp,2)-estimoptions.constrainAtoBlimits(pp,1));
            % x=(y-A)/(B-A), converts A-to-B y, into 0-to-1 x
            % And then the next if-statement converts this 0-to-1 into unconstrained
        end
        if estimoptions.constrain0to1(pp)==1
            % Constrain parameter to be 0 to 1 (be working with log(p/(1-p)), where p is parameter) then always take exp()/(1+exp()) before inputting to model
            modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=min(50,max(-50,  log(modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))/(1-modelestimparamsvecup(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))) ));
            modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=min(50,max(-50,  log(modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))/(1-modelestimparamsvecdown(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))) ));
            % Note: the max() and min() are because otherwise p=0 or 1 returns -Inf or Inf [Matlab evaluates 1/(1+exp(-50)) as one, and 1/(1+exp(50)) as about 10^-22.
        end
        % Store the epsilon parameters
        epsilonparamup(:,ee)=modelestimparamsvecup;
        epsilonparamdown(:,ee)=modelestimparamsvecdown;

    end

    %% Can now calculate derivatives to the epsilon change in parameters as the finite-difference
    for ee=1:length(epsilonmodvec)
        epsilon=epsilonmodvec(ee)*epsilonraw; % Not actually used for anything (as I used it to create epsilonparamup and epsilonparamdown, and these are used below)

        % ObjValue=zeros(sum(~isnan(targetmomentvec)),1);
        ObjValue_upwind=zeros(sum(~isnan(targetmomentvec)),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'
        ObjValue_downwind=zeros(sum(~isnan(targetmomentvec)),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'
        % J=zeros(sum(~isnan(targetmomentvec)),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'

        % Note: estimoptions.vectoroutput=1, so ObjValue is a vector
        ObjValue=CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
        for pp=1:length(estimparamsvec)
            epsilonparamvec=estimparamsvec;
            epsilonparamvec(pp)=epsilonparamup(pp,ee); % add epsilon*x to the pp-th parameter
            ObjValue_upwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
            epsilonparamvec(pp)=epsilonparamdown(pp,ee); % subtract epsilon*x from the pp-th parameter
            ObjValue_downwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
        end

        % Use finite-difference to compute the derivatives
        J_up=(ObjValue_upwind-ObjValue)./((epsilonparamup(:,ee)-estimparamsvec)');
        J_down=(ObjValue-ObjValue_downwind)./((estimparamsvec-epsilonparamdown(:,ee))');
        J_centered=(ObjValue_upwind-ObjValue_downwind)./((epsilonparamup(:,ee)-epsilonparamdown(:,ee))');
        % Jacobian matix of derivatives of model moments with respect to parameters, evaluated at the parameter point estimates
        
        % I decided to use the centered finite difference as my default for the derivative
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J=J_centered;
        % Store some other stuff as well
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J_up=J_up;
        estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).J_down=J_down;
        if estimoptions.efficientW==0
            % This is standard formula for the asymptotic variance of method of moments estimator
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma=((J_centered'*WeightingMatrix*J_centered)^(-1)) * J_centered'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_centered * ((J_centered'*WeightingMatrix*J_centered)^(-1));
            % report the up and down derivatives too
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_up=((J_up'*WeightingMatrix*J_up)^(-1)) * J_up'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_up * ((J_up'*WeightingMatrix*J_up)^(-1));
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_down=((J_down'*WeightingMatrix*J_down)^(-1)) * J_down'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J_down * ((J_down'*WeightingMatrix*J_down)^(-1));
        elseif estimoptions.efficientW==1
            % When using the efficient weighting matrix W=Omega^(-1), the asymptotic variance of the method of moments estimator simplifies to
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma=(J_centered'*WeightingMatrix*J_centered)^(-1);
            % report the up and down derivatives too
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_up=(J_up'*WeightingMatrix*J_up)^(-1);
            estsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).Sigma_down=(J_down'*WeightingMatrix*J_down)^(-1);
        end
        if ee==eedefault
            J=J_centered; % This is the one used to report Sigma (parameter std deviations) [corresponds to epsilon=sqrt(2.2)*10^(-4)]
        end
    end
    
    % For later
    epsilon=epsilonmodvec(eedefault)*epsilonraw; % sqrt(2.2)*10^(-4)
    % What I have here as default uses epsilon of the order 10^(-4)
    % Grey Gordon's numerical derivative code used 10^(-6)

    if estimoptions.efficientW==0
        estimparamscovarmatrix=((J'*WeightingMatrix*J)^(-1)) * J'*WeightingMatrix*CoVarMatrixDataMoments*WeightingMatrix*J * ((J'*WeightingMatrix*J)^(-1));
        % This is standard formula for the asymptotic variance of method of moments estimator
        % See, e.g., Kirkby - "Classical (not Simulated!) Method of Moments Estimation of Life-Cycle Models"
    elseif estimoptions.efficientW==1
        % When using the efficient weighting matrix W=Omega^(-1), the asymptotic variance of the method of moments estimator simplifies to
        estimparamscovarmatrix=(J'*WeightingMatrix*J)^(-1);
    end
end




%% Bootstrap standard errors
if estimoptions.bootstrapStdErrors==1
    error('HAVE NOT YET IMPLEMENTED BOOTSTRAP STANDARD ERRORS (you have estimoptions.bootstrapStdErrors=1)')
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
    
    % Sensitivity of estimated paraemters to the pre-calibrated parameters
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

        for pp=1:length(estimoptions.CalibParamsNames)
            Parameters.(estimoptions.CalibParamsNames{pp})=(1+epsilon)*CalibParams.(estimoptions.CalibParamsNames{pp}); % add epsilon*x to the pp-th parameter
            ObjValue_upwind(:,pp)=CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
            Parameters.(estimoptions.CalibParamsNames{pp})=(1-epsilon)*CalibParams.(estimoptions.CalibParamsNames{pp}); % subtract epsilon*x from the pp-th parameter
            ObjValue_downwind(:,pp)=CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
            % restore calib param
            Parameters.(estimoptions.CalibParamsNames{pp})=CalibParams.(estimoptions.CalibParamsNames{pp});
        end
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
        EstimParams.(EstimParamNames{pp})=estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1));
    else
        EstimParams.(EstimParamNames{pp})=Parameters.(EstimParamNames{pp}); % When skipping estimation, just returns the same parameters as you input
    end

    if estimoptions.bootstrapStdErrors==0
        estimparamscovarmatrix_diag=diag(estimparamscovarmatrix); % Just the diagonal of the covar matrix of the parameter vector
        if estimoptions.constrainpositive(pp)==0
            estsummary.EstimParamsStdDev.(EstimParamNames{pp})=sqrt(estimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        elseif estimoptions.constrainpositive(pp)==1
            % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
            estsummary.EstimParamsStdDev.(EstimParamNames{pp})=exp(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))+estimparamscovarmatrix_diag(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)))-exp(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
        end
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

if estimoptions.skipestimation==0
    estsummary.objectivefnval=fval;
    estsummary.notes.objectivefnval='The objective function value is the value of (M_d-M_m)W(M_d-M_m).';
else
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
end

if estimoptions.previousiterations.niters>0  % If there were any previous iterations (using estimoptions.iterateGMM) then get that output (I hid them in estimoptions)
    for ii=1:estimoptions.previousiterations.niters
        estsummary.iterateGMM.(['iteration',num2str(ii)]).estimparams=estimoptions.(['storeiter',num2str(ii)]).estimparams;
        estsummary.iterateGMM.(['iteration',num2str(ii)]).CoVarMatrixSimMoments=estimoptions.(['storeiter',num2str(ii)]).CoVarMatrixSimMoments;
    end
end








end