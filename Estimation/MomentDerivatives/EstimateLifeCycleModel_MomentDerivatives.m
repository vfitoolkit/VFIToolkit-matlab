function [MomentDerivatives,SortedMomentDerivatives,momentderivsummary]=EstimateLifeCycleModel_MomentDerivatives(EstimParamNames, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, estimoptions, vfoptions,simoptions)
% Compute derivatives of model moments with respect to the 'estimated parameters'
%
% dM_m(theta)/dtheta
%
% M_m are all the moments the toolkit does with 'AllStats' and 'LifeCycleProfiles'
% theta is a vector of parameters, EstimParamNames, based on values in Params
%
% Most of this is copy-past from EstimateLifeCycleModel_MethodOfMoments(), hence the 
% naming of things is a bit odd for the current purpose.
%

%% Setup estimoptions
estimoptions.vectoroutput=1; % Set to one as part of computing std deviations.
estimoptions.verbose=0; % if you set it to one, then it will print out every moment which is just stupid

if ~isfield(estimoptions,'verbose')
    estimoptions.verbose=0;
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
estimoptions.simulatemoments=0; % Set to zero to get point estimates, is needed later if you bootstrap standard errors

if ~isfield(simoptions,'npoints')
    simoptions.npercentiles=100; % follow defaults elsewhere
end
if ~isfield(simoptions,'nquantiles')
    simoptions.nquantiles=20; % follow defaults elsewhere
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
% Do all possible moments
usingallstats=1;
usinglcp=1;

% Go through FnsToEvaluate to fill in all the names, and then put all the possible moment-types into this as well.
fnnames=fieldnames(FnsToEvaluate);
% First, do those in AllStats
allstatmomentnames={};
allstatmomentcounter=0;
allstatmomentsizes=0;
% Second, do those in AgeConditionalStats
acsmomentnames={};
acsmomentcounter=0;
acsmomentsizes=0;
for ff=1:length(fnnames)
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'Mean'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'Median'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'RatioMeanToMedian'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'Variance'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'StdDeviation'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=simoptions.npercentiles;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'LorenzCurve'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'Gini'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'Maximum'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'Minimum'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=simoptions.nquantiles+1;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'QuantileCutoffs'};
    allstatmomentcounter=allstatmomentcounter+1;
    allstatmomentsizes(allstatmomentcounter)=simoptions.nquantiles;
    allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'QuantileMeans'};
    % allstatmomentcounter=allstatmomentcounter+1;
    % allstatmomentsizes(allstatmomentcounter)=1;
    % allstatmomentnames(allstatmomentcounter,:)={fnnames{ff},'MoreInequality'};


    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'Mean'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'Median'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'RatioMeanToMedian'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'Variance'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'StdDeviation'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j*simoptions.npercentiles;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'LorenzCurve'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'Gini'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'Maximum'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'Minimum'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j*(simoptions.nquantiles+1);
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'QuantileCutoffs'};
    acsmomentcounter=acsmomentcounter+1;
    acsmomentsizes(acsmomentcounter)=N_j*simoptions.nquantiles;
    acsmomentnames(acsmomentcounter,:)={fnnames{ff},'QuantileMeans'};
    % acsmomentcounter=acsmomentcounter+1;
    % acsmomentsizes(acsmomentcounter)=N_j;
    % acsmomentnames(acsmomentcounter,:)={fnnames{ff},'MoreInequality'};

end
allstatcummomentsizes=cumsum(allstatmomentsizes); % Note: this is zero is AllStats is unused
acscummomentsizes=cumsum(acsmomentsizes); % Note: this is zero is AllStats is unused
% To do AllStats faster, we use simoptions.whichstats so that we only compute the stats we want.
AllStats_whichstats=ones(7,1);
ACStats_whichstats=ones(7,1);



%% Set up a fake targetmomentvec, only thing it ends up doing below is determining size of some vectors, but we need a placebo as input to EstimateLifeCycleModel_MomentDerivatives
targetmomentvec=zeros(allstatcummomentsizes(end)+acscummomentsizes(end),1);


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
    if isfield(vfoptions,'ExogShockFnParamNames')
        for jj=1:N_j
            ExogShockFnParamsVec=CreateVectorFromParams(Params, vfoptions.ExogShockFnParamNames,jj);
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
    else
        for jj=1:N_j
            [z_grid,pi_z]=vfoptions.ExogShockFn(N_j);
            pi_z_J(:,:,jj)=gpuArray(pi_z);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
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
            if isfield(vfoptions,'EiidShockFnParamNames')
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Params, vfoptions.EiidShockFnParamNames,jj);
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
            else
                for jj=1:N_j
                    [vfoptions.e_grid,vfoptions.pi_e]=vfoptions.EiidShockFn(N_j);
                    vfoptions.pi_e_J(:,jj)=gpuArray(vfoptions.pi_e);
                    if all(size(vfoptions.e_grid)==[sum(vfoptions.n_e),1])
                        vfoptions.e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(vfoptions.n_e,vfoptions.e_grid,1));
                    else % already joint-grid
                        vfoptions.e_gridvals_J(:,:,jj)=gpuArray(vfoptions.e_grid,1);
                    end
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
% estimoptions.logmoments can be specified by names
if isstruct(estimoptions.logmoments)
    error('estimoptions.logmoments can normally be names, but in EstimateLifeCycleModel_MomentDerivatives is must be scalar 0 or 1')
elseif any(estimoptions.logmoments>0) % =1 means log of moments (can be set up as vector, zeros(length(EstimParamNames),1)
   % If set this up, and then set up 
   if isscalar(estimoptions.logmoments)
       estimoptions.logmoments=ones(length(targetmomentvec),1); % log all of them
   else
       error('estimoptions.logmoments can normally be a vector, but in EstimateLifeCycleModel_MomentDerivatives is must be scalar 0 or 1')
   end
   % log of targetmoments [no need to do this as inputs should already be log()]
   % targetmomentvec=(1-estimoptions.logmoments).*targetmomentvec + estimoptions.logmoments.*log(targetmomentvec.*estimoptions.logmoments+(1-estimoptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Set up the objective function and the initial calibration parameter vector
% Note: _objectivefn is shared between Method of Moments Estimation and Calibration
EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);

%% We are not estimating, so just use initial values
estimparamsvec=estimparamsvec0;


%% Compute the moment derivatives with respect to the estimated parameters
% We want to compute J (the jacobian matrix of derivatives of model moments to the estimated parameters).
% To make it easier to compute the derivatives by finite-difference, I turn off the parameter constraints and just use the model 
% parameter values (rather than the internal-transformed-parameters) directly. Just makes it easier to follow what is going on (at least in my head).
% To faciliate this I use estimoptionsJacobian=estimoptions, but with modifications.
% Later I did some searching, and it seems there are no precise answers online, but some people (Python 'optimagic' on github) made same decision I did, of taking 
% derivatives based on 'external' parameters rather than 'internal' (transformed) parameters.
% Other open issue, what do you do when the resulting standard deviations mean confidence intervals reach outside your contraints?

% First, need the Jacobian matrix, which involves computing all the
% derivatives of the individual moments with respect to the estimated parameters
estimoptionsJacobian=estimoptions;
estimoptionsJacobian.constrainpositive=zeros(length(EstimParamNames),1); % eliminate constraints for Jacobian
estimoptionsJacobian.constrain0to1=zeros(length(EstimParamNames),1); % eliminate constraints for Jacobian
estimoptionsJacobian.constrainAtoB=zeros(length(EstimParamNames),1); % eliminate constraints for Jacobian
% Note: idea is that we don't want to apply constraints inside CalibrateLifeCycleModel_objectivefn() while computing finite-differences
estimoptionsJacobian.vectoroutput=1; % Was set to zero to get point estimates, now set to one as part of computing std deviations.

% To change the estimoptions, we have to reset EstimateMoMObjectiveFn
EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptionsJacobian, vfoptions,simoptions);

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

% For parameters of size 10^(-2) or less, use alternative epsilon values
epsilonalt=[10^(-2),10^(-2),10^(-1),10^(-1)]; % Note: this must be same length as epsilonmodvec (default follows eedefault)


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
        epsilonparamvec(pp)=epsilonparamup(pp,ee); % add epsilon*x to the pp-th parameter
        ObjValue_upwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions);
        epsilonparamvec(pp)=epsilonparamdown(pp,ee); % subtract epsilon*x from the pp-th parameter
        ObjValue_downwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimomitparams_counter, estimomitparamsmatrix, estimoptionsJacobian, vfoptions,simoptions);
    end
    epsilonparamvec=modelestimparamsvec; % and using estimoptionsJacobian, so using the actual parameters, rather than the transformed parameters

    % Use finite-difference to compute the derivatives
    J_up=(ObjValue_upwind-ObjValue)./((epsilonparamup(:,ee)-epsilonparamvec)');
    J_down=(ObjValue-ObjValue_downwind)./((epsilonparamvec-epsilonparamdown(:,ee))');
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

    % Main results on the derivatives, calculated based on finite-differences
    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference=J_full;
    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference_centered=J_centered;
    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference_up=J_up;
    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference_down=J_down;
   
    % Add a bunch of extra info
    momentderivsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvec=epsilonparamvec; % Is actually independent of ee anyway
    momentderivsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvecup=epsilonparamup(:,ee);
    momentderivsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).estimparamsvecdown=epsilonparamdown(:,ee);
    momentderivsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).violateconstraintbottom=violateconstraintbottom;
    momentderivsummary.doublechecks.(['epsilon',num2str(epsilonmodvec(ee))]).violateconstrainttop=violateconstrainttop;
    % if ee==eedefault
    %     MomentDerivativesMatrix=J_full; % This is the main reported calculation of the derivatives
    % end
end


%% Clean up output
MomentDerivativesMatrix=momentderivsummary.(['epsilon',num2str(epsilonmodvec(eedefault))]).FiniteDifference;
% My derivatives of choice use epsilon=sqrt(2.2)*10^(-4)
% They are computed as the centered finite difference

MomentDerivatives=struct(); % Put them in a structure so can be made sense of by user
for pp=1:length(EstimParamNames)
    MomentDerivatives.(['wrt_',EstimParamNames{pp}]).AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2})=MomentDerivativesMatrix(1:allstatcummomentsizes(1),pp);
    for cc=2:size(allstatmomentnames,1)
        MomentDerivatives.(['wrt_',EstimParamNames{pp}]).AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2})=MomentDerivativesMatrix(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc),pp);
    end
    MomentDerivatives.(['wrt_',EstimParamNames{pp}]).AgeConditionalStats.(acsmomentnames{1,1}).(acsmomentnames{1,2})=MomentDerivativesMatrix(allstatcummomentsizes(end)+1:allstatcummomentsizes(end)+acscummomentsizes(1),pp);
    for cc=2:size(acsmomentnames,1)
        MomentDerivatives.(['wrt_',EstimParamNames{pp}]).AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2})=MomentDerivativesMatrix(allstatcummomentsizes(end)+acscummomentsizes(cc-1)+1:allstatcummomentsizes(end)+acscummomentsizes(cc),pp);
    end
end

% Note that 'summary' contains the same derivatives, computed for
% alternative values of epsilon, as well as upwind, downwind, and centered
% finite differences.

% SortedMomentDerivatives is just the same content as MomentDerivatives,
% but sort them by the magnitude of the derivatives, instead of just
% unsorted
SortedMomentDerivatives=struct(); 
% NOT YET IMPLEMENTED




end

