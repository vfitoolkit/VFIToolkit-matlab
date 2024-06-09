function [MomentDerivatives,SortedMomentDerivatives,momentderivsummary]=EstimateLifeCycleModel_MomentDerivatives(EstimParamNames, n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, estimoptions, vfoptions,simoptions)
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
    estimoptions.constrainpositive=zeros(length(EstimParamNames),1); % if equal 1, then that parameter is constrained to be positive
end
if ~isfield(estimoptions,'constrain0to1')
    estimoptions.constrain0to1=zeros(length(EstimParamNames),1); % if equal 1, then that parameter is constrained to be 0 to 1 [I want to later modify to also allow setting min and max bounds, but this is not yet implemented]
end
if ~isfield(estimoptions,'logmoments')
    estimoptions.logmoments=0; % =1 means log of moments (can be set up as vector, zeros(length(EstimParamNames),1)
    % Note: the input target moment should be the raw moment, log() will be taken internally (don't input the log(moment))
end
estimoptions.simulatemoments=0; % Set to zero to get point estimates, is needed later if you bootstrap standard errors

if ~isfield(simoptions,'npoints')
    simoptions.npercentiles=100; % follow defaults elsewhere
end
if ~isfield(simoptions,'nquantiles')
    simoptions.nquantiles=20; % follow defaults elsewhere
end


%% Setup for which parameters are being estimated
estimparamsvec=[]; % column vector
estimparamsvecindex=zeros(length(EstimParamNames)+1,1); % Note, first element remains zero
for pp=1:length(EstimParamNames)
    % Get all the parameters
    if size(Parameters.(EstimParamNames{pp}),2)==1
        estimparamsvec=[estimparamsvec; Parameters.(EstimParamNames{pp})];
    else
        estimparamsvec=[estimparamsvec; Parameters.(EstimParamNames{pp})']; % transpose
    end
    estimparamsvecindex(pp+1)=estimparamsvecindex(pp)+length(Parameters.(EstimParamNames{pp}));
    
    % If the parameter is constrained in some way then we need to transform it
    if estimoptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=log(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1)));
    end
    if estimoptions.constrain0to1(pp)==1
        % Constrain parameter to be 0 to 1 (be working with log(p/(1-p)), where p is parameter) then always take exp()/(1+exp()) before inputting to model
        estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))=log(estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))/(1-estimparamsvec(estimparamsvecindex(pp)+1:estimparamsvecindex(pp+1))));
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
% estimoptions.logmoments will either be scalar, or a vector of zeros and ones
%    [scalar of zero is interpreted as vector of zeros, scalar of one is interpreted as vector of ones]
if any(estimoptions.logmoments>0) % =1 means log of moments (can be set up as vector, zeros(length(EstimParamNames),1)
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
   % % User should have inputted the moments themselves, not the logs
   % % I would like to throw an error/warning if input the log(moment) but cannot think of any good way to detect this.
   % % log of targetmoments 
   % targetmomentvec=(1-estimoptions.logmoments).*targetmomentvec + estimoptions.logmoments.*log(targetmomentvec.*estimoptions.logmoments+(1-estimoptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Set up the objective function and the initial calibration parameter vector
% Note: _objectivefn is shared between Method of Moments Estimation and Calibration
EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);



%% Compute the moment derivatives with respect to the estimated parameters
% First, need the Jacobian matrix, which involves computing all the
% derivatives of the individual moments with respect to the estimated parameters

estimoptions.vectoroutput=1; % Was set to zero to get point estimates, now set to one as part of computing std deviations.
% To change the estimoptions, we have to reset EstimateMoMObjectiveFn
EstimateMoMObjectiveFn=@(estimparamsvec) CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Params, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);

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

for ee=1:4
    epsilon=epsilonmodvec(ee)*epsilonraw;

    % ObjValue=zeros(length(targetmomentvec),1);
    ObjValue_upwind=zeros(length(targetmomentvec),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'
    ObjValue_downwind=zeros(length(targetmomentvec),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'
    % J=zeros(length(targetmomentvec),length(estimparamsvec)); % Jacobian matrix of 'derivative of model moments with respect to parameters, evaluated at parameter point estimates'

    % Note: estimoptions.vectoroutput=1, so ObjValue is a vector
    ObjValue=CalibrateLifeCycleModel_objectivefn(estimparamsvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
    for pp=1:length(estimparamsvec)
        epsilonparamvec=estimparamsvec;
        epsilonparamvec(pp)=(1+epsilon)*estimparamsvec(pp); % add epsilon*x to the pp-th parameter
        ObjValue_upwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
        epsilonparamvec(pp)=(1-epsilon)*estimparamsvec(pp); % subtract epsilon*x from the pp-th parameter
        ObjValue_downwind(:,pp)=CalibrateLifeCycleModel_objectivefn(epsilonparamvec,EstimParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, estimparamsvecindex, estimoptions, vfoptions,simoptions);
    end
    FiniteDifference_up=(ObjValue_upwind-ObjValue)./(epsilon*estimparamsvec');
    FiniteDifference_down=(ObjValue-ObjValue_downwind)./(epsilon*estimparamsvec');
    FiniteDifference_centered=(ObjValue_upwind-ObjValue_downwind)./(2*epsilon*estimparamsvec');
    % Jacobian matix of derivatives of model moments with respect to parameters, evaluated at the parameter point estimates

    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference_up=FiniteDifference_up;
    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference_down=FiniteDifference_down;
    momentderivsummary.(['epsilon',num2str(epsilonmodvec(ee))]).FiniteDifference_centered=FiniteDifference_centered;
end


%% Clean up output
MomentDerivativesMatrix=momentderivsummary.(['epsilon',num2str(epsilonmodvec(eedefault))]).FiniteDifference_centered;
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

