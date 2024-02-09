function [CalibParams,calibsummary]=CalibrateLifeCycleModel(CalibParamNames,TargetMoments,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.


%% Setup caliboptions
if ~isfield(caliboptions,'verbose')
    caliboptions.verbose=1; % sum of squares is the default
end
if ~isfield(caliboptions,'metric')
    caliboptions.metric='sum_squared'; % sum of squares is the default
end
if ~isfield(caliboptions,'weights')
    caliboptions.weights=1; % all moments have equal weights is default (this is a vector of one, just don't know the length yet :)
end
if ~isfield(caliboptions,'constrainpositive')
    caliboptions.constrainpositive=zeros(length(CalibParamNames),1); % if equal 1, then that parameter is constrained to be positive [I want to later modify to use exp()/(1+exp()) to be used to also allow setting min and max bounds, but this is not yet implemented]
end
if ~isfield(caliboptions,'toleranceparams')
    caliboptions.toleranceparams=10^(-4); % tolerance accuracy of the calibrated parameters
end
if ~isfield(caliboptions,'toleranceobjective')
    caliboptions.toleranceobjective=10^(-6); % tolerance accuracy of the objective function
end
if ~isfield(caliboptions,'fminalgo')
    caliboptions.fminalgo=4; % CMA-ES by default, I tried fminsearch() by default but it regularly fails to converge to a decent solution
end
caliboptions.simulatemoments=0; % Not needed here (the objectivefn is shared with other estimation commands)
caliboptions.vectoroutput=0; % Not needed here (the objectivefn is shared with other estimation commands)


%% Setup for which parameters are being calibrated
calibparamsvec0=[]; % column vector
calibparamsvecindex=zeros(length(CalibParamNames)+1,1); % Note, first element remains zero
for pp=1:length(CalibParamNames)
    if size(Parameters.(CalibParamNames{pp}),2)==1
        calibparamsvec0=[calibparamsvec0; Parameters.(CalibParamNames{pp})];
    else
        calibparamsvec0=[calibparamsvec0; Parameters.(CalibParamNames{pp})']; % transpose
    end
    calibparamsvecindex(pp+1)=calibparamsvecindex(pp)+length(Parameters.(CalibParamNames{pp}));
end

for pp=1:length(CalibParamNames)
    if caliboptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=log(calibparamsvec0(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)));
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
% Ends up a colmumn vector (create row vector, then transpose)

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
                targetmomentvec=[targetmomentvec,TargetMoments.AllStats.(a1vec{a1}).(a2vec{a2})];
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
                targetmomentvec=[targetmomentvec,TargetMoments.AgeConditionalStats.(a1vec{a1}).(a2vec{a2})];
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


if isscalar(caliboptions.weights)
    caliboptions.weights=caliboptions.weights.*ones(size(targetmomentvec));
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
    if isfield(vfoptions,'ExogShockFnParamNames')
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




%% Set up the objective function and the initial calibration parameter vector
CalibrationObjectiveFn=@(calibparamsvec) CalibrateLifeCycleModel_objectivefn(calibparamsvec,CalibParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames,usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, calibparamsvecindex, caliboptions, vfoptions,simoptions);


% calibparamsvec0 is our initial guess for calibparamsvec


%% Choosing algorithm for the optimization problem
% https://au.mathworks.com/help/optim/ug/choosing-the-algorithm.html#bscj42s
minoptions = optimset('TolX',caliboptions.toleranceparams,'TolFun',caliboptions.toleranceobjective);
if caliboptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    caliboptions.multiGEcriterion=0;
    [calibparamsvec,calibsummary]=fzero(CalibrationObjectiveFn,calibparamsvec0,minoptions);    
elseif caliboptions.fminalgo==1
    [calibparamsvec,calibsummary]=fminsearch(CalibrationObjectiveFn,calibparamsvec0,minoptions);
elseif caliboptions.fminalgo==2
    % Use the optimization toolbox so as to take advantage of automatic differentiation
    z=optimvar('z',length(calibparamsvec0));
    optimfun=fcn2optimexpr(CalibrationObjectiveFn, z);
    prob = optimproblem("Objective",optimfun);
    z0.z=calibparamsvec0;
    [sol,calibsummary]=solve(prob,z0);
    calibparamsvec=sol.z;
    % Note, doesn't really work as automatic differentiation is only for
    % supported functions, and the objective here is not a supported function
elseif caliboptions.fminalgo==3
    goal=zeros(length(calibparamsvec0),1);
    weight=ones(length(calibparamsvec0),1); % I already implement weights via caliboptions
    [calibparamsvec,calibsummaryVec] = fgoalattain(CalibrationObjectiveFn,calibparamsvec0,goal,weight);
    calibsummary=sum(abs(calibsummaryVec));
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
    [calibparamsvec,calibsummary,counteval,stopflag,out,bestever] = cmaes_vfitoolkit(CalibrationObjectiveFn,calibparamsvec0,caliboptions.insigma,caliboptions.inopts); % ,varargin);
elseif caliboptions.fminalgo==5
    % Update based on rules in caliboptions.fminalgo5.howtoupdate
    error('fminalgo=5 is not possible with model calibration/estimation')
elseif caliboptions.fminalgo==6
    if ~isfield(caliboptions,'lb') || ~isfield(caliboptions,'ub')
        error('When using constrained optimization (caliboptions.fminalgo=6) you must set the lower and upper bounds of the GE price parameters using caliboptions.lb and caliboptions.ub') 
    end
    [calibparamsvec,calibsummary]=fmincon(CalibrationObjectiveFn,calibparamsvec0,[],[],[],[],caliboptions.lb,caliboptions.ub,[],minoptions);    
end



for pp=1:length(CalibParamNames)
    if caliboptions.constrainpositive(pp)==0
        CalibParams.(CalibParamNames{pp})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
    elseif caliboptions.constrainpositive(pp)==1
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        CalibParams.(CalibParamNames{pp})=exp(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)));
    end
end













end