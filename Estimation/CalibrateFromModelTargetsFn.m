function [Parameters,fval,counteval,exitflag]=CalibrateFromModelTargetsFn(Parameters, ParamNamesToEstimate, EstimationTargets, ModelTargetsFn, estimationoptions, GEPriceParamNames, GeneralEqmTargetNames)
%
% ------------Inputs----------------
% Parameters (structure): gives both the values for parameters which are not calibration exercise, and also gives the initial values for the
%     parameters that are part of the calibration exercise.
% ParamNamesToEstimate (cell of strings): gives the names of the parameters to be estimated.
% EstimationTargets (structure): gives the values for the targets of the estimation.
% ModelTargetsFn (function): takes 'Parameters' as an input, returns a structure with the model-values for the estimation targets 
%     (and the values for the general equilibrium conditions if you want to do a general eqm estimation)
% estimationoptions (structure): can be used to set options relating to 
%     (i) upper and lower bounds for the parameters values, 
%     (ii) how the distance between a given model and data target is calculated, 
%     (iii) the weights given to each of the distances for the individual targets
%     (iv) various other options relating to which algorithms are used and their internal options
%
% ------------Optional Inputs----------------
% If the model to be estimated is general equilibrium you also need the following two inputs,
% GeneralEqmParamNames (cell of strings): an optional input, only use if you wish to solve a general eqm 
%     model (if partial eqm simply do not input). Gives names of the parameters/prices determined as part 
%     of general equilibrium conditions.
% GeneralEqmTargetNames (cell of strings): an optional input, only use if you wish to solve a general eqm 
%     model (if partial eqm simply do not input). Gives names of the general equilibrium conditions.
% Note: GeneralEqmParamNames and GeneralEqmTargetNames must be of equal length.
% To solve General Equilibrium estimation problems you should give both the GeneralEqmParamNames and GeneralEqmTargetNames inputs. 
%     Your 'ModelTargetsFn' should return a structure that contains both the estimation targets and the general equilibrium conditions.
%
%-------------Outputs--------------
% Parameters (structure): contains all the model parameters, with the values for those that were estimated now updated 
%     to their estimated values.
% fval (scalar number): the value of the weighted-sum of the distances between the model values for the targets and 
%     the actual estimation target values.

%% Check some options and if none are given set defaults
if ~isfield(estimationoptions,'verbose')
    estimationoptions.verbose=0;
end

%% Check if this will be a General Equilibrium estimation.
if nargin>5
    ImposeGE=1;
end

NestedFixedPt=0;
WeightForGEConditions=100; % Default value is 100
if ImposeGE==1
    if length(GEPriceParamNames)~=length(GeneralEqmTargetNames)
        fprintf('WARNING: In EstimateModel() the (optional) inputs GeneralEqmParamNames and GeneralEqmTargetNames must be of equal length')
        return
    end
    
    % Two options for dealing with General Equilibrium:
    %   (i) joint-fixed-point (default), 'joint-fixed-pt'
    %   (ii) nested-fixed-point, 'nested-fixed-pt'
    if ~isfield(estimationoptions, 'fixedpointalgo')
        estimationoptions.fixedpointalgo='joint-fixed-pt';
    end
    
    if strcmp(estimationoptions.fixedpointalgo,'joint-fixed-pt')
        if isfield(estimationoptions,'WeightForGEConditions')==1
            WeightForGEConditions=estimationoptions.WeightForGEConditions; % A scalar. Note that this can then be further overruled for specific GeneralEqmConditions by using estimationoptions.TargetWeights.('String_GeneralEqmCondName')
        end
    else % Nested fixed point.
        clear NestedFixedPt % Will instead create a structure which contains all the details of the GeneralEqm aspect and use 'EstimateModel' command to solve this.
        ImposeGE=0;
        NestedFixedPt.GeneralEqmParamNames=GEPriceParamNames;
        NestedFixedPt.GeneralEqmTargetNames=GeneralEqmTargetNames;
    end
end

%%
fprintf('Beginning Model Estimation \n')

%%
if ImposeGE==1 % Because this is the GE
    NumberOfMarketsToClear=length(GEPriceParamNames);
    FullListOfNames={ParamNamesToEstimate{:},GEPriceParamNames{:}};
else
    NumberOfMarketsToClear=0;
    FullListOfNames=ParamNamesToEstimate;
end
NumberOfParametersToCalibrate=length(FullListOfNames);


% Create a vector containing all the parameters to estimate (in order)
CalibParamsVec_initvals=CreateVectorFromParams(Parameters, FullListOfNames);
CalibParamsVec_initvals=CalibParamsVec_initvals'; % CMAES codes require that this is a column vector.
% CalibParamsVec_initvals=CreateVectorFromParams(Parameters, ParamNamesToEstimate);
% if ImposeGE==1 % Because this is the GE
%     CalibParamsVec_initvals=[CalibParamsVec_initvals, CreateVectorFromParams(Parameters, GeneralEqmParamNames)];
% end

% None of the following needed here. Most of the created variables are only used
% internally by EstimateModel_DistanceFn. Precalculated here just for speed.
NamesEstimationTargets=fieldnames(EstimationTargets);
% Create a vector containing all the targets for the estimate (in order)
EstimationTargetsVec=CreateVectorFromParams2(EstimationTargets, NamesEstimationTargets);
NumberOfEstimationTargets=length(EstimationTargetsVec);
% Because the EstimationTargets can be vector valued (not just scalar) we need to do a bit of extra trickery.
EstimationTargetSizesVec=nan(length(NamesEstimationTargets),1);
for jj=1:length(NamesEstimationTargets)
    EstimationTargetSizesVec(jj)=length(EstimationTargets.(NamesEstimationTargets{jj}));
end

% Note: Since some of the estimation targets can be vectors
% length(NamesEstimationTargets) need not equal (can be smaller)
% than NumberOfEstimationTargets (which is a vector of scalar elements)

% Figure out which of these targets are the General Eqm conditions (the individual GE targets must be scalars, not vectors)
GenEqmTargetsIndicatorVec=zeros(1,NumberOfEstimationTargets); % Note that if there are no GE targets this will just remain vector of zeros.
for ii=1:NumberOfMarketsToClear % if ImposeGE==1 % Because this is the GE
    tempGenEqmName=GeneralEqmTargetNames(ii);
    for jj=1:length(NamesEstimationTargets)
        if strcmp(NamesEstimationTargets{jj},tempGenEqmName)
            GenEqmTargetsIndicatorVec(sum(EstimationTargetSizesVec(1:jj-1))+1:sum(EstimationTargetSizesVec(1:jj)))=1;
        end
    end
end
GenEqmTargetsIndicatorVec=logical(GenEqmTargetsIndicatorVec); % Note that if there are no GE targets this will just remain vector of zeros.

EstimationTargetsVec(GenEqmTargetsIndicatorVec)=0; % GeneralEqm conditions (which should equal zero)

% CalibDistTypeVec
% First set the defaults:
CalibDistTypeVec=2*ones(1,NumberOfEstimationTargets); % absolute value of error divided by value of target (default value=2)
CalibDistTypeVec(abs(EstimationTargetsVec)<1)=4; % Absolute difference between model and target for those parameters that are less than one in absolute value
% if ImposeGE==1 % Because this is the GE
CalibDistTypeVec(GenEqmTargetsIndicatorVec)=3; % GE conditions are evaluated based on square of distance from zero.
% end
% Now do any distance functions that have been set as part of the estimationoptions
if isfield(estimationoptions,'TargetDistanceFns')==0
    for jj=1:NumberOfEstimationTargets
        if isfield(estimationoptions.TargetDistanceFns,FullListOfNames{jj})
            temp=estimationoptions.TargetDistanceFns.(FullListOfNames{jj});
            if isstr(temp)
                if strcmp(temp,'relative-square-difference')
                    temp=1;
                elseif strcmp(temp,'relative-absolute-difference')
                    temp=2;
                elseif strcmp(temp,'square-difference')
                    temp=3;
                elseif strcmp(temp,'absolute-difference')
                    temp=4;
                end
            end
            if iscolumn(temp)
                CalibDistTypeVec(sum(EstimationTargetSizesVec(1:jj-1))+1:sum(EstimationTargetSizesVec(1:jj)))=temp;
            else
                CalibDistTypeVec(sum(EstimationTargetSizesVec(1:jj-1))+1:sum(EstimationTargetSizesVec(1:jj)))=temp';
            end
        end
    end
end

% CalibWeightsVec
% First set the defaults:
CalibWeightsVec=ones(1,NumberOfEstimationTargets); % EstimateModel_DistanceFn requires the weights to be a column vector (default value=1)
% if ImposeGE==1 % Because this is the GE
CalibWeightsVec(GenEqmTargetsIndicatorVec)=WeightForGEConditions; % GE conditions are given high weights
% end
% Now do any weights that have been set as part of the estimationoptions
if isfield(estimationoptions,'TargetWeights')==0
    for jj=1:NumberOfEstimationTargets
        if isfield(estimationoptions.TargetWeights,FullListOfNames{jj})
            temp=estimationoptions.TargetWeights.(FullListOfNames{jj});
            if iscolumn(temp)
                CalibWeightsVec(sum(EstimationTargetSizesVec(1:jj-1))+1:sum(EstimationTargetSizesVec(1:jj)))=temp;
            else
                CalibWeightsVec(sum(EstimationTargetSizesVec(1:jj-1))+1:sum(EstimationTargetSizesVec(1:jj)))=temp';
            end
        end
    end
end

% if ImposeGE==1 % Because this is the GE
CalibFn=@(CalibParamsVec) EstimateModel_DistanceFn_OLD(CalibParamsVec, ModelTargetsFn, Parameters, EstimationTargetsVec, FullListOfNames, NamesEstimationTargets, CalibDistTypeVec, CalibWeightsVec, estimationoptions.verbose, NestedFixedPt);
% end

%% Use the CMA-ES algorithm for the distance minimization
fprintf('Starting CMA-ES algorithm of Andreasen (2010) [Covariance Matrix Adaptation Evolutionary Stategy] \n')

%% Settings for CMA-ES
% If no values given, set lower and upper bounds on parameter value to
% being 0.1 and 10 times the initial guess.
lb=nan(size(CalibParamsVec_initvals));
ub=nan(size(CalibParamsVec_initvals));
if isfield(estimationoptions,'ParamBounds')==1
    for ii=1:NumberOfParametersToCalibrate
        if isfield(estimationoptions.ParamBounds,FullListOfNames{ii}) % Also checks for GEPriceParams
            bounds=estimationoptions.ParamBounds.(FullListOfNames{ii});
            lb(ii)=bounds(1);
            ub(ii)=bounds(2);
        else
            if CalibParamsVec_initvals(ii)>0
                lb(ii)=0.1*CalibParamsVec_initvals(ii);
                ub(ii)=10*CalibParamsVec_initvals(ii);
            elseif CalibParamsVec_initvals(ii)==0 % If it is zero we would be left with lb & ub being equal. 
                lb=-0.1; % I thus arbitrarily impose these bounds. I arbirarily chose these bounds for no good reason.
                ub=0.1;  % 
            elseif CalibParamsVec_initvals(ii)<0
                lb(ii)=10*CalibParamsVec_initvals(ii);
                ub(ii)=0.1*CalibParamsVec_initvals(ii);
            end
        end
    end
else
    for ii=1:NumberOfParametersToCalibrate
        if CalibParamsVec_initvals(ii)>0
            lb(ii)=0.1*CalibParamsVec_initvals(ii);
            ub(ii)=10*CalibParamsVec_initvals(ii);
        elseif CalibParamsVec_initvals(ii)==0 % If it is zero we would be left with lb & ub being equal.
            lb=-0.1; % I thus arbitrarily impose these bounds. I arbirarily chose these bounds for no good reason.
            ub=0.1;  %
        elseif CalibParamsVec_initvals(ii)<0
            lb(ii)=10*CalibParamsVec_initvals(ii);
            ub(ii)=0.1*CalibParamsVec_initvals(ii);
        end
    end
end
opts.LBounds = lb;                          %Lower bound for CalibParamsVec
opts.UBounds = ub;                          %Upper bound for CalibParamsVec

if isfield(estimationoptions,'CMAES')==0
    % Set all the remaining options to defaults
    CMAES_sigma = 0.5*abs(ub-lb);             %The std deviation in the initial search distributions
else
    if isfield(estimationoptions.CMAES,'CMAES_sigma')==1
        CMAES_sigma=estimationoptions.CMAES.CMAES_sigma;
    else
        CMAES_sigma = 0.5*abs(ub-lb);             %The std deviation in the initial search distributions
    end
end

%% Finally ready to Calibrate the model!!!

% DEBUGGING PURPOSES ONLY
fprintf('SOME DEBUGGING STUFF \n')
[NumberOfParametersToCalibrate, numel(CalibParamsVec_initvals), numel(CMAES_sigma)]
CalibParamsVec_initvals
CMAES_sigma
[lb,ub]

[CalibParamsVec, fval, counteval, exitflag] = cmaes_vfitoolkit(CalibFn,CalibParamsVec_initvals,CMAES_sigma,opts);

% CalibParamsTemp=CreateParamsStrucFromParamsVec(ParamNamesToEstimate, CalibParamsVec);
% for jj=1:NumberOfParametersToCalibrate
%     Parameters.(ParamNamesToEstimate{:})=CalibParamsTemp.(ParamNamesToEstimate{:});
% end
% Because this is the GE
% FullListOfNames={ParamNamesToEstimate{:},MarketClearanceNames{:}};
CalibParamsTemp=CreateParamsStrucFromParamsVec(FullListOfNames, CalibParamsVec);
for jj=1:length(FullListOfNames)
    Parameters.(FullListOfNames{jj})=CalibParamsTemp.(FullListOfNames{jj});
end


end