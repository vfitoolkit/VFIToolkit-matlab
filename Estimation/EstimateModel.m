function [EstimatedParameters,fval,counteval,exitflag]=EstimateModel(ParamNamesToEstimate, EstimationTargets, ModelToEstimate, estimationoptions)
%
% ------------Inputs----------------
% ParamNamesToEstimate (cell of strings): gives the names of the parameters to be estimated.
% EstimationTargets (structure): gives the values for the targets of the estimation.
% ModelToEstimate (structure): contains all parts of the model (e.g. Parameters, grids, FnsToEvaluate, ReturnFn, etc.)
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
if ~exist('estimationoptions','var')
    estimationoptions.verbose=0;
    estimationoptions.nestedfixedpoint=0;
    estimationoptions.testmodel=1; % Test the model before beginning the estimation (gives more feedback and checks that model is correctly set up)
else
    if ~isfield(estimationoptions,'verbose')
        estimationoptions.verbose=0;
    end
    if ~isfield(estimationoptions,'nestedfixedpoint')
        estimationoptions.nestedfixedpoint=0; % By default treat GE conditions as joint targets, rather than solving GE as nested fixed point
    end
    if ~isfield(estimationoptions,'testmodel')
        estimationoptions.testmodel=1; % Test the model before beginning the estimation (gives more feedback and checks that model is correctly set up)
    end
end

%%
if estimationoptions.verbose==1
    fprintf('Setting up Model Estimation \n')
end

%% Check if ModelToEstimate appears to contain what it should
if ~isfield(ModelToEstimate,'Params') && ~isfield(ModelToEstimate,'Parameters')
	fprintf('ERROR: ModelToEstimate does not specify Params/Parameters \n')
elseif isfield(ModelToEstimate,'Params')
    ModelToEstimate.Parameters=ModelToEstimate.Params; % Codes assume it is called Parameters
end

if ~isfield(ModelToEstimate,'n_d')
	fprintf('ERROR: ModelToEstimate does not specify n_d \n')
elseif ~isfield(ModelToEstimate,'n_a')
	fprintf('ERROR: ModelToEstimate does not specify n_a \n')
elseif ~isfield(ModelToEstimate,'n_z')
	fprintf('ERROR: ModelToEstimate does not specify n_z \n')
elseif ~isfield(ModelToEstimate,'d_grid')
	fprintf('ERROR: ModelToEstimate does not specify d_grid \n')
elseif ~isfield(ModelToEstimate,'a_grid')
	fprintf('ERROR: ModelToEstimate does not specify a_grid \n')
elseif ~isfield(ModelToEstimate,'z_grid')
	fprintf('ERROR: ModelToEstimate does not specify z_grid \n')
elseif ~isfield(ModelToEstimate,'pi_z')
	fprintf('ERROR: ModelToEstimate does not specify pi_z \n')
elseif ~isfield(ModelToEstimate,'DiscountFactorParamNames')
	fprintf('ERROR: ModelToEstimate does not specify DiscountFactorParamNames \n')
elseif ~isfield(ModelToEstimate,'ReturnFn')
	fprintf('ERROR: ModelToEstimate does not specify ReturnFn \n')
end

if ~isfield(ModelToEstimate,'N_j')
	fprintf('WARNING: ModelToEstimate does not specify time horizon, have assumed infinite horizon \n')
    ModelToEstimate.N_j=Inf;
elseif isfinite(ModelToEstimate.N_j) % Finite horizon
    if ~isfield(ModelToEstimate,'jequaloneDist')
        fprintf('ERROR: ModelToEstimate does not specify jequaloneDist \n')
    elseif ~isfield(ModelToEstimate,'AgeWeightsParamNames')
        fprintf('ERROR: ModelToEstimate does not specify AgeWeightsParamNames \n')
    end
end
if ~isfield(ModelToEstimate,'FnsToEvaluate')
	fprintf('ERROR: ModelToEstimate does not specify FnsToEvaluate \n')
elseif ~isfield(ModelToEstimate,'WhichModelStatistics')
	fprintf('ERROR: ModelToEstimate does not specify WhichModelStatistics \n')
end

%% Check if this will be a General Equilibrium estimation.
if isfield(ModelToEstimate,'GeneralEqmEqns')
    if ~isfield(estimationoptions,'ImposeGE')
        estimationoptions.ImposeGE=1;    
    end
    if ~isfield(ModelToEstimate,'GEPriceParamNames')
        fprintf('ERROR: ModelToEstimate uses GeneralEqmEqns but does not specify GEPriceParamNames \n')        
    end
    if estimationoptions.ImposeGE==1
        % Two options for dealing with General Equilibrium:
        %   (i) joint-fixed-point (default), 'joint-fixed-pt'
        %   (ii) nested-fixed-point, 'nested-fixed-pt'
        if ~isfield(estimationoptions, 'fixedpointalgo')
            estimationoptions.fixedpointalgo='joint-fixed-pt';
        end
        
        ParamNamesToEstimate={ParamNamesToEstimate{:},GEPriceParamNames{:}}; % Add General eqm parameters to the list of those to estimate
        
        % Add the General Eqm Conditions as Estimation Targets
        GEConditionNames=fieldnames(ModelToEstimate.GeneralEqmEqns);
        for gg=1:length(GEConditionNames)
            EstimationTargets.(GEConditionNames{gg})=0; % GE conditions should all evaluate to zero
        end
        
        % To be able to evaluate the GE conditions need AggVars, which may
        % not have been requested as one of the estimation targets so need
        % to add it to what will be calculated.
        if max(strcmp(ModelToEstimate.WhichModelStatistics,'AggVars'))==0
            ModelToEstimate.WhichModelStatistics{1+length(ModelToEstimate.WhichModelStatistics)}='AggVars';
        end
    end
else
    estimationoptions.ImposeGE=0;
end

%% Set up initial value for the parameters being estimated, as well as weights for the estimation targets

% Create a vector containing all the parameters to estimate (in order)
CalibParamsStruct=struct();
for ii=1:length(ParamNamesToEstimate)
    CalibParamsStruct.(ParamNamesToEstimate{ii})=ModelToEstimate.Parameters.(ParamNamesToEstimate{ii});
end
% To be able to access the parameters need to know the size of each of them
[CalibParamsVec,CalibParamsNames,CalibParamsSize]=CreateParamsVecNamesSize(CalibParamsStruct);
% Note that CalibParamNames is effectively just a copy of ParamNamesToEstimate

%% Create a few things that are used to calculate the distance once all the model statistics have been calculated

NumOfEstimationTargets=0;
NumOfEstimationTargetsIndivNumbersPerTarget=[];
NumOfEstimationTargetsIndivNumbers=0;
% I break down EstimationTargets. Allow up to 5 layers
fnames1=fieldnames(EstimationTargets);
for f1=1:length(fnames1)
    if isstruct(EstimationTargets.(fnames1{f1}))
        fnames2=fieldnames(EstimationTargets.(fnames1{f1}));
        for f2=1:length(fnames2)
            if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}))
                fnames3=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}));
                for f3=1:length(fnames3)
                    if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}))
                        fnames4=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}));
                        for f4=1:length(fnames4)
                            if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}))
                                fnames5=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}));
                                for f5=1:length(fnames5)
                                    if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5}))
                                        fprintf('ERROR: EstimationTargets is a structure with a depth of more than 5 (depth 5 is the max) \n')
                                        dbstack
                                        break
                                    else
                                        NumOfEstimationTargets=NumOfEstimationTargets+1;
                                        NumOfEstimationTargetsIndivNumbers=NumOfEstimationTargetsIndivNumbers+numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5}));
                                        NumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)=numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5}));
                                    end
                                end
                            else
                                NumOfEstimationTargets=NumOfEstimationTargets+1;
                                NumOfEstimationTargetsIndivNumbers=NumOfEstimationTargetsIndivNumbers+numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}));
                                NumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)=numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}));
                            end
                        end
                    else
                        NumOfEstimationTargets=NumOfEstimationTargets+1;
                        NumOfEstimationTargetsIndivNumbers=NumOfEstimationTargetsIndivNumbers+numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}));
                        NumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)=numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}));
                    end
                end
            else
                NumOfEstimationTargets=NumOfEstimationTargets+1;
                NumOfEstimationTargetsIndivNumbers=NumOfEstimationTargetsIndivNumbers+numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}));
                NumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)=numel(EstimationTargets.(fnames1{f1}).(fnames2{f2}));
            end
        end
    else
        NumOfEstimationTargets=NumOfEstimationTargets+1;
        NumOfEstimationTargetsIndivNumbers=NumOfEstimationTargetsIndivNumbers+numel(EstimationTargets.(fnames1{f1}));
        NumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)=numel(EstimationTargets.(fnames1{f1}));
    end
end

CumNumOfEstimationTargetsIndivNumbersPerTarget=cumsum(NumOfEstimationTargetsIndivNumbersPerTarget);
ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget=[0,CumNumOfEstimationTargetsIndivNumbersPerTarget];

%% I can actually create 'which calibration targets' from the fnames1


%% Create vectors that will be used to store the targets, as well as how to compute the distance and what weights to apply when adding them
TargetValuesVec=nan(NumOfEstimationTargetsIndivNumbers,1); % Use nan as want to cause errors if don't put in numbers later
TargetDistanceVec=2*ones(NumOfEstimationTargetsIndivNumbers,1); % Two is square of error divided by value of target (unless target is of absolute value less than 0.5, in which case just use absolute difference
TargetWeightsVec=ones(NumOfEstimationTargetsIndivNumbers,1);

% The default weight applied to General Eqm conditions
if ~isfield(estimationoptions,'DefaultGEWeight')
    estimationoptions.DefaultGEWeight=20;
end

% I break down EstimationTargets. Allow up to 5 layers
NumOfEstimationTargets=0;
fnames1=fieldnames(EstimationTargets);
for f1=1:length(fnames1)
    if isstruct(EstimationTargets.(fnames1{f1}))
        fnames2=fieldnames(EstimationTargets.(fnames1{f1}));
        for f2=1:length(fnames2)
            if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}))
                fnames3=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}));
                for f3=1:length(fnames3)
                    if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}))
                        fnames4=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}));
                        for f4=1:length(fnames4)
                            if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}))
                                fnames5=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}));
                                for f5=1:length(fnames5)
                                    if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5}))
                                        fprintf('ERROR: EstimationTargets is a structure with a depth of more than 5 (depth 5 is the max) \n')
                                        dbstack
                                        break
                                    else
                                        NumOfEstimationTargets=NumOfEstimationTargets+1;
                                        CurrentTarget=EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5});
                                        CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                                        TargetValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                                        try % If this field exists in EstimationWeights
                                            temp=estimationoptions.EstimationWeights.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5});
                                            if isscalar(temp)
                                                TargetWeightsVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                                            else
                                                TargetWeightsVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                                            end
                                        catch
                                            % If the field does not exist in estimationoptions.EstimationWeights then do nothing (leave weight as default of one)
                                        end
                                        try % If this field exists in EstimationDistanceMeasure
                                            temp=estimationoptions.EstimationDistanceMeasure.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5});
                                            if isstr(temp) % Note that when using string it must be same for all parts of this target vector/matrix (you can use numbers in a vector/matrix to allow different distance measures for different elements of target)
                                                if strcmp(temp,'relative-square-difference')
                                                    temp=1;
                                                elseif strcmp(temp,'relative-absolute-difference')
                                                    temp=2;
                                                elseif strcmp(temp,'square-difference')
                                                    temp=3;
                                                elseif strcmp(temp,'absolute-difference')
                                                    temp=4;
                                                end
                                            elseif isscalar(temp)
                                                TargetDistanceVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                                            else
                                                TargetDistanceVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                                            end
                                        catch
                                            % If the field does not exist in estimationoptions.EstimationDistanceMeasure then do nothing (leave weight as default of one)
                                        end
                                    end
                                end
                            else
                                NumOfEstimationTargets=NumOfEstimationTargets+1;
                                CurrentTarget=EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4});
                                CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                                TargetValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                                try % If this field exists in EstimationWeights
                                    temp=estimationoptions.EstimationWeights.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4});
                                    if isscalar(temp)
                                        TargetWeightsVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                                    else
                                        TargetWeightsVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                                    end
                                catch
                                    % If the field does not exist in estimationoptions.EstimationWeights then do nothing (leave weight as default of one)
                                end
                                try % If this field exists in EstimationDistanceMeasure
                                    temp=estimationoptions.EstimationDistanceMeasure.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4});
                                    if isstr(temp) % Note that when using string it must be same for all parts of this target vector/matrix (you can use numbers in a vector/matrix to allow different distance measures for different elements of target)
                                        if strcmp(temp,'relative-square-difference')
                                            temp=1;
                                        elseif strcmp(temp,'relative-absolute-difference')
                                            temp=2;
                                        elseif strcmp(temp,'square-difference')
                                            temp=3;
                                        elseif strcmp(temp,'absolute-difference')
                                            temp=4;
                                        end
                                    elseif isscalar(temp)
                                        TargetDistanceVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                                    else
                                        TargetDistanceVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                                    end
                                catch
                                    % If the field does not exist in estimationoptions.EstimationDistanceMeasure then do nothing (leave weight as default of one)
                                end
                            end
                        end
                    else
                        NumOfEstimationTargets=NumOfEstimationTargets+1;
                        CurrentTarget=EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3});
                        CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                        TargetValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                        try % If this field exists in EstimationWeights
                            temp=estimationoptions.EstimationWeights.(fnames1{f1}).(fnames2{f2}).(fnames3{f3});
                            if isscalar(temp)
                                TargetWeightsVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                            else
                                TargetWeightsVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                            end
                        catch
                            % If the field does not exist in estimationoptions.EstimationWeights then do nothing (leave weight as default of one)
                        end
                        try % If this field exists in EstimationDistanceMeasure
                            temp=estimationoptions.EstimationDistanceMeasure.(fnames1{f1}).(fnames2{f2}).(fnames3{f3});
                            if isstr(temp) % Note that when using string it must be same for all parts of this target vector/matrix (you can use numbers in a vector/matrix to allow different distance measures for different elements of target)
                                if strcmp(temp,'relative-square-difference')
                                    temp=1;
                                elseif strcmp(temp,'relative-absolute-difference')
                                    temp=2;
                                elseif strcmp(temp,'square-difference')
                                    temp=3;
                                elseif strcmp(temp,'absolute-difference')
                                    temp=4;
                                end
                            elseif isscalar(temp)
                                TargetDistanceVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                            else
                                TargetDistanceVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                            end
                        catch
                            % If the field does not exist in estimationoptions.EstimationDistanceMeasure then do nothing (leave weight as default of one)
                        end
                    end
                end
            else
                NumOfEstimationTargets=NumOfEstimationTargets+1;
                CurrentTarget=EstimationTargets.(fnames1{f1}).(fnames2{f2});
                CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                TargetValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                try % If this field exists in EstimationWeights
                    temp=estimationoptions.EstimationWeights.(fnames1{f1}).(fnames2{f2});
                    if isscalar(temp)
                        TargetWeightsVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                    else
                        TargetWeightsVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                    end
                catch
                    % If the field does not exist in estimationoptions.EstimationWeights then do nothing (leave weight as default of one)
                end
                try % If this field exists in EstimationDistanceMeasure
                    temp=estimationoptions.EstimationDistanceMeasure.(fnames1{f1}).(fnames2{f2});
                    if isstr(temp) % Note that when using string it must be same for all parts of this target vector/matrix (you can use numbers in a vector/matrix to allow different distance measures for different elements of target)
                        if strcmp(temp,'relative-square-difference')
                            temp=1;
                        elseif strcmp(temp,'relative-absolute-difference')
                            temp=2;
                        elseif strcmp(temp,'square-difference')
                            temp=3;
                        elseif strcmp(temp,'absolute-difference')
                            temp=4;
                        end
                    elseif isscalar(temp)
                        TargetDistanceVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
                    else
                        TargetDistanceVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
                    end
                catch
                    % If the field does not exist in estimationoptions.EstimationDistanceMeasure then do nothing (leave weight as default of one)
                end
            end
        end
    else
        NumOfEstimationTargets=NumOfEstimationTargets+1;
        CurrentTarget=EstimationTargets.(fnames1{f1});
        CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
        TargetValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
        
        % Because of how I put GE conditions into the EstimationTargets,
        % any estimation targets will be here (they are never structures,
        % all must be numbers in this first layer)
        % For the GE conditions, set the default weights to 20
        if estimationoptions.ImposeGE==1
            if estimationoptions.nestedfixedpoint==0
                % Add the General Eqm Conditions as Estimation Targets
                if max(strcmp(GEConditionNames,fnames1{f1}))==1
                    TargetWeightsVec(CurrVecIndexes)=DefaultGEWeight; % Note that code then checks if you have specified weight and overwrites this if you do
                end
            end
        end
        
        try % If this field exists in EstimationWeights
            temp=estimationoptions.EstimationWeights.(fnames1{f1});
            if isscalar(temp)
                TargetWeightsVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
            else
                TargetWeightsVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
            end
        catch
            % If the field does not exist in estimationoptions.EstimationWeights then do nothing (leave weight as default of one)
        end
        try % If this field exists in EstimationDistanceMeasure
            temp=estimationoptions.EstimationDistanceMeasure.(fnames1{f1});
            if isstr(temp) % Note that when using string it must be same for all parts of this target vector/matrix (you can use numbers in a vector/matrix to allow different distance measures for different elements of target)
                if strcmp(temp,'relative-square-difference')
                    temp=1;
                elseif strcmp(temp,'relative-absolute-difference')
                    temp=2;
                elseif strcmp(temp,'square-difference')
                    temp=3;
                elseif strcmp(temp,'absolute-difference')
                    temp=4;
                end
            elseif isscalar(temp)
                TargetDistanceVec(CurrVecIndexes)=temp*ones(numel(CurrentTarget),1);
            else
                TargetDistanceVec(CurrVecIndexes)=reshape(temp,[numel(CurrentTarget),1]);
            end
        catch
            % If the field does not exist in estimationoptions.EstimationDistanceMeasure then do nothing (leave weight as default of one)
        end
    end
end

if max(TargetValuesVec(TargetDistanceVec==1)==0)
    fprintf('WARNING: Cannot use relative-square-difference as the measure of distance for an estimation target which is zero (as would divide by zero), switched to square-difference \n')
    TargetDistanceVec(TargetValuesVec(TargetDistanceVec==1)==0)=3;
end
if max(TargetValuesVec(TargetDistanceVec==2)==0)
    fprintf('WARNING: Cannot use relative-absolute-difference as the measure of distance for an estimation target which is zero (as would divide by zero), switched to absolute-difference \n')
    TargetDistanceVec(TargetValuesVec(TargetDistanceVec==2)==0)=4;
end

%% Check if any of the targets are 'nan'. If they are then print warning that they will be ignored.
if max(isnan(TargetValuesVec))==1
    fprintf('Following WARNINGS are from from EstimationModel() \n')
    % There is a nan somewhere, so go through the whole structure to find and report all the nan targets by name
    % I break down EstimationTargets. Allow up to 5 layers
    fnames1=fieldnames(EstimationTargets);
    for f1=1:length(fnames1)
        if isstruct(EstimationTargets.(fnames1{f1}))
            fnames2=fieldnames(EstimationTargets.(fnames1{f1}));
            for f2=1:length(fnames2)
                if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}))
                    fnames3=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}));
                    for f3=1:length(fnames3)
                        if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}))
                            fnames4=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}));
                            for f4=1:length(fnames4)
                                if isstruct(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}))
                                    fnames5=fieldnames(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}));
                                    for f5=1:length(fnames5)
                                        % Code already threw an error if the 5th layer is a structure
                                        if max(isnan(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5})))==1
                                            fprintf('WARNING: %s.%s.%s.%s.%s EstimationTargets contains some nan values, these will be ignored/omitted from the calculation of the distance \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4},fnames5{f5})
                                        end
                                    end
                                else
                                    if max(isnan(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4})))==1
                                        fprintf('WARNING: %s.%s.%s.%s EstimationTargets contains some nan values, these will be ignored/omitted from the calculation of the distance \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4})
                                    end
                                end
                            end
                        else
                            if max(isnan(EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3})))==1
                                fprintf('WARNING: %s.%s.%s EstimationTargets contains some nan values, these will be ignored/omitted from the calculation of the distance \n', fnames1{f1},fnames2{f2},fnames3{f3})
                            end
                        end
                    end
                else
                    if max(isnan(EstimationTargets.(fnames1{f1}).(fnames2{f2})))==1
                        fprintf('WARNING: %s.%s EstimationTargets contains some nan values, these will be ignored/omitted from the calculation of the distance \n', fnames1{f1},fnames2{f2})
                    end
                end
            end
        else
            if max(isnan(EstimationTargets.(fnames1{f1})))==1
                fprintf('WARNING: %s EstimationTargets contains some nan values, these will be ignored/omitted from the calculation of the distance \n', fnames1{f1})
            end
        end
    end
end

%% Create the function that is takes CalibParamsVec and the ModelToEstimate and returns the distance between the targest and model statistics

% Do a sets of CalibFn that makes sure that all of the model statistics are being calculated, and that they are all the appropriate size
CalibFn=@(CalibParamsVec) EstimateModel_DistanceFn(CalibParamsVec,CalibParamsNames,CalibParamsSize, ModelToEstimate, TargetValuesVec, TargetDistanceVec, TargetWeightsVec, ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget, estimationoptions, EstimationTargets);
% Note that EstimationTargets is only used if TestModel=1

if estimationoptions.testmodel==1
    % Run the test of the model
    CalibFn(CalibParamsVec)
    estimationoptions.testmodel=0;
    % Redefine CalibFn using TestModel=0
    CalibFn=@(CalibParamsVec) EstimateModel_DistanceFn(CalibParamsVec,CalibParamsNames,CalibParamsSize, ModelToEstimate, TargetValuesVec, TargetDistanceVec, TargetWeightsVec, ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget, estimationoptions, EstimationTargets);
end


%% Use the CMA-ES algorithm for the distance minimization
fprintf('Starting CMA-ES algorithm of Andreasen (2010) [Covariance Matrix Adaptation Evolutionary Stategy] \n')

%% Settings for CMA-ES
% If no values given, set lower and upper bounds on parameter value to
% being 0.1 and 10 times the initial guess.
lb=nan(size(CalibParamsVec));
ub=nan(size(CalibParamsVec));
if isfield(estimationoptions,'ParamBounds')==1
    for ii=1:length(ParamNamesToEstimate)
        if isfield(estimationoptions.ParamBounds,ParamNamesToEstimate{ii}) % Also checks for GEPriceParams
            bounds=estimationoptions.ParamBounds.(ParamNamesToEstimate{ii});
            lb(ii)=bounds(1);
            ub(ii)=bounds(2);
        else
            if CalibParamsVec(ii)>0
                lb(ii)=0.1*CalibParamsVec(ii);
                ub(ii)=10*CalibParamsVec(ii);
            elseif CalibParamsVec(ii)==0 % If it is zero we would be left with lb & ub being equal. 
                lb=-0.1; % I thus arbitrarily impose these bounds. I arbirarily chose these bounds for no good reason.
                ub=0.1;  % 
            elseif CalibParamsVec(ii)<0
                lb(ii)=10*CalibParamsVec(ii);
                ub(ii)=0.1*CalibParamsVec(ii);
            end
        end
    end
else
    for ii=1:length(ParamNamesToEstimate)
        if CalibParamsVec(ii)>0
            lb(ii)=0.1*CalibParamsVec(ii);
            ub(ii)=10*CalibParamsVec(ii);
        elseif CalibParamsVec(ii)==0 % If it is zero we would be left with lb & ub being equal.
            lb=-0.1; % I thus arbitrarily impose these bounds. I arbirarily chose these bounds for no good reason.
            ub=0.1;  %
        elseif CalibParamsVec(ii)<0
            lb(ii)=10*CalibParamsVec(ii);
            ub(ii)=0.1*CalibParamsVec(ii);
        end
    end
end
opts.LBounds = lb;                          %Lower bound for CalibParamsVec
opts.UBounds = ub;                          %Upper bound for CalibParamsVec


if isfield(estimationoptions,'CMAES')==0
    % Set all the remaining options to defaults
    CMAES_Insigma = 0.5*abs(ub-lb);             %The std deviation in the initial search distributions
    CMAES_sigma   = 1;                          %The step size
    opts.SigmaMax = 5;                          %The maximal value for sigma
%     opts.LBounds = lb;                          %Lower bound for CalibParamsVec
%     opts.UBounds = ub;                          %Upper bound for CalibParamsVec
    opts.MaxIter = 1000;                     %The maximum number of iterations
    opts.PopSize = 20;                          %The population size
    %Note: MaxEvals will be roughly MaxIter*PopSize (not exactly due to resampling)
    opts.VerboseModulo = 10;                    %Display results after every 10'th iteration
    opts.TolFun = 1e-06;                        %Function tolerance
    opts.TolX   = 1e-06;                        %Tolerance in the parameters
    opts.Plotting = 'off';                      %Dislpay plotting or not
    opts.Saving  =  'on';                       %Saving results
    opts.SaveFileName = './SavedOutput/CMAES_run_data.mat';    %Results are saved to this file
else
    if isfield(estimationoptions.CMAES,'CMAES_Insigma')==1
        CMAES_Insigma=estimationoptions.CMAES.CMAES_Insigma;
    else
        CMAES_Insigma = 0.5*abs(ub-lb);             %The std deviation in the initial search distributions
    end
    if isfield(estimationoptions.CMAES,'CMAES_sigma')==1
        CMAES_sigma=estimationoptions.CMAES.CMAES_sigma;
    else
        CMAES_sigma   = 1;                          %The step size
    end
    if isfield(estimationoptions.CMAES,'SigmaMax')==1
        opts.SigmaMax=estimationoptions.CMAES.SigmaMax;
    else
        opts.SigmaMax = 5;                          %The maximal value for sigma
    end
    if isfield(estimationoptions.CMAES,'MaxIter')==1
        opts.MaxIter=estimationoptions.CMAES.MaxIter;
    else
        opts.MaxIter = 1000;                     %The maximum number of iterations
    end
    if isfield(estimationoptions.CMAES,'PopSize')==1
        opts.PopSize=estimationoptions.CMAES.PopSize;
    else
        opts.PopSize = 20;                          %The population size
    end
    if isfield(estimationoptions.CMAES,'VerboseModulo')==1
        opts.VerboseModulo=estimationoptions.CMAES.VerboseModulo;
    else
        opts.VerboseModulo = 10;                    %Display results after every 10'th iteration
    end
    if isfield(estimationoptions.CMAES,'TolFun')==1
        opts.TolFun=estimationoptions.CMAES.TolFun;
    else
        opts.TolFun = 1e-06;                        %Function tolerance
    end
    if isfield(estimationoptions.CMAES,'TolX')==1
        opts.TolX=estimationoptions.CMAES.TolX;
    else
        opts.TolX   = 1e-06;                        %Tolerance in the parameters
    end
    if isfield(estimationoptions.CMAES,'Plotting')==1
        opts.Plotting=estimationoptions.CMAES.Plotting;
    else
        opts.Plotting = 'off';                      %Dislpay plotting or not
    end
    if isfield(estimationoptions.CMAES,'Saving')==1
        opts.Saving=estimationoptions.CMAES.Saving;
    else
        opts.Saving  =  'on';                       %Saving results
    end
    if isfield(estimationoptions.CMAES,'SaveFileName')==1
        opts.SaveFileName=estimationoptions.CMAES.SaveFileName;
    else
        opts.SaveFileName = './SavedOutput/CMAES_run_data.mat';    %Results are saved to this file
    end
end

%% Finally ready to Calibrate the model!!!

% DEBUGGING PURPOSES ONLY
fprintf('SOME DEBUGGING STUFF \n')
[length(ParamNamesToEstimate), numel(CalibParamsVec), numel(CMAES_Insigma)]
CalibParamsVec
CMAES_Insigma
[lb,ub]

if estimationoptions.verbose==1
    fprintf('Setting up Model Estimation \n')
end

[CalibParamsVec, fval, counteval, exitflag] = cmaes_dsge(CalibFn,CalibParamsVec,CMAES_sigma, CMAES_Insigma,opts);

if estimationoptions.verbose==1
    fprintf('Completed Model Estimation \n')
    CalibParamsVec
end

%% Clean up output (put the estimated parameters in a structure)
% Includes general eqm parameters if general eqm was used
CalibParamsTemp=CreateParamsStrucFromParamsVec(ParamNamesToEstimate, CalibParamsVec);
EstimatedParameters=struct();
for jj=1:length(ParamNamesToEstimate)
    EstimatedParameters.(ParamNamesToEstimate{jj})=CalibParamsTemp.(ParamNamesToEstimate{jj});
end

EstimatedParameters


end