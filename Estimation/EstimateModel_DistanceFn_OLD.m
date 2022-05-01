function Dist=EstimateModel_DistanceFn_OLD(CalibParamsVec, ModelTargetsFn, Parameters, EstimationTargetsVec, ParamNamesToEstimate, NamesEstimationTargets, CalibDistTypeVec, CalibWeightsVec, Verbose, NestedFixedPt)
% Name of function is slightly misleading. While it does target 'moments'
% it does so without any taking into account the uncertainty in the
% moments.

% Takes in current parameter values, model statistics fn, data statistics, a vector telling it how to
% calculate each of the distances (ie. absolute distance, percentage dist,
% etc.), and a vector to give each of them weights.

% Outputs a vector of weighted distances (often you will just want the sum
% of this output).

% CalibParamsVec: a column vector containing the parameters to be calibrated
% ModelTargetsFn: a function which takes as inputs the CalibParamsVec, and gives as outputs the ModelMoments as a structure.
% DataMoments: a vector containing the target data moments
% CalibDistTypeVec: a vector (length equal to number of model/data moments)
%containing numbers in range 1-4 that say how to calculate the distances
%(ie. absolute distance, percentage dist., etc.)
% CalibWeightsVec: a vector containing the weights to apply to each of the
%individual moment distances
% Verbose: if Verbose==1, prints some feedback on how the calibration is
%going.

NumOfEstimationTargets=length(EstimationTargetsVec);

%% If using 'nested-fixed-pt' algorithm for General Equilibrium estimation
if isstruct(NestedFixedPt) % Do the 'interior fixed-pt problem' of finding the general eqm.
    estimationoptions=struct(); % I don't bother to set any of them. 
    % Note that this means that the interior fixed-pt problem forced to use defaults. 
    % (Some time when less lazy I will change this but would require that the relevant estimationoptions are passed in via NestedFixedPt)
    NumberOfGeneralEqmConditions=length(NestedFixedPt.GeneralEqmParamNames);      
    for ii=1:NumberOfGeneralEqmConditions
        GeneralEqmTargets.(NestedFixedPt.GeneralEqmTargetNames{ii})=0; % We know that all the GeneralEqmTargets are zero, we really just need the fieldnames.     
    end
    [Parameters,fval,counteval,exitflag]=EstimateModel(Parameters, GeneralEqmParamNames, GeneralEqmTargets, ModelTargetsFn, estimationoptions);
end

%% Calculate ModelTargetsVec from current values of the CalibParamsVec, using ModelTargetsFn

% Create Parameters based on current CalibParamsVec
CalibParamsTemp=CreateParamsStrucFromParamsVec(ParamNamesToEstimate, CalibParamsVec);
for jj=1:length(ParamNamesToEstimate)
    Parameters.(ParamNamesToEstimate{jj})=CalibParamsTemp.(ParamNamesToEstimate{jj});
end

ModelTargetsStruct=ModelTargetsFn(Parameters);

ModelTargetsVec=CreateVectorFromParams2(ModelTargetsStruct, NamesEstimationTargets);

%% DEBUGGING
fprintf('Debug output: EstimateModel_DistanceFn \n')
size(ModelTargetsVec)
size(EstimationTargetsVec)

%% Calculate the weighted distance of the calibration statistics from their targets
Dist=zeros(NumOfEstimationTargets,1);

for i=1:NumOfEstimationTargets
    if CalibDistTypeVec(i)==1 % Square error divided by absolute value of target: 'squared-absolute-difference'
        if EstimationTargetsVec(i)~=0
            Dist(i)=((ModelTargetsVec(i)-EstimationTargetsVec(i))^2)/abs(EstimationTargetsVec(i));
        else
            fprintf('WARNING: Calibration Statistic number %d is zero valued, so have overriden corresponding choice for CalibDistType', i)
            Dist(i)=((ModelTargetsVec(i)-EstimationTargetsVec(i))^2);
        end
    elseif CalibDistTypeVec(i)==2 % abs(error) divided by absolute value of target: 'relative-absolute-difference'
        if EstimationTargetsVec(i)~=0 
            Dist(i)=abs(ModelTargetsVec(i)-EstimationTargetsVec(i))/abs(EstimationTargetsVec(i));
        else
            fprintf('WARNING: Calibration Statistic number %d is zero valued, so have overriden corresponding choice for CalibDistType', i)
            Dist(i)=abs(ModelTargetsVec(i)-EstimationTargetsVec(i));
        end
    elseif CalibDistTypeVec(i)==3 % Square error: 'squared-difference'
        Dist(i)=(ModelTargetsVec(i)-EstimationTargetsVec(i))^2;
    elseif CalibDistTypeVec(i)==4 % abs(error): 'absolute-difference'
        Dist(i)=abs(ModelTargetsVec(i)-EstimationTargetsVec(i));    
    end
end

% if iscolumn(CalibWeightsVec) % Can probably get rid of this check given that EstimateModel_DistanceFn is only really intended to be called via EstimateModel anyway
%     Dist=Dist.*CalibWeightsVec;
% else
%     disp('CalibWeights, in the EstimateModel_DistanceFn should be a column vector')
Dist=sum(Dist.*CalibWeightsVec');
% end

if Verbose==1
    disp('Current values of the parameters to be calibrated are')
    disp(CalibParamsVec')
    disp('Current moments of model are')
    disp(ModelTargetsVec')
    disp('The target moments are')
    disp(EstimationTargetsVec')
    disp('Current vector of weighted distances are')
    disp(Dist')
    disp('Current sum of weighted distances is')
    disp(sum(Dist))
end

% Save the current status of the EstimateModel_DistanceFn this allows you to check how the estimation is progressing while it is running 
save ./SavedOutput/EstimateModel_Status.mat CalibParamsVec EstimationTargetsVec ModelTargetsVec Dist



end