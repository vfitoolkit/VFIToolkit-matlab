function Dist=EstimateModel_DistanceFn(CalibParamsVec,CalibParamsNames,CalibParamsSize, ModelToEstimate, TargetValuesVec, TargetDistanceVec, TargetWeightsVec, ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget, estimationoptions, EstimationTargets)
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

%% Calculate ModelTargetsVec from current values of the CalibParamsVec, using ModelTargetsFn

% Create Parameters based on current CalibParamsVec
CalibParamsStruct=CreateVecNamesSizeParams(CalibParamsVec,CalibParamsNames,CalibParamsSize);
for jj=1:length(CalibParamsNames)
    ModelToEstimate.Params.(CalibParamsNames{jj})=CalibParamsStruct.(CalibParamsNames{jj});
end

%%
ModelStatistics=CalculateModelStatistics(ModelToEstimate,estimationoptions); % The following commented-out lines are just the contents of CalculateModelStatistics()
% %% I will store all of the model moments in
% ModelStatistics=struct();
% 
% %% Solve model and calculate the requested statistics
% % Currently just assumes Case 1
% if ModelToEstimate.N_j==Inf
%     if estimationoptions.nestedfixedpoint==1
%         [p_eqm,~,~]=HeteroAgentStationaryEqm_Case1(ModelToEstimate.n_d, ModelToEstimate.n_a, ModelToEstimate.n_s, 0, ModelToEstimate.pi_s, ModelToEstimate.d_grid, ModelToEstimate.a_grid, ModelToEstimate.s_grid, ModelToEstimate.ReturnFn, ModelToEstimate.FnsToEvaluate, ModelToEstimate.GeneralEqmEqns, Params, ModelToEstimate.DiscountFactorParamNames, ModelToEstimate.ReturnFnParamNames, ModelToEstimate.FnsToEvaluateParamNames, ModelToEstimate.GeneralEqmEqnParamNames, ModelToEstimate.GEPriceParamNames, ModelToEstimate.heteroagentoptions, ModelToEstimate.simoptions, ModelToEstimate.vfoptions); %Note n_p=0 is enforced
%         for ii=1:length(ModelToEstimate.GEPriceParamNames)
%             Params.(ModelToEstimate.GEPriceParamNames{ii})=p_eqm.(ModelToEstimate.GEPriceParamNames{ii});
%         end
%     end
%     [~,Policy]=ValueFnIter_Case1(ModelToEstimate.n_d,ModelToEstimate.n_a,ModelToEstimate.n_z,ModelToEstimate.d_grid,ModelToEstimate.a_grid,ModelToEstimate.z_grid, ModelToEstimate.pi_z, ModelToEstimate.ReturnFn, Params, ModelToEstimate.DiscountFactorParamNames, ModelToEstimate.ReturnFnParamNames, ModelToEstimate.vfoptions);
%     StationaryDist=StationaryDist_Case1(Policy,ModelToEstimate.n_d,ModelToEstimate.n_a,ModelToEstimate.n_z,ModelToEstimate.pi_z, ModelToEstimate.simoptions);
%     if max(strcmp(ModelToEstimate.WhichModelStatistics,'AggVars'))==1
%         ModelStatistics.AggVars=EvalFnOnAgentDist_AggVars_Case1(StationaryDist, Policy, ModelToEstimate.FnsToEvaluate,Params, ModelToEstimate.FnsToEvaluateParamNames, ModelToEstimate.n_d, ModelToEstimate.n_a, ModelToEstimate.n_z, ModelToEstimate.d_grid, ModelToEstimate.a_grid,ModelToEstimate.z_grid);
%     end
%     if estimationoptions.ImposeGE==1 && estimationoptions.nestedfixedpoint==0 % Treat general eqm conditions like additional targets
%         for ii=1:length(ModelToEstimate.heteroagentoptions.AggVarsNames)
%             Params.(ModelToEstimate.heteroagentoptions.AggVarsNames{ii})=AggVars(ii);
%         end
%         ModelStatistics.GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1_new(ModelToEstimate.GeneralEqmEqns, ModelToEstimate.GeneralEqmEqnInputNames, Params, ModelToEstimate.simoptions.parallel));
%     end
% else % Finite horizon
%     if estimationoptions.nestedfixedpoint==1
%         [p_eqm,~,~]=HeteroAgentStationaryEqm_Case1_FHorz(ModelToEstimate.jequaloneDist,ModelToEstimate.AgeWeightsParamNames,ModelToEstimate.n_d, ModelToEstimate.n_a, ModelToEstimate.n_z, ModelToEstimate.N_j, 0, ModelToEstimate.pi_z, ModelToEstimate.d_grid, ModelToEstimate.a_grid, ModelToEstimate.z_grid, ModelToEstimate.ReturnFn, ModelToEstimate.FnsToEvaluate, ModelToEstimate.GeneralEqmEqns, Params, ModelToEstimate.DiscountFactorParamNames, ModelToEstimate.ReturnFnParamNames, ModelToEstimate.FnsToEvaluateParamNames, ModelToEstimate.GeneralEqmEqnParamNames, ModelToEstimate.GEPriceParamNames,ModelToEstimate.heteroagentoptions,ModelToEstimate.simoptions,ModelToEstimate.vfoptions); %Note n_p=0 is enforced
%         for ii=1:length(ModelToEstimate.GEPriceParamNames)
%             Params.(ModelToEstimate.GEPriceParamNames{ii})=p_eqm.(ModelToEstimate.GEPriceParamNames{ii});
%         end
%     end
% 
%     % Solve the value function and policy function
%     [V, Policy]=ValueFnIter_Case1_FHorz(ModelToEstimate.n_d,ModelToEstimate.n_a,ModelToEstimate.n_z,ModelToEstimate.N_j, ModelToEstimate.d_grid, ModelToEstimate.a_grid, ModelToEstimate.z_grid, ModelToEstimate.pi_z, ModelToEstimate.ReturnFn, Params, ModelToEstimate.DiscountFactorParamNames, ModelToEstimate.ReturnFnParamNames,ModelToEstimate.vfoptions);
%     % Agent distribution
%     StationaryDist=StationaryDist_FHorz_Case1(ModelToEstimate.jequaloneDist,ModelToEstimate.AgeWeightsParamNames,Policy,ModelToEstimate.n_d,ModelToEstimate.n_a,ModelToEstimate.n_z,ModelToEstimate.N_j,ModelToEstimate.pi_z,Params,ModelToEstimate.simoptions);
%     if max(strcmp(ModelToEstimate.WhichModelStatistics,'AggVars'))==1
%         ModelStatistics.AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, Policy, ModelToEstimate.FnsToEvaluate, Params, ModelToEstimate.FnsToEvaluateParamNames, ModelToEstimate.n_d, ModelToEstimate.n_a, ModelToEstimate.n_z,ModelToEstimate.N_j, ModelToEstimate.d_grid, ModelToEstimate.a_grid, ModelToEstimate.z_grid,[],ModelToEstimate.simoptions);
%     end
%     if max(strcmp(ModelToEstimate.WhichModelStatistics,'LifeCycleProfiles'))==1
%         ModelStatistics.LifeCycleProfiles=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,ModelToEstimate.FnsToEvaluate,ModelToEstimate.FnsToEvaluateParamNames,Params,ModelToEstimate.n_d,ModelToEstimate.n_a,ModelToEstimate.n_z,ModelToEstimate.N_j,ModelToEstimate.d_grid,ModelToEstimate.a_grid,ModelToEstimate.z_grid);
%     end
%     if max(strcmp(ModelToEstimate.WhichModelStatistics,'LorenzCurves'))==1
%     	ModelStatistics.LorenzCurve=EvalFnOnAgentDist_LorenzCurve_FHorz_Case1(StationaryDist, Policy, ModelToEstimate.FnsToEvaluate, Params, ModelToEstimate.FnsToEvaluateParamNames, ModelToEstimate.n_d, ModelToEstimate.n_a, ModelToEstimate.n_z, ModelToEstimate.N_j, ModelToEstimate.d_grid, ModelToEstimate.a_grid, ModelToEstimate.z_grid);
%     end
%     if estimationoptions.ImposeGE==1 && estimationoptions.nestedfixedpoint==0 % Treat general eqm conditions like additional targets
%         for ii=1:length(ModelToEstimate.heteroagentoptions.AggVarsNames)
%             Params.(ModelToEstimate.heteroagentoptions.AggVarsNames{ii})=AggVars(ii);
%         end
%         ModelStatistics.GeneralEqmConditionsVec=real(GeneralEqmConditions_Case1_new(ModelToEstimate.GeneralEqmEqns, ModelToEstimate.GeneralEqmEqnInputNames, Params, ModelToEstimate.simoptions.parallel));
%     end
% end


%% Test the model outputs to make sure that they appear to be in ball-park of the targets
if estimationoptions.testmodel==1    
    fprintf('EstimateModel_DistanceFn: Running test of model statistics \n')
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
                                        % Already given an error if this is a structure (in EstimationModel())
                                        try 
                                            temp=ModelStatistics.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5});
                                            tempE=EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5});
                                            % It exists, now make sure it is the correct size
                                            if size(temp)~=size(tempE)
                                                fprintf('ERROR: %s.%s.%s.%s.%s is different size/shape in EstimationTargets versus ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4},fnames5{f5})
                                                dbstack
                                                break
                                            else
                                                for ii=1:length(tempE)
                                                    if abs(tempE(ii))>0.5
                                                        if tempE(ii)/temp(ii)>20 || temp(ii)/tempE(ii)>20
                                                            fprintf('WARNING: %s.%s.%s.%s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4},fnames5{f5})
                                                        end
                                                    elseif abs(tempE(ii)-temp(ii))>2
                                                        fprintf('WARNING: %s.%s.%s.%s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4},fnames5{f5})
                                                    end
                                                end
                                            end
                                        catch
                                            fprintf('ERROR: %s.%s.%s.%s.%s is an EstimationTargets but not in ModelStatistics (the ModelToEstimate does not create this statistic) \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4},fnames5{f5})
                                            dbstack
                                            break
                                        end
                                    end
                                else
                                    try
                                        temp=ModelStatistics.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4});
                                        tempE=EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4});
                                        % It exists, now make sure it is the correct size
                                        if size(temp)~=size(tempE)
                                            fprintf('ERROR: %s.%s.%s.%s is differerent size/shape in EstimationTargets versus ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4})
                                            dbstack
                                            break
                                        else
                                            for ii=1:length(tempE)
                                                if abs(tempE(ii))>0.5
                                                    if tempE(ii)/temp(ii)>20 || temp(ii)/tempE(ii)>20
                                                        fprintf('WARNING: %s.%s.%s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4})
                                                    end
                                                elseif abs(tempE(ii)-temp(ii))>2
                                                    fprintf('WARNING: %s.%s.%s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4})
                                                end
                                            end
                                        end
                                    catch
                                        fprintf('ERROR: %s.%s.%s.%s is an EstimationTargets but not in ModelStatistics (the ModelToEstimate does not create this statistic) \n', fnames1{f1},fnames2{f2},fnames3{f3},fnames4{f4})
                                        dbstack
                                        break
                                    end
                                end
                            end
                        else
                            try
                                temp=ModelStatistics.(fnames1{f1}).(fnames2{f2}).(fnames3{f3});
                                tempE=EstimationTargets.(fnames1{f1}).(fnames2{f2}).(fnames3{f3});
                                % It exists, now make sure it is the correct size
                                if size(temp)~=size(tempE)
                                    fprintf('ERROR: %s.%s.%s is differerent size/shape in EstimationTargets versus ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3})
                                    dbstack
                                    break
                                else
                                    for ii=1:length(tempE)
                                        if abs(tempE(ii))>0.5
                                            if tempE(ii)/temp(ii)>20 || temp(ii)/tempE(ii)>20
                                                fprintf('WARNING: %s.%s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3})
                                            end
                                        elseif abs(tempE(ii)-temp(ii))>2
                                            fprintf('WARNING: %s.%s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2},fnames3{f3})
                                        end
                                    end
                                end
                            catch
                                fprintf('ERROR: %s.%s.%s is an EstimationTargets but not in ModelStatistics (the ModelToEstimate does not create this statistic) \n', fnames1{f1},fnames2{f2},fnames3{f3})
                                dbstack
                                break
                            end
                        end
                    end
                else
                    try
                        temp=ModelStatistics.(fnames1{f1}).(fnames2{f2});
                        tempE=EstimationTargets.(fnames1{f1}).(fnames2{f2});
                        % It exists, now make sure it is the correct size
                        if size(temp)~=size(tempE)
                            fprintf('ERROR: %s.%s is differerent size/shape in EstimationTargets versus ModelStatistics \n', fnames1{f1},fnames2{f2})
                            dbstack
                            break
                        else
                            for ii=1:length(tempE)
                                if abs(tempE(ii))>0.5
                                    if tempE(ii)/temp(ii)>20 || temp(ii)/tempE(ii)>20
                                        fprintf('WARNING: %s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2})
                                    end
                                elseif abs(tempE(ii)-temp(ii))>2
                                    fprintf('WARNING: %s.%s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1},fnames2{f2})
                                end
                            end
                        end
                    catch
                        fprintf('ERROR: %s.%s is an EstimationTargets but not in ModelStatistics (the ModelToEstimate does not create this statistic) \n', fnames1{f1},fnames2{f2})
                        dbstack
                        break
                    end
                end
            end
        else
            try
                temp=ModelStatistics.(fnames1{f1});
                tempE=EstimationTargets.(fnames1{f1});
                % It exists, now make sure it is the correct size
                if size(temp)~=size(tempE)
                    fprintf('ERROR: %s is differerent size/shape in EstimationTargets versus ModelStatistics \n', fnames1{f1})
                    dbstack
                    break
                else
                    for ii=1:length(tempE)
                        if abs(tempE(ii))>0.5
                            if tempE(ii)/temp(ii)>20 || temp(ii)/tempE(ii)>20
                                fprintf('WARNING: %s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1})
                            end
                        elseif abs(tempE(ii)-temp(ii))>2
                            fprintf('WARNING: %s in EstimationTargets is VERY different to initial value in ModelStatistics \n', fnames1{f1})
                        end
                    end
                end
            catch
                fprintf('ERROR: %s is an EstimationTargets but not in ModelStatistics (the ModelToEstimate does not create this statistic) \n', fnames1{f1})
                dbstack
                break
            end
        end
    end

end

%% Let user know that have solved the model and now calculating the distance itself
if estimationoptions.verbose==1
    fprintf('EstimateModel_DistanceFn: Have solved and simulated model, now calculating the distance \n')
end

%% Get all the model statistics that are targets
ModelStatisticsValuesVec=zeros(size(TargetValuesVec));
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
                                    NumOfEstimationTargets=NumOfEstimationTargets+1;
                                    CurrentTarget=ModelStatistics.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4}).(fnames5{f5});
                                    CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                                    ModelStatisticsValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                                end
                            else
                                NumOfEstimationTargets=NumOfEstimationTargets+1;
                                CurrentTarget=ModelStatistics.(fnames1{f1}).(fnames2{f2}).(fnames3{f3}).(fnames4{f4});
                                CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                                ModelStatisticsValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                            end
                        end
                    else
                        NumOfEstimationTargets=NumOfEstimationTargets+1;
                        CurrentTarget=ModelStatistics.(fnames1{f1}).(fnames2{f2}).(fnames3{f3});
                        CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                        ModelStatisticsValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
                    end
                end
            else
                NumOfEstimationTargets=NumOfEstimationTargets+1;
                CurrentTarget=ModelStatistics.(fnames1{f1}).(fnames2{f2});
                CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
                ModelStatisticsValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
            end
        end
    else
        NumOfEstimationTargets=NumOfEstimationTargets+1;
        CurrentTarget=ModelStatistics.(fnames1{f1});
        CurrVecIndexes=(ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets)+1):1:ShiftCumNumOfEstimationTargetsIndivNumbersPerTarget(NumOfEstimationTargets+1);
        ModelStatisticsValuesVec(CurrVecIndexes)=reshape(CurrentTarget,[numel(CurrentTarget),1]);
    end
end

% Before calculating the distance quickly throw a warning if some model
% statistics are nan when the estimation target is not nan.
if max(isnan(ModelStatisticsValuesVec(~isnan(TargetValuesVec))))==1
    fprintf('WARNING: Some ModelStatistics are nan where the EstimationTargets are not (EstimationModel_DistanceFn()) \n')
end

%% Calculate the weighted distance of the calibration statistics from their targets
DistVec=zeros(size(TargetValuesVec));

% TargetDistanceVec=1 is % Square error divided by absolute value of target: 'squared-absolute-difference'
DistVec(TargetDistanceVec==1)=((ModelStatisticsValuesVec(TargetDistanceVec==1)-TargetValuesVec(TargetDistanceVec==1)).^2)./abs(TargetValuesVec(TargetDistanceVec==1));
% TargetDistanceVec=2 is % abs(error) divided by absolute value of target: 'relative-absolute-difference'
DistVec(TargetDistanceVec==2)=abs(ModelStatisticsValuesVec(TargetDistanceVec==2)-TargetValuesVec(TargetDistanceVec==2))./abs(TargetValuesVec(TargetDistanceVec==2));
% TargetDistanceVec=3 is % Square error: 'squared-difference'
DistVec(TargetDistanceVec==3)=((ModelStatisticsValuesVec(TargetDistanceVec==3)-TargetValuesVec(TargetDistanceVec==3)).^2);
% TargetDistanceVec=4 is % abs(error): 'absolute-difference'
DistVec(TargetDistanceVec==4)=abs(ModelStatisticsValuesVec(TargetDistanceVec==4)-TargetValuesVec(TargetDistanceVec==4));

Dist=sum(DistVec.*TargetWeightsVec,'omitnan');

if estimationoptions.verbose==1
    disp('Current values of the parameters to be calibrated are')
    disp(CalibParamsVec')
    disp('Current moments of model are')
    disp(ModelStatisticsValuesVec')
    disp('The target moments are')
    disp(TargetValuesVec')
    disp('Current vector of weighted distances are')
    disp(DistVec')
    disp('Current sum of weighted distances is')
    disp(sum(Dist))
end

% Save the current status of the EstimateModel_DistanceFn this allows you to check how the estimation is progressing while it is running 
save ./SavedOutput/EstimateModel_Status.mat CalibParamsVec TargetValuesVec ModelStatisticsValuesVec Dist



end