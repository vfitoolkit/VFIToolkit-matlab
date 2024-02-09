function Obj=CalibrateLifeCycleModel_PType_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames,PTypeDistParamNames, PTypeParamFn, FnsToEvaluate, usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, calibparamsvecindex, calibparamssizes, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.

% Do any transformations of parameters before we say what they are
for pp=1:length(CalibParamNames)
    if caliboptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=exp(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)));
    end
end

if caliboptions.verbose==1
    fprintf('Current calib parameters')
    CalibParamNames
    calibparamsvec
end

for pp=1:length(CalibParamNames)
    Parameters.(CalibParamNames{pp})=reshape(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)),calibparamssizes(pp,:));
end


% Some PType parameters may be parametrized by other parameters
Parameters=PTypeParamFn(Parameters);


%% Solve the model and calculate the stats
[~, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist,AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,N_i,pi_z_J,Parameters,simoptions);

if usingallstats==1
    simoptions.whichstats=AllStats_whichstats;
    AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy, FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_gridvals_J,simoptions);
end
if usinglcp==1
    simoptions.whichstats=ACStats_whichstats;
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,FnsToEvaluate,Parameters,n_d,n_a,n_z,N_j,N_i,d_grid,a_grid,z_gridvals_J,simoptions);
end


%% Get current values of the target moments as a vector
currentmomentvec=zeros(size(targetmomentvec));
if usingallstats==1
    currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2});
    for cc=2:size(allstatmomentnames,1)
        if isempty(allstatmomentnames{cc,3})
            currentmomentvec(allstatcummomentsizes(cc-1):allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2});
        else
            currentmomentvec(allstatcummomentsizes(cc-1):allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2}).(allstatmomentnames{cc,3});
        end
    end
end
if usinglcp==1
    currentmomentvec(allstatcummomentsizes(end)+1:allstatcummomentsizes(end)+acscummomentsizes(1))=AgeConditionalStats.(acsmomentnames{1,1}).(acsmomentnames{1,2});
    for cc=2:size(acsmomentnames,1)
        if isempty(acsmomentnames{cc,3})
            currentmomentvec(allstatcummomentsizes(end)+acscummomentsizes(cc-1)+1:allstatcummomentsizes(end)+acscummomentsizes(cc))=AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2});
        else
            currentmomentvec(allstatcummomentsizes(end)+acscummomentsizes(cc-1)+1:allstatcummomentsizes(end)+acscummomentsizes(cc))=AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2}).(acsmomentnames{cc,3});
        end
    end
end



%% Evaluate the objective function
% currentmomentvec is the current moment values
% targetmomentvec is the target moment values

if strcmp(caliboptions.metric,'sum_squared')
    Obj=sum(caliboptions.weights.*(targetmomentvec-currentmomentvec).^2);
elseif strcmp(caliboptions.metric,'sum_logratiosquared')
    Obj=sum(caliboptions.weights.*((log(targetmomentvec./currentmomentvec)).^2));
end
Obj=Obj/length(targetmomentvec); % This is done so that the tolerances for convergence are sensible



%% Verbose
if caliboptions.verbose==1
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec; targetmomentvec]
    fprintf('Current objective fn value is %8.12f \n', Obj)
end











end