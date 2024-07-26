function Obj=CalibrateLifeCycleModel_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, FnsToEvaluate, FnsToEvaluateParamNames, usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, calibparamsvecindex, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.

% Do any transformations of parameters before we say what they are
for pp=1:length(CalibParamNames)
    if caliboptions.constrainpositive(pp)==1 % Forcing this parameter to be positive
        % Constrain parameter to be positive (be working with log(parameter) and then always take exp() before inputting to model)
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=exp(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))).*(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))>-5); % Note: I force anything less that -50 to evaluate to exp(-50)~=0 rounding off what would otherwise be 10e-22, as otherwise can get stuck wandering around crazy negative numbers 
    elseif caliboptions.constrain0to1(pp)==1
        temp=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        % Constrain parameter to be 0 to 1 (be working with x=log(p/(1-p)), where p is parameter) then always take 1/(1+exp(-x)) before inputting to model
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=1/(1+exp(-calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))));
        % Note: This does not include the endpoints of 0 and 1 as 1/(1+exp(-x)) maps from the Real line into the open interval (0,1)
        %       R is not compact, and [0,1] is compact, so cannot have a continuous bijection (one-to-one and onto) function from R into [0,1].
        %       So I settle for a function from R to (0,1) and then trim ends of R to give 0 and 1, like I do for constrainpositive I use +-50 as the cutoffs
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)).*(temp>-50); % set values less than -50 to zero
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)).*(1-(temp>50))+(temp>50); % set values greater than 50 to one
    end
    % Note: sometimes, need to do both of constrainAtoB and constrain0to1, so cannot use elseif
    if caliboptions.constrainAtoB(pp)==1
        % Constrain parameter to be A to B
        calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))=caliboptions.constrainAtoBlimits(pp,1)+(caliboptions.constrainAtoBlimits(pp,2)-caliboptions.constrainAtoBlimits(pp,1))*calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        % Note, this parameter will have first been converted to 0 to 1 already, so just need to further make it A to B
        % y=A+(B-A)*x, converts 0-to-1 x, into A-to-B y
    end
end

if caliboptions.verbose==1 && caliboptions.vectoroutput==0
    fprintf('Current parameter values: \n')
    for pp=1:length(CalibParamNames)
        if calibparamsvecindex(pp+1)-calibparamsvecindex(pp)==1
            fprintf(['    ',CalibParamNames{pp},'= %8.6f \n'],calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))
        else
            fprintf(['    ',CalibParamNames{pp},'=  \n'])
            calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))' % want the output as a row
        end
    end
end

for pp=1:length(CalibParamNames)
    Parameters.(CalibParamNames{pp})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
end


%% Solve the model and calculate the stats
[~, Policy]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);

StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z_J,Parameters,simoptions);

if caliboptions.simulatemoments==0
    if usingallstats==1
        simoptions.whichstats=AllStats_whichstats;
        AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,simoptions);
    end
    if usinglcp==1
        simoptions.whichstats=ACStats_whichstats;
        AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_gridvals_J,simoptions);
    end
elseif caliboptions.simulatemoments==1
    % Do a panel data simulation.
    % Not used in calibration, but needed for estimation when bootstrapping standard errors
    % Set random number generator seed to estimoptions.rngindex
    rng(estimoptions.rngindex) % Often, each time the objectivefn is evaluated we want to be able to use same random number sequence
    simPanelValues=SimPanelValues_FHorz_Case1(jequaloneDist,Policy,FnsToEvaluate,Parameters,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_J,pi_z_J,simoptions);
    % Compute the moments (same as CalibrateLifeCycleModel_objectivefn(), except the panel data versions)
    if usingallstats==1
        simoptions.whichstats=AllStats_whichstats;
        AllStats=PanelValues_AllStats_FHorz(simPanelValues,simoptions);
    end
    if usinglcp==1
        simoptions.whichstats=ACStats_whichstats;
        AgeConditionalStats=PanelValues_LifeCycleProfiles_FHorz(simPanelValues,N_j,simoptions);
    end
end


%% Get current values of the target moments as a vector
currentmomentvec=zeros(size(targetmomentvec));
if usingallstats==1
    currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2});
    for cc=2:size(allstatmomentnames,1)
        currentmomentvec(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2});
    end
end
if usinglcp==1
    currentmomentvec(allstatcummomentsizes(end)+1:allstatcummomentsizes(end)+acscummomentsizes(1))=AgeConditionalStats.(acsmomentnames{1,1}).(acsmomentnames{1,2});
    for cc=2:size(acsmomentnames,1)
        currentmomentvec(allstatcummomentsizes(end)+acscummomentsizes(cc-1)+1:allstatcummomentsizes(end)+acscummomentsizes(cc))=AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2});
    end
end

%% Option to log moments (if targets are log, then this will have been already applied)
if any(caliboptions.logmoments>0) % need to log some moments
    currentmomentvec=(1-caliboptions.logmoments).*currentmomentvec + caliboptions.logmoments.*log(currentmomentvec.*caliboptions.logmoments+(1-caliboptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Evaluate the objective function
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if caliboptions.vectoroutput==1
    % Output the vector of currentmomentvec
    % This is only used to get standard deviations of parameters as part of
    % method of moments estimation (rather than writing a whole new
    % function), it is not what you want most of the time.
    Obj=currentmomentvec(actualtarget);
else
    % currentmomentvec is the current moment values
    % targetmomentvec is the target moment values
    % Both are column vectors

    % Note: MethodOfMoments and sum_squared are doing the same calculation, I
    % just write them in ways that make it more obvious that they do what they say.
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Obj=(targetmomentvec-currentmomentvec)'*caliboptions.weights*(targetmomentvec-currentmomentvec);
        % For the purpose of doing log(moments) I switched to the following
        % (otherwise getting silly current moments can seem attractive)
        Obj=(currentmomentvec(actualtarget)-targetmomentvec(actualtarget))'*caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj=sum(caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget)).^2,[],'omitnan');
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj=sum(caliboptions.weights.*(log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget)).^2),[],'omitnan');
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj=Obj/length(CalibParamNames); % This is done so that the tolerances for convergence are sensible
end



%% Verbose
if caliboptions.verbose==1 && caliboptions.vectoroutput==0
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    fprintf('Current objective fn value is %8.12f \n', Obj)
end












end