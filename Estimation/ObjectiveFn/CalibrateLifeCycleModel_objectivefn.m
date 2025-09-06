function Obj=CalibrateLifeCycleModel_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, ReturnFnParamNames, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, ParametrizeParamsFn, FnsToEvaluate, FnsToEvaluateParamNames, usingallstats, usinglcp,targetmomentvec, allstatmomentnames, acsmomentnames, allstatcummomentsizes, acscummomentsizes, AllStats_whichstats, ACStats_whichstats, calibparamsvecindex, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.

% Untransform the parameters (when dealing with constraints the inputs are the transformed parameters, so want to switch them back to original model parameters)
[calibparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(calibparamsvec,calibparamsvecindex,CalibParamNames,caliboptions);

if caliboptions.verbose==1
    fprintf(' \n')
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
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix(:,sum(calibomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        Parameters.(CalibParamNames{pp})=currparamraw;
    else
        Parameters.(CalibParamNames{pp})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
    end
end

%% ParametrizeParamsFn can be used to parametrize the parameters
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Do grids if those depend on parameters being calibrated (otherwise they are already done)
if caliboptions.calibrateshocks==1
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz(n_z,z_grid,pi_z,N_j,Parameters,vfoptions,3);
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
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
    if isempty(allstatmomentnames{1,3})
        currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2});
    else
        currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2}).(allstatmomentnames{1,3});
    end
    for cc=2:size(allstatmomentnames,1)
        if isempty(allstatmomentnames{cc,3})
            currentmomentvec(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2});
        else
            currentmomentvec(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2}).(allstatmomentnames{cc,3});
        end
    end
end
if usinglcp==1
    if isempty(acsmomentnames{1,3})
        currentmomentvec(allstatcummomentsizes(end)+1:allstatcummomentsizes(end)+acscummomentsizes(1))=AgeConditionalStats.(acsmomentnames{1,1}).(acsmomentnames{1,2});
    else
        currentmomentvec(allstatcummomentsizes(end)+1:allstatcummomentsizes(end)+acscummomentsizes(1))=AgeConditionalStats.(acsmomentnames{1,1}).(acsmomentnames{1,2}).(acsmomentnames{1,3});
    end
    for cc=2:size(acsmomentnames,1)
        if isempty(acsmomentnames{cc,3})
            currentmomentvec(allstatcummomentsizes(end)+acscummomentsizes(cc-1)+1:allstatcummomentsizes(end)+acscummomentsizes(cc))=AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2});
        else
            currentmomentvec(allstatcummomentsizes(end)+acscummomentsizes(cc-1)+1:allstatcummomentsizes(end)+acscummomentsizes(cc))=AgeConditionalStats.(acsmomentnames{cc,1}).(acsmomentnames{cc,2}).(acsmomentnames{cc,3});
        end
    end
end

%% Option to log moments (if targets are log, then this will have been already applied)
if any(caliboptions.logmoments>0) % need to log some moments
    currentmomentvec=(1-caliboptions.logmoments).*currentmomentvec + caliboptions.logmoments.*log(currentmomentvec.*caliboptions.logmoments+(1-caliboptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Evaluate the objective function (which is being minimized)
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if caliboptions.vectoroutput==1 % vector output
    % Output the vector of currentmomentvec
    % Main use it for computing derivatives of moments with respect to parameters
    Obj=currentmomentvec(actualtarget);
elseif caliboptions.vectoroutput==0 % scalar output
    % currentmomentvec is the current moment values
    % targetmomentvec is the target moment values
    % Both are column vectors

    % Note: MethodOfMoments and sum_squared are doing essentially the same calculation (only different is size of weights, 
    % which will be a matrix for MethodOfMoments but a vector for sum_squared), I just write them in ways that make it more 
    % obvious that they do what they say.
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Obj=(targetmomentvec-currentmomentvec)'*caliboptions.weights*(targetmomentvec-currentmomentvec);
        % For the purpose of doing log(moments) I switched to the following (otherwise getting silly current moments can seem attractive)
        Obj=(currentmomentvec(actualtarget)-targetmomentvec(actualtarget))'*caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj=sum(caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget)).^2,'omitnan');
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj=sum(caliboptions.weights.*(log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget)).^2),'omitnan');
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj=Obj/length(CalibParamNames); % This is done so that the tolerances for convergence are sensible

    if penalty>0
        if Obj>0
            Obj=1.2*penalty*Obj; % 20% penalty for being too far in violation of restrictions
        else % Obj is negative, so penalty is to reduce magnitude
            Obj=0.8*(1/penalty)*Obj; % 20% penalty for being too far in violation of restrictions
        end
    end
elseif caliboptions.vectoroutput==2
    % Weighted vector (for use with least-squares residuals algorithms)
    % Note: the outer-layers of code already took 'square root' of the weights
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Is essentially the square-root of 'MethodOfMoments' [it is the form of input used by Matlab's lsqnonlin()]
        Obj=caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj=caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj=caliboptions.weights.*log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget));
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj=gather(Obj); % lsqnonlin() doesn't work with gpu, so have to gather()
end



%% Verbose
if caliboptions.verbose==1 && caliboptions.vectoroutput==0
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    fprintf('Current objective fn value is %8.12f \n', Obj)
    if penalty>0
        if Obj>0
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 1.2*penalty)
        else  % Obj is negative, so penalty is to reduce magnitude
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 0.8*(1/penalty) )
        end
    end
elseif caliboptions.verbose==1 && caliboptions.vectoroutput==2
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    fprintf('Current (sum-of-squares of) objective fn value is %8.12f \n', Obj'*Obj)
    if penalty>0
        if Obj>0
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 1.2*penalty)
        else  % Obj is negative, so penalty is to reduce magnitude
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 0.8*(1/penalty) )
        end
    end
end











end