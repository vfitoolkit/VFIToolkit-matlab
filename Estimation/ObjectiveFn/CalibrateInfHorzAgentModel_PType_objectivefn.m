function Obj=CalibrateInfHorzAgentModel_PType_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,Names_i,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, usingallstats,usingautocorr,usingcrosssec,usingcustomstats, targetmomentvec, allstatmomentnames,autocorrmomentnames,crosssecmomentnames,cmsmomentnames, allstatcummomentsizes,autocorrcummomentsizes,crossseccummomentsizes,cmscummomentsizes, AllStats_whichstats,AutoCorrStats_whichstats,CrossSecStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_AutoCorrStats, FnsToEvaluate_CrossSecStats, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats and
% LifeCycleProfiles. Lastly there is caliboptions.

% Untransform the parameters (when dealing with constraints the inputs are the transformed parameters, so want to switch them back to original model parameters)
[calibparamsvec,penalty]=ParameterConstraints_TransformParamsToOriginal(calibparamsvec,calibparamsvecindex,CalibParamNames,caliboptions);
% Note: ptype makes no difference to this.

if caliboptions.verbose==1
    fprintf(' \n')
    fprintf('Current parameter values: \n')
    for pp=1:nCalibParams
        if nCalibParamsFinder(pp,2)==0 % parameter does not depend on ptype
            if calibparamsvecindex(pp+1)-calibparamsvecindex(pp)==1
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'= %8.6f \n'],calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))
            else
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'=  \n'])
                calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))' % want the output as a row
            end
        else  % parameter depends on ptype
            if calibparamsvecindex(pp+1)-calibparamsvecindex(pp)==1
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'.',Names_i{nCalibParamsFinder(pp,2)},'= %8.6f \n'],calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)))
            else
                fprintf(['    ',CalibParamNames{nCalibParamsFinder(pp,1)},'.',Names_i{nCalibParamsFinder(pp,2)},'=  \n'])
                calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1))' % want the output as a row
            end
        end
    end
end

for pp=1:nCalibParams
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix(:,sum(calibomitparams_counter(1:pp)));
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        if nCalibParamsFinder(pp,2)==0 % parameter does not depend on ptype
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)})=reshape(currparamraw,calibparamssizes(pp,:));
        else
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)})=reshape(currparamraw,calibparamssizes(pp,:));
        end
    else
        if nCalibParamsFinder(pp,2)==0 % parameter does not depend on ptype
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)})=reshape(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)),calibparamssizes(pp,:));
        else
            Parameters.(CalibParamNames{nCalibParamsFinder(pp,1)}).(Names_i{nCalibParamsFinder(pp,2)})=reshape(calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1)),calibparamssizes(pp,:));
        end
    end
end

%% ParametrizeParamsFn can be used to parametrize the parameters (including the distribution of permanent types)
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Do grids if those depend on parameters being calibrated (otherwise they are already done)
if caliboptions.calibrateshocks==1
    % Internally, only ever use joint-grids (makes all the code much easier to write)
    [z_gridvals, pi_z, vfoptions]=ExogShockSetup(n_z,z_gridvals,pi_z,Parameters,vfoptions,3);
    % output: z_gridvals, pi_z, vfoptions.e_gridvals, vfoptions.pi_e
    simoptions.e_gridvals=vfoptions.e_gridvals;
    simoptions.pi_e=vfoptions.pi_e;
end

%% Solve the model and calculate the stats
[V, Policy]=ValueFnIter_Case1_PType(n_d,n_a,n_z,Names_i,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

StationaryDist=StationaryDist_Case1_PType(PTypeDistParamNames,Policy,n_d,n_a,n_z,Names_i,pi_z,Parameters,simoptions);

%% Custom Model Stats
if usingcustomstats==1
    CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,Names_i,d_grid,a_grid,caliboptions.CustomModelStatsInputs.z_grid,caliboptions.CustomModelStatsInputs.pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
end

%% Calculate model stats
if usingallstats==1
    simoptions.whichstats=AllStats_whichstats;
    AllStats=EvalFnOnAgentDist_AllStats_Case1_PType(StationaryDist,Policy, FnsToEvaluate_AllStats,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,simoptions);
end
if usingautocorr==1
    error('Have not yet implemented EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz_PType; ask on forum if you want/need this')
    % simoptions.whichstats=AutoCorrStats_whichstats;
    % AutoCorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz_PType(StationaryDist,Policy,FnsToEvaluate_AutoCorrStats,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,pi_z,simoptions);
end
if usingcrosssec==1
    error('Have not yet implemented EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz_PType; ask on forum if you want/need this')
    % simoptions.whichstats=CrossSecStats_whichstats;
    % CrossSectionCovarCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz_PType(StationaryDist,Policy,FnsToEvaluate_CrossSecStats,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,simoptions);
end


%% Get current values of the target moments as a vector
currentmomentvec=zeros(size(targetmomentvec));
if usingallstats==1
    if isempty(allstatmomentnames{1,3})
        currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2});
    else
        if isempty(allstatmomentnames{1,4})
            currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2}).(allstatmomentnames{1,3});
        else
            currentmomentvec(1:allstatcummomentsizes(1))=AllStats.(allstatmomentnames{1,1}).(allstatmomentnames{1,2}).(allstatmomentnames{1,3}).(allstatmomentnames{1,4});        
        end
    end
    for cc=2:size(allstatmomentnames,1)
        if isempty(allstatmomentnames{cc,3})
            currentmomentvec(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2});
        else
            if isempty(allstatmomentnames{cc,4})
                currentmomentvec(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2}).(allstatmomentnames{cc,3});
            else
                currentmomentvec(allstatcummomentsizes(cc-1)+1:allstatcummomentsizes(cc))=AllStats.(allstatmomentnames{cc,1}).(allstatmomentnames{cc,2}).(allstatmomentnames{cc,3}).(allstatmomentnames{cc,4});
            end
        end
    end
end
if usingautocorr==1
    sofar=allstatcummomentsizes(end);
    if isempty(autocorrmomentnames{1,3})
        currentmomentvec(sofar+1:sofar+autocorrcummomentsizes(1))=AutoCorrTransProbs.(autocorrmomentnames{1,1}).(autocorrmomentnames{1,2});
    else
        currentmomentvec(sofar+1:sofar+autocorrcummomentsizes(1))=AutoCorrTransProbs.(autocorrmomentnames{1,1}).(autocorrmomentnames{1,2}).(autocorrmomentnames{1,3});
    end
    for cc=2:size(autocorrmomentnames,1)
        if isempty(autocorrmomentnames{cc,3})
            currentmomentvec(sofar+autocorrcummomentsizes(cc-1)+1:sofar+autocorrcummomentsizes(cc))=AutoCorrTransProbs.(autocorrmomentnames{cc,1}).(autocorrmomentnames{cc,2});
        else
            currentmomentvec(sofar+autocorrcummomentsizes(cc-1)+1:sofar+autocorrcummomentsizes(cc))=AutoCorrTransProbs.(autocorrmomentnames{cc,1}).(autocorrmomentnames{cc,2}).(autocorrmomentnames{cc,3});
        end
    end
end
if usingcrosssec==1
    sofar=allstatcummomentsizes(end)+autocorrcummomentsizes(end);
    if isempty(crosssecmomentnames{1,3})
        currentmomentvec(sofar+1:sofar+crossseccummomentsizes(1))=CrossSectionCovarCorr.(crosssecmomentnames{1,1}).(crosssecmomentnames{1,2}).(crosssecmomentnames{1,3});
    elseif isempty(crosssecmomentnames{1,4})
        currentmomentvec(sofar+1:sofar+crossseccummomentsizes(1))=CrossSectionCovarCorr.(crosssecmomentnames{1,1}).(crosssecmomentnames{1,2}).(crosssecmomentnames{1,3});
    else
        currentmomentvec(sofar+1:sofar+crossseccummomentsizes(1))=CrossSectionCovarCorr.(crosssecmomentnames{1,1}).(crosssecmomentnames{1,2}).(crosssecmomentnames{1,3}).(crosssecmomentnames{1,4});
    end
    for cc=2:size(crosssecmomentnames,1)
        if isempty(crosssecmomentnames{cc,3})
            currentmomentvec(sofar+crossseccummomentsizes(cc-1)+1:sofar+crossseccummomentsizes(cc))=CrossSectionCovarCorr.(crosssecmomentnames{cc,1}).(crosssecmomentnames{cc,2});
        elseif isempty(crosssecmomentnames{cc,4})
            currentmomentvec(sofar+crossseccummomentsizes(cc-1)+1:sofar+crossseccummomentsizes(cc))=CrossSectionCovarCorr.(crosssecmomentnames{cc,1}).(crosssecmomentnames{cc,2}).(crosssecmomentnames{cc,3});
        else
            currentmomentvec(sofar+crossseccummomentsizes(cc-1)+1:sofar+crossseccummomentsizes(cc))=CrossSectionCovarCorr.(crosssecmomentnames{cc,1}).(crosssecmomentnames{cc,2}).(crosssecmomentnames{cc,3}).(crosssecmomentnames{cc,4});
        end
    end
end
if usingcustomstats==1
    sofar=allstatcummomentsizes(end)+autocorrcummomentsizes(end)+crossseccummomentsizes(end);
    currentmomentvec(sofar+1:sofar+cmscummomentsizes(1))=CustomStats.(cmsmomentnames{1,1});
    for cc=2:size(cmsmomentnames,1)
        currentmomentvec(sofar+cmscummomentsizes(cc-1)+1:sofar+cmscummomentsizes(cc))=CustomStats.(cmsmomentnames{cc,1});
    end
end


%% Option to log moments (if targets are log, then this will have been already applied)
if any(caliboptions.logmoments>0) % need to log some moments
    currentmomentvec=(1-caliboptions.logmoments).*currentmomentvec + caliboptions.logmoments.*log(currentmomentvec.*caliboptions.logmoments+(1-caliboptions.logmoments)); % Note: take log, and for those we don't log I end up taking log(1) (which becomes zero and so disappears)
end


%% Evaluate the objective function (which is being minimized)
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if caliboptions.vectoroutput==1
    % Output the vector of currentmomentvec
    % Main use it for computing derivatives of moments with respect to parameters
    Obj=currentmomentvec(actualtarget);
elseif caliboptions.vectoroutput==0 % scalar output
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
if caliboptions.verbose==1
    if usingcustomstats==1
        fprintf('Current CustomModelStats variables: \n')
        for ii=1:length(cmsmomentnames)
            fprintf('	%s: %8.4f \n',cmsmomentnames{ii},CustomStats.(cmsmomentnames{ii}))
        end
    end
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    if caliboptions.vectoroutput==0
        fprintf('Current objective fn value is %8.12f \n', Obj)
    elseif caliboptions.vectoroutput==2
        fprintf('Current (sum-of-squares of) objective fn value is %8.12f \n', Obj'*Obj)
    end
    if penalty>0
        if Obj>0
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 1.2*penalty)
        else  % Obj is negative, so penalty is to reduce magnitude
            fprintf('Current penalty is to multiply objective fn by %8.2f \n', 0.8*(1/penalty) )
        end
    end
end









end