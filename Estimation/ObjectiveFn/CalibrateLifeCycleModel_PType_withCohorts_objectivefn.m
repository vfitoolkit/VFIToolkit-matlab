function Obj=CalibrateLifeCycleModel_PType_withCohorts_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, jequaloneDist,AgeWeightParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, targetmomentvec, cohortmoments, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, vfoptions,simoptions)
% As CalibrateLifeCycleModel_PType_objectivefn, but with cohorts entering the model at different ages
% (caliboptions.cohortagejshifter): jequaloneDist is a cell over cohorts, each a structure over ptypes; each cohort
% gets its own agent distribution (StationaryDist_Case1_FHorz_PType with simoptions.jequaloneDistAge, and with the
% ptype weights scaled by the ptypes' masses in that cohort) and its own statistics, and the moments are stacked
% cohort by cohort as set up by SetupTargetMoments_FHorz_withCohorts (which gives cohortmoments).
% Age weights by cohort: simoptions.agemass_withCohort (ncohorts-by-N_j, or a structure of these by ptype) if given,
% otherwise cohort c keeps the AgeWeightParamNames value of its entry age at every age (by ptype if that parameter
% is a structure).
% Not implemented with cohorts: CustomModelStats, bootstrapped standard errors.
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
    % Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
    [z_gridvals_J, pi_z_J, vfoptions]=ExogShockSetup_FHorz_PType(n_z,z_gridvals_J,pi_z_J,N_j,Names_i,Parameters,vfoptions,3);
    % output: z_gridvals_J, pi_z_J, vfoptions.e_gridvals_J, vfoptions.pi_e_J
    simoptions.e_gridvals_J=vfoptions.e_gridvals_J;
    simoptions.pi_e_J=vfoptions.pi_e_J;
end

% Same for semi-exogenous shocks
if caliboptions.calibsemiexo==1
    vfoptions=SemiExogShockSetup_FHorz_PType(n_d,N_j,Names_i,d_grid,Parameters,vfoptions,3);
    simoptions.semiz_gridvals_J=vfoptions.semiz_gridvals_J;
    simoptions.pi_semiz_J=vfoptions.pi_semiz_J;
end


%% Solve the model
[~, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

%% Age weights and ptype weights by cohort
ncohorts=caliboptions.ncohorts;
N_i=length(Names_i);
ptweights_orig=Parameters.(PTypeDistParamNames{1}); % may come from ParametrizeParamsFn, so take it here each time
ptweights0=ptweights_orig(:);
if isempty(simoptions.agemass_withCohort)
    if isstruct(Parameters.(AgeWeightParamNames{1}))
        mewjc=struct();
        for ii=1:N_i
            temp=Parameters.(AgeWeightParamNames{1}).(Names_i{ii});
            mewjc.(Names_i{ii})=zeros(ncohorts,N_j);
            for cc=1:ncohorts
                mewjc.(Names_i{ii})(cc,:)=temp(caliboptions.cohortagejshifter(cc))*ones(1,N_j);
            end
        end
    else
        temp=Parameters.(AgeWeightParamNames{1});
        mewjc=zeros(ncohorts,N_j);
        for cc=1:ncohorts
            mewjc(cc,:)=temp(caliboptions.cohortagejshifter(cc))*ones(1,N_j);
        end
    end
else
    mewjc=simoptions.agemass_withCohort;
end
simoptions.agemass_withCohort=mewjc; % so StationaryDist_FHorz_Case1 knows the age weights are cohort masses (and do not sum to one); PType_Options passes it through by ptype if a structure

%% Agent distribution and statistics, cohort by cohort, and the current values of the target moments as a vector
currentmomentvec=zeros(size(targetmomentvec));
for cc=1:ncohorts
    simoptions.jequaloneDistAge=caliboptions.cohortagejshifter(cc);
    if isstruct(mewjc)
        for ii=1:N_i
            Parameters.(AgeWeightParamNames{1}).(Names_i{ii})=mewjc.(Names_i{ii})(cc,:);
        end
    else
        Parameters.(AgeWeightParamNames{1})=mewjc(cc,:);
    end
    % ptype weights for this cohort: the ptype's overall weight times its mass in this cohort (a ptype with no mass in this cohort gets weight zero)
    ptw=ptweights0.*caliboptions.cohortmasses(:,cc);
    Parameters.(PTypeDistParamNames{1})=ptw/sum(ptw);
    StationaryDist=StationaryDist_Case1_FHorz_PType(jequaloneDist{cc},AgeWeightParamNames,PTypeDistParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,pi_z_J,Parameters,simoptions);
    sofar=cohortmoments.offset(cc);
    if cohortmoments.usingallstats(cc)==1
        simoptions.whichstats=cohortmoments.AllStats_whichstats{cc};
        AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist,Policy,cohortmoments.FnsToEvaluate_AllStats{cc},Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions);
        allstatmomentnames=cohortmoments.allstatmomentnames{cc};
        allstatcummomentsizes=cohortmoments.allstatcummomentsizes{cc};
        for mm=1:size(allstatmomentnames,1)
            if mm==1
                ind1=sofar+1;
            else
                ind1=sofar+allstatcummomentsizes(mm-1)+1;
            end
            ind2=sofar+allstatcummomentsizes(mm);
            if isempty(allstatmomentnames{mm,3})
                currentmomentvec(ind1:ind2)=AllStats.(allstatmomentnames{mm,1}).(allstatmomentnames{mm,2});
            else
                if isempty(allstatmomentnames{mm,4})
                    currentmomentvec(ind1:ind2)=AllStats.(allstatmomentnames{mm,1}).(allstatmomentnames{mm,2}).(allstatmomentnames{mm,3});
                else
                    currentmomentvec(ind1:ind2)=AllStats.(allstatmomentnames{mm,1}).(allstatmomentnames{mm,2}).(allstatmomentnames{mm,3}).(allstatmomentnames{mm,4});
                end
            end
        end
        sofar=sofar+allstatcummomentsizes(end);
    end
    if cohortmoments.usinglcp(cc)==1
        simoptions.whichstats=cohortmoments.ACStats_whichstats{cc};
        AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist,Policy,cohortmoments.FnsToEvaluate_ACStats{cc},Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_gridvals_J,simoptions);
        acsmomentnames=cohortmoments.acsmomentnames{cc};
        acscummomentsizes=cohortmoments.acscummomentsizes{cc};
        for mm=1:size(acsmomentnames,1)
            if mm==1
                ind1=sofar+1;
            else
                ind1=sofar+acscummomentsizes(mm-1)+1;
            end
            ind2=sofar+acscummomentsizes(mm);
            if isempty(acsmomentnames{mm,3})
                currentmomentvec(ind1:ind2)=AgeConditionalStats.(acsmomentnames{mm,1}).(acsmomentnames{mm,2});
            else
                if isempty(acsmomentnames{mm,4})
                    currentmomentvec(ind1:ind2)=AgeConditionalStats.(acsmomentnames{mm,1}).(acsmomentnames{mm,2}).(acsmomentnames{mm,3});
                else
                    currentmomentvec(ind1:ind2)=AgeConditionalStats.(acsmomentnames{mm,1}).(acsmomentnames{mm,2}).(acsmomentnames{mm,3}).(acsmomentnames{mm,4});
                end
            end
        end
    end
end
Parameters.(PTypeDistParamNames{1})=ptweights_orig; % restore

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
    if penalty>0
        Obj=[Obj; sqrt(penalty)]; % append penalty residual: contributes 'penalty' to lsqnonlin's sum(Obj.^2)
    else
        Obj=[Obj; 0]; % penalty=0 (params inside cutoffs); append 0 to keep residual length constant
    end
end


%% Verbose
if caliboptions.verbose==1
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