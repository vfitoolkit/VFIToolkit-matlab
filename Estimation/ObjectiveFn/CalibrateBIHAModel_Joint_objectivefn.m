function [Obj,GEout]=CalibrateBIHAModel_Joint_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, GEPriceParamNames, ParametrizeParamsFn, FnsToEvaluate, GeneralEqmEqnsCell, GEeqnNames, GeneralEqmEqnParamNames, usingallstats,usingautocorr,usingcrosssec,usingcustomstats, targetmomentvec, allstatmomentnames,autocorrmomentnames,crosssecmomentnames,cmsmomentnames, allstatcummomentsizes,autocorrcummomentsizes,crossseccummomentsizes,cmscummomentsizes, AllStats_whichstats,AutoCorrStats_whichstats,CrossSecStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_AutoCorrStats, FnsToEvaluate_CrossSecStats, nCalibParams, calibparamsvecindex, calibomitparams_counter, calibomitparamsmatrix, caliboptions, heteroagentoptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats
% AutoCorrTransProbs and CrossSectionCovarCorr. Lastly there is caliboptions.
%
% Joint: the GE prices are the last entries of CalibParamNames (calibrated alongside the other parameters), and the GE conditions
% are part of the objective.
% Second output GEout: .prices (the GE prices, in the order of GEPriceParamNames) and .conditions (the GE conditions, in the order of
% GEeqnNames), both column vectors.
% vectoroutput=1 outputs the model moments at the targeted entries followed by the GE conditions.

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

for pp=1:nCalibParams
    if calibomitparams_counter(pp)>0
        currparamraw=calibomitparamsmatrix{sum(calibomitparams_counter(1:pp))}; % (a cell: one omitted-values vector per masked parameter, of any length)
        currparamraw(isnan(currparamraw))=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
        Parameters.(CalibParamNames{pp})=currparamraw;
    else
        Parameters.(CalibParamNames{pp})=calibparamsvec(calibparamsvecindex(pp)+1:calibparamsvecindex(pp+1));
    end
end

%% ParametrizeParamsFn can be used to parametrize the parameters (including the distribution of permanent types)
if ~isempty(ParametrizeParamsFn)
    Parameters=ParametrizeParamsFn(Parameters);
end

%% Do grids if those depend on parameters being calibrated (otherwise they are already done)
if caliboptions.calibrateshocks==1
    % Internally, only ever use joint-grids (makes all the code much easier to write)
    % The user's own grids are needed if either CustomModelStats is given them
    KeepOriginalGrid=((usingcustomstats==1 && caliboptions.CustomModelStats_usergrids==1) || (heteroagentoptions.useCustomModelStats==1 && heteroagentoptions.CustomModelStats_usergrids==1));
    [z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz(n_z,z_gridvals,pi_z,Parameters,vfoptions,3,KeepOriginalGrid);
    % output: z_gridvals, pi_z, vfoptions.e_gridvals, vfoptions.pi_e
    simoptions.e_gridvals=vfoptions.e_gridvals;
    simoptions.pi_e=vfoptions.pi_e;
end

%% Solve the model
% (V is kept: either CustomModelStats, from caliboptions or from heteroagentoptions, takes it)
[V, Policy]=ValueFnIter_InfHorz(n_d,n_a,n_z,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames,[], vfoptions);

StationaryDist=StationaryDist_InfHorz(Policy,n_d,n_a,n_z,pi_z,simoptions,Parameters);

% Note: I could make it a bit faster by just doing these AggVars for the general eqm as part of the AllStats
AggVars=EvalFnOnAgentDist_AggVars_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters,[],n_d,n_a,n_z,d_grid, a_grid, z_gridvals,simoptions);

AggVarNames=fieldnames(AggVars);
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars.(AggVarNames{aa}).Mean;
end

%% Custom Model Stats
if usingcustomstats==1
    if caliboptions.CustomModelStats_usergrids==0
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    elseif caliboptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            caliboptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            caliboptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,caliboptions.CustomModelStatsInputs.z_grid,caliboptions.CustomModelStatsInputs.pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    end
end
% Custom Model Stats can be in heteroagent options or caliboptions so when doing joint-optimization we need to handle both possibilities
if heteroagentoptions.useCustomModelStats==1
    if heteroagentoptions.CustomModelStats_usergrids==0
        CustomStats2=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    elseif heteroagentoptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            heteroagentoptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            heteroagentoptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats2=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,d_grid,a_grid,heteroagentoptions.CustomModelStatsInputs.z_grid,heteroagentoptions.CustomModelStatsInputs.pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    end
    % Note: anything else you want, just 'hide' it in heteroagentoptions
    customstatnames=fieldnames(CustomStats2);
    for pp=1:length(customstatnames)
        Parameters.(customstatnames{pp})=CustomStats2.(customstatnames{pp});
    end
end

%% intermediateEqns
if heteroagentoptions.useintermediateEqns==1
    for aa=1:heteroagentoptions.nIntermediateEqns
        Parameters.(heteroagentoptions.intermediateEqnNames{aa})=real(GeneralEqmConditions_Case1_v3(heteroagentoptions.intermediateEqnsCell{aa},heteroagentoptions.intermediateEqnParamNames(aa).Names,Parameters));
    end
end

%% Evaluate General Eqm Eqns
GeneralEqmConditionsVec=zeros(1,length(GEeqnNames));
for gg=1:length(GEeqnNames)
    % use of real() is a hack that could disguise errors, but I couldn't find why matlab was treating output as complex
    GeneralEqmConditionsVec(gg)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
end
GEout.prices=zeros(length(GEPriceParamNames),1);
for pp=1:length(GEPriceParamNames)
    GEout.prices(pp)=gather(Parameters.(GEPriceParamNames{pp}));
end
GEout.conditions=gather(GeneralEqmConditionsVec');

%% Calculate model stats
% caliboptions.whichcombos=1 (CalibrateBIHAModel's default): the stats commands compute only the targeted (function, statistic,
% restriction) combinations, through simoptions.whichcombos (and, for AllStats, a per-combination simoptions.whichstats) built by
% SetupTargetMoments_InfHorz (caliboptions.selectors). =0: every statistic of every targeted function (whichstats all ones). The three
% commands take differently shaped selectors, so each gets its own copy of simoptions. The conditional restrictions stay in simoptions
% for all three (restricted AutoCorr and cross-section targets are allowed since 2026-10-08; the restrictions used to be stripped
% before those two commands).
if usingallstats==1
    simoptions_AllStats=simoptions;
    if caliboptions.whichcombos==1
        simoptions_AllStats.whichcombos=caliboptions.selectors.AllStats.whichcombos;
        simoptions_AllStats.whichstats=caliboptions.selectors.AllStats.whichstats;
    else % caliboptions.whichcombos=0: every statistic of every targeted function
        simoptions_AllStats.whichstats=ones(1,7);
    end
    AllStats=EvalFnOnAgentDist_AllStats_InfHorz(StationaryDist,Policy, FnsToEvaluate_AllStats,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_gridvals,simoptions_AllStats);
end
if usingautocorr==1
    simoptions_AutoCorr=simoptions;
    if isfield(simoptions,'timehorizons')
        simoptions_AutoCorr.timehorizons=union(simoptions.timehorizons,caliboptions.autocorrtimehorizons); % the horizons the targets name (their _kK suffixes), plus any the user asked for
    else
        simoptions_AutoCorr.timehorizons=caliboptions.autocorrtimehorizons;
    end
    if caliboptions.whichcombos==1
        simoptions_AutoCorr.whichcombos=caliboptions.selectors.AutoCorr.whichcombos; % [nFns,1+nRestr] (the command has no per-combination whichstats; it reports its byproducts)
    end
    AutoCorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz(StationaryDist,Policy,FnsToEvaluate_AutoCorrStats,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_gridvals,pi_z,simoptions_AutoCorr);
end
if usingcrosssec==1
    simoptions_CrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_CrossSec.whichcombos=caliboptions.selectors.CrossSec.whichcombos; % pair-shaped, [nFns,nFns,1+nRestr] (the restricted targets are on pages 2:end)
    end
    CrossSectionCovarCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz(StationaryDist,Policy,FnsToEvaluate_CrossSecStats,Parameters,[],n_d,n_a,n_z,d_grid,a_grid,z_gridvals,simoptions_CrossSec);
end


%% Get current values of the target moments as a vector
% Each kind: walk the names of the target (one to four levels, the empty cells skipped) into the command's output, which has the same
% nesting (restrictions, MoreInequality, CovarianceWith/CorrelationWith and the matrices alike). An AutoCorr statistic with a _kK
% horizon suffix is read from the command's .tperiodsK.(statistic). [Before 2026-10-08 the first cross-section row always read a
% third name, so a two-level target (fn.Mean) as the first cross-section target errored.]
currentmomentvec=zeros(size(targetmomentvec));
sofar=0;
if usingallstats==1
    for cc=1:size(allstatmomentnames,1)
        if cc==1
            idx=sofar+1:sofar+allstatcummomentsizes(1);
        else
            idx=sofar+allstatcummomentsizes(cc-1)+1:sofar+allstatcummomentsizes(cc);
        end
        if all(isnan(targetmomentvec(idx))) % a wholly omitted target: nothing was selected for it, so the command may not have computed (or created) it
            currentmomentvec(idx)=NaN;
            continue
        end
        temp=AllStats;
        for kk=1:size(allstatmomentnames,2)
            if ~isempty(allstatmomentnames{cc,kk})
                temp=temp.(allstatmomentnames{cc,kk});
            end
        end
        currentmomentvec(idx)=temp(:);
    end
    sofar=sofar+allstatcummomentsizes(end);
end
if usingautocorr==1
    for cc=1:size(autocorrmomentnames,1)
        if cc==1
            idx=sofar+1:sofar+autocorrcummomentsizes(1);
        else
            idx=sofar+autocorrcummomentsizes(cc-1)+1:sofar+autocorrcummomentsizes(cc);
        end
        if all(isnan(targetmomentvec(idx))) % a wholly omitted target (e.g. .(restriction).(fn).(stat)=NaN): its combination is off, so the command did not create the struct
            currentmomentvec(idx)=NaN;
            continue
        end
        names=autocorrmomentnames(cc,:);
        names=names(~cellfun(@isempty,names));
        tk=regexp(names{end},'_k(\d+)$','tokens','once'); % a horizon suffix: stat_kK is the command's .tperiodsK.(stat)
        if ~isempty(tk)
            names=[names(1:end-1),{['tperiods',tk{1}]},{names{end}(1:end-length(tk{1})-2)}];
        end
        temp=AutoCorrTransProbs;
        for kk=1:numel(names)
            temp=temp.(names{kk});
        end
        currentmomentvec(idx)=temp(:);
    end
    sofar=sofar+autocorrcummomentsizes(end);
end
if usingcrosssec==1
    for cc=1:size(crosssecmomentnames,1)
        if cc==1
            idx=sofar+1:sofar+crossseccummomentsizes(1);
        else
            idx=sofar+crossseccummomentsizes(cc-1)+1:sofar+crossseccummomentsizes(cc);
        end
        if all(isnan(targetmomentvec(idx))) % a wholly omitted target: nothing was selected for it, so the command may not have created it
            currentmomentvec(idx)=NaN;
            continue
        end
        temp=CrossSectionCovarCorr;
        for kk=1:size(crosssecmomentnames,2)
            if ~isempty(crosssecmomentnames{cc,kk})
                temp=temp.(crosssecmomentnames{cc,kk});
            end
        end
        currentmomentvec(idx)=temp(:);
    end
    sofar=sofar+crossseccummomentsizes(end);
end
if usingcustomstats==1
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
% Create Obj1 for calib targets, then Obj2 for general eqm conditions.
% These are then combined for Obj
actualtarget=(~isnan(targetmomentvec)); % I use NaN to omit targets
if caliboptions.vectoroutput==1
    % Output the vector of currentmomentvec
    % Main use it for computing derivatives of moments with respect to parameters
    Obj1=currentmomentvec(actualtarget);
    Obj2=GeneralEqmConditionsVec;
    Obj=[Obj1; Obj2'];
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
        Obj1=(currentmomentvec(actualtarget)-targetmomentvec(actualtarget))'*caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj1=sum(caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget)).^2,'omitnan');
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj1=sum(caliboptions.weights.*(log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget)).^2),'omitnan');
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end
    Obj1=Obj1/caliboptions.nCalibParamsNonGE; % This is done so that the tolerances for convergence are sensible (the number of calibrated parameters, not counting the GE prices, so the same as in the nested objective)

    if penalty>0
        if Obj1>0
            Obj1=1.2*penalty*Obj1; % 20% penalty for being too far in violation of restrictions
        else % Obj is negative, so penalty is to reduce magnitude
            Obj1=0.8*(1/penalty)*Obj1; % 20% penalty for being too far in violation of restrictions
        end
    end

    % General eqm conditions are always evaluated using sum-of-squares
    Obj2=sum(heteroagentoptions.multiGEweights.*(GeneralEqmConditionsVec.^2));

    Obj=Obj1+caliboptions.relativeGEweight*Obj2;

elseif caliboptions.vectoroutput==2
    % Weighted vector (for use with least-squares residuals algorithms)
    % Note: the outer-layers of code already took 'square root' of the weights
    if strcmp(caliboptions.metric,'MethodOfMoments')
        % Is essentially the square-root of 'MethodOfMoments' [it is the form of input used by Matlab's lsqnonlin()]
        Obj1=caliboptions.weights*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_squared')
        Obj1=caliboptions.weights.*(currentmomentvec(actualtarget)-targetmomentvec(actualtarget));
    elseif strcmp(caliboptions.metric,'sum_logratiosquared')
        Obj1=caliboptions.weights.*log(currentmomentvec(actualtarget)./targetmomentvec(actualtarget));
        % Note: This does the same as using sum_squared together with caliboptions.logmoments=1
    end

    % The square roots of the scalar form's weights and divisor, so that the sum of squares of this vector is exactly the scalar objective
    % (vectoroutput=0), penalty aside [was multiGEweights and relativeGEweight themselves, so the GE conditions entered the sum of squares
    % with their squares, and the targets part was not divided]
    Obj2=sqrt(heteroagentoptions.multiGEweights).*GeneralEqmConditionsVec;

    Obj=[Obj1/sqrt(caliboptions.nCalibParamsNonGE); sqrt(caliboptions.relativeGEweight)*Obj2'];

    Obj=gather(Obj); % lsqnonlin() doesn't work with gpu, so have to gather()
    if penalty>0
        Obj=[Obj; sqrt(penalty)]; % append penalty residual: contributes 'penalty' to lsqnonlin's sum(Obj.^2)
    else
        Obj=[Obj; 0]; % penalty=0 (params inside cutoffs); append 0 to keep residual length constant
    end
end


%% Verbose
if caliboptions.verbose==1
    if usingcustomstats==1
        fprintf('Current CustomModelStats variables (from caliboptions): \n')
        for ii=1:length(cmsmomentnames)
            fprintf('	%s: %8.4f \n',cmsmomentnames{ii},CustomStats.(cmsmomentnames{ii}))
        end
    end
    if heteroagentoptions.useCustomModelStats==1
        fprintf('Current CustomModelStats variables (from heteroagentoptions): \n')
        for ii=1:length(customstatnames)
            fprintf('	%s: %8.4f \n',customstatnames{ii},CustomStats2.(customstatnames{ii}))
        end
    end
    fprintf('Current and target moments (first row is current, second row is target) \n')
    [currentmomentvec(actualtarget)'; targetmomentvec(actualtarget)'] % these are columns, so transpose into rows
    if caliboptions.vectoroutput==0
        fprintf('Current objective fn value is %8.12f \n', Obj)
        fprintf('Current objective fn value breakdown: Obj1=%8.6f and Obj2=%8.6f (with caliboptions.relativeGEweight=%8.6f) \n', Obj1, Obj2, caliboptions.relativeGEweight)
        if penalty>0
            if Obj1>0
                fprintf('Current penalty is to multiply the targets part of the objective fn by %8.2f \n', 1.2*penalty)
            else  % Obj1 is negative, so penalty is to reduce magnitude
                fprintf('Current penalty is to multiply the targets part of the objective fn by %8.2f \n', 0.8*(1/penalty) )
            end
        end
    elseif caliboptions.vectoroutput==2
        % Obj1 and Obj2 are vectors of residuals here, so print their sums of squares (not the vectors themselves)
        fprintf('Current (sum-of-squares of) objective fn value is %8.12f \n', Obj'*Obj)
        fprintf('Current objective fn value breakdown: targets part=%8.6f and GE part=%8.6f (with caliboptions.relativeGEweight=%8.6f); their sum is the sum-of-squares above (less any penalty) \n', gather(sum(Obj1(:).^2))/caliboptions.nCalibParamsNonGE, gather(caliboptions.relativeGEweight*sum(Obj2(:).^2)), caliboptions.relativeGEweight)
        if penalty>0
            fprintf('Current penalty residual is %8.6f (it adds %8.6f to the sum-of-squares) \n', sqrt(penalty), penalty)
        end
    end

    fprintf(' \n')
    fprintf('Current GE prices (this is just repeat of part of current calibration params): \n')
    for ii=1:length(GEPriceParamNames)
        fprintf('	%s: %8.4f \n',GEPriceParamNames{ii},Parameters.(GEPriceParamNames{ii}))
    end
    fprintf('Current aggregate variables: \n')
    for ii=1:length(AggVarNames)
        fprintf('	%s: %8.4f \n',AggVarNames{ii},Parameters.(AggVarNames{ii})) % Note, this is done differently here because AggVars itself has been set as a matrix
    end
    if heteroagentoptions.useintermediateEqns==1
        fprintf('Current intermediateEqn variables: \n')
        for aa=1:length(heteroagentoptions.intermediateEqnNames)
            fprintf('	%s: %8.4f \n',heteroagentoptions.intermediateEqnNames{aa},Parameters.(heteroagentoptions.intermediateEqnNames{aa})) % Note, this is done differently here because AggVars itself has been set as a matrix
        end
    end
    fprintf('Current GeneralEqmEqns: \n')
    for ii=1:length(GEeqnNames)
        fprintf('	%s: %8.4f \n',GEeqnNames{ii},GeneralEqmConditionsVec(ii))
    end
end


end
