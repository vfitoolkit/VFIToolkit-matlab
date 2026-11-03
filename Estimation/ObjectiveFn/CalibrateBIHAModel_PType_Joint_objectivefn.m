function [Obj,GEout]=CalibrateBIHAModel_PType_Joint_objectivefn(calibparamsvec, CalibParamNames,n_d,n_a,n_z,Names_i,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, GEPriceParamNames, PTypeDistParamNames, ParametrizeParamsFn, FnsToEvaluate, GeneralEqmEqnsCell, GEeqnNames, GeneralEqmEqnParamNames, usingallstats,usingautocorr,usingcrosssec,usingcustomstats, targetmomentvec, allstatmomentnames,autocorrmomentnames,crosssecmomentnames,cmsmomentnames, allstatcummomentsizes,autocorrcummomentsizes,crossseccummomentsizes,cmscummomentsizes, AllStats_whichstats,AutoCorrStats_whichstats,CrossSecStats_whichstats, FnsToEvaluate_AllStats, FnsToEvaluate_AutoCorrStats, FnsToEvaluate_CrossSecStats, nCalibParams, nCalibParamsFinder, calibparamsvecindex, calibparamssizes, calibomitparams_counter, calibomitparamsmatrix, caliboptions, heteroagentoptions, vfoptions,simoptions)
% Note: Inputs are CalibParamNames,TargetMoments, and then everything
% needed to be able to run ValueFnIter, StationaryDist, AllStats,
% AutoCorrTransProbs and CrossSectionCovarCorr (the PType versions). Lastly there is caliboptions, which also carries
% caliboptions.whichcombos (1 or 0), caliboptions.selectors and caliboptions.autocorrtimehorizons from SetupTargetMoments_InfHorz.
%
% Joint: the GE prices are the last entries of CalibParamNames (calibrated alongside the other parameters), and the GE conditions
% are part of the objective.
% Second output GEout: .prices (the GE prices, in the order of GEPriceParamNames) and .conditions (the GE conditions, in the order of
% GEeqnNames), both column vectors.
% vectoroutput=1 outputs the model moments at the targeted entries followed by the GE conditions.

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
        currparamraw=calibomitparamsmatrix{sum(calibomitparams_counter(1:pp))}; % (a cell: one omitted-values vector per masked parameter, of any length)
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
    % The user's own grids are needed if either CustomModelStats is given them
    KeepOriginalGrid=((usingcustomstats==1 && caliboptions.CustomModelStats_usergrids==1) || (heteroagentoptions.useCustomModelStats==1 && heteroagentoptions.CustomModelStats_usergrids==1));
    [z_gridvals, pi_z, vfoptions]=ExogShockSetup_InfHorz_PType(n_z,z_gridvals,pi_z,Names_i,Parameters,vfoptions,3,KeepOriginalGrid);
    % output: z_gridvals (struct keyed by Names_i), pi_z (struct), vfoptions.e_gridvals (struct), vfoptions.pi_e (struct)
    simoptions.e_gridvals=vfoptions.e_gridvals;
    simoptions.pi_e=vfoptions.pi_e;
end

%% Solve the model
% (V is kept: either CustomModelStats, from caliboptions or from heteroagentoptions, takes it)
[V, Policy]=ValueFnIter_InfHorz_PType(n_d,n_a,n_z,Names_i,d_grid, a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions);

StationaryDist=StationaryDist_InfHorz_PType(PTypeDistParamNames,Policy,n_d,n_a,n_z,Names_i,pi_z,Parameters,simoptions);

% Note: I could make it a bit faster by just doing these AggVars for the general eqm as part of the AllStats
AggVars=EvalFnOnAgentDist_AggVars_InfHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,Names_i,d_grid, a_grid, z_gridvals, simoptions);

AggVarNames=fieldnames(AggVars);
for aa=1:length(AggVarNames)
    Parameters.(AggVarNames{aa})=AggVars.(AggVarNames{aa}).Mean;
end
% General eqm conditions (or intermediateEqns) by ptype (heteroagentoptions.GEptype, intermediateEqnsptype, set up by the calling command;
% all zeros if not set): they read the AggVars and the parameters that depend on ptype as their '_name' versions, e.g. N_typeA, Tr_typeA
% (as in HeteroAgentStationaryEqm_InfHorz_PType; here the '_name' parameters are made at every evaluation, as calibrated parameters change)
if ~isfield(heteroagentoptions,'GEptype')
    heteroagentoptions.GEptype=zeros(1,length(GEeqnNames));
end
if heteroagentoptions.useintermediateEqns==1 && ~isfield(heteroagentoptions,'intermediateEqnsptype')
    heteroagentoptions.intermediateEqnsptype=zeros(1,heteroagentoptions.nIntermediateEqns);
end
useptypeconditions=any(heteroagentoptions.GEptype==1);
if heteroagentoptions.useintermediateEqns==1
    useptypeconditions=(useptypeconditions || any(heteroagentoptions.intermediateEqnsptype==1));
end
if useptypeconditions
    N_i=length(Names_i);
    for aa=1:length(AggVarNames)
        for ii=1:N_i
            if isfield(AggVars.(AggVarNames{aa}),Names_i{ii})
                Parameters.([AggVarNames{aa},'_',Names_i{ii}])=AggVars.(AggVarNames{aa}).(Names_i{ii}).Mean;
            else
                Parameters.([AggVarNames{aa},'_',Names_i{ii}])=0; % (as HeteroAgentStationaryEqm does, an AggVar a ptype does not have is zero)
            end
        end
    end
    for pp=1:length(heteroagentoptions.paramnamesptype)
        temp=Parameters.(heteroagentoptions.paramnamesptype{pp});
        for ii=1:N_i
            if isstruct(temp)
                if isfield(temp,Names_i{ii})
                    Parameters.([heteroagentoptions.paramnamesptype{pp},'_',Names_i{ii}])=temp.(Names_i{ii});
                end
            elseif size(temp,1)==N_i
                Parameters.([heteroagentoptions.paramnamesptype{pp},'_',Names_i{ii}])=temp(ii,:);
            else
                Parameters.([heteroagentoptions.paramnamesptype{pp},'_',Names_i{ii}])=temp(:,ii);
            end
        end
    end
end

%% Custom Model Stats
if usingcustomstats==1
    if caliboptions.CustomModelStats_usergrids==0
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    elseif caliboptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            caliboptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            caliboptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats=caliboptions.CustomModelStats(V,Policy,StationaryDist,Parameters,FnsToEvaluate,n_d,n_a,n_z,Names_i,d_grid,a_grid,caliboptions.CustomModelStatsInputs.z_grid,caliboptions.CustomModelStatsInputs.pi_z,caliboptions,caliboptions.CustomModelStatsInputs.vfoptions,caliboptions.CustomModelStatsInputs.simoptions);
    end
end
% Custom Model Stats can be in heteroagent options or caliboptions so when doing joint-optimization we need to handle both possibilities
% (heteroagentoptions.CustomModelStats takes the same inputs HeteroAgentStationaryEqm_InfHorz_PType hands it)
if heteroagentoptions.useCustomModelStats==1
    if heteroagentoptions.CustomModelStats_usergrids==0
        CustomStats2=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,z_gridvals,pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    elseif heteroagentoptions.CustomModelStats_usergrids==1
        if caliboptions.calibrateshocks==1 % shock grids depend on parameters being calibrated, so give the user's own grids as rebuilt from the current parameters
            heteroagentoptions.CustomModelStatsInputs.z_grid=vfoptions.user_z_grid;
            heteroagentoptions.CustomModelStatsInputs.pi_z=vfoptions.user_pi_z;
        end
        CustomStats2=heteroagentoptions.CustomModelStats(V,Policy,StationaryDist,Parameters,heteroagentoptions.CustomModelStatsInputs.FnsToEvaluate,heteroagentoptions.CustomModelStatsInputs.n_d,heteroagentoptions.CustomModelStatsInputs.n_a,heteroagentoptions.CustomModelStatsInputs.n_z,Names_i,heteroagentoptions.CustomModelStatsInputs.d_grid,heteroagentoptions.CustomModelStatsInputs.a_grid,heteroagentoptions.CustomModelStatsInputs.z_grid,heteroagentoptions.CustomModelStatsInputs.pi_z,heteroagentoptions,heteroagentoptions.CustomModelStatsInputs.vfoptions,heteroagentoptions.CustomModelStatsInputs.simoptions);
    end
    % Note: anything else you want, just 'hide' it in heteroagentoptions
    customstatnames=fieldnames(CustomStats2);
    for pp=1:length(customstatnames)
        Parameters.(customstatnames{pp})=CustomStats2.(customstatnames{pp});
    end
end

%% intermediateEqns
if heteroagentoptions.useintermediateEqns==1
    gg_c=0;
    for aa=1:heteroagentoptions.nIntermediateEqns
        if heteroagentoptions.intermediateEqnsptype(aa)==0
            gg_c=gg_c+1;
            Parameters.(heteroagentoptions.intermediateEqnNames{aa})=GeneralEqmConditions_Case1_v3(heteroagentoptions.intermediateEqnsCell{aa},heteroagentoptions.intermediateEqnParamNames(gg_c).Names,Parameters);
            if useptypeconditions % an intermediateEqn named x_typeA is the typeA value of x
                intEqnnames_aa=heteroagentoptions.intermediateEqnNames{aa};
                for ii=1:N_i
                    lname=length(Names_i{ii});
                    if length(intEqnnames_aa)>lname+1
                        if strcmp(intEqnnames_aa(end-lname:end),['_',Names_i{ii}])
                            Parameters.(intEqnnames_aa(1:end-lname-1)).(Names_i{ii})=Parameters.(intEqnnames_aa);
                        end
                    end
                end
            end
        else % this intermediateEqn is conditional on ptype
            for ii=1:N_i
                gg_c=gg_c+1;
                temp=GeneralEqmConditions_Case1_v3g_ptype(heteroagentoptions.intermediateEqnsCell{aa},heteroagentoptions.intermediateEqnParamNames(gg_c).Names,Parameters);
                Parameters.(heteroagentoptions.intermediateEqnNames{aa}).(Names_i{ii})=temp;
                Parameters.([heteroagentoptions.intermediateEqnNames{aa},'_',Names_i{ii}])=temp;
            end
        end
    end
end

%% Evaluate General Eqm Eqns (a condition by ptype gives one value per ptype, in the order of Names_i)
GeneralEqmConditionsVec=zeros(1,length(GeneralEqmEqnParamNames));
gg_c=0;
for gg=1:length(GEeqnNames)
    if heteroagentoptions.GEptype(gg)==0
        gg_c=gg_c+1;
        GeneralEqmConditionsVec(gg_c)=GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg_c).Names, Parameters);
    else
        for ii=1:N_i
            gg_c=gg_c+1;
            GeneralEqmConditionsVec(gg_c)=GeneralEqmConditions_Case1_v3g_ptype(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg_c).Names, Parameters);
        end
    end
end
GEout.prices=[]; % stacked: a GE price that depends on ptype is N_i entries, in the order of Names_i
for pp=1:length(GEPriceParamNames)
    temp=Parameters.(GEPriceParamNames{pp});
    if isstruct(temp)
        for ii=1:length(Names_i)
            GEout.prices=[GEout.prices; gather(temp.(Names_i{ii}))];
        end
    else
        GEout.prices=[GEout.prices; gather(temp(:))];
    end
end
GEout.conditions=gather(GeneralEqmConditionsVec');

%% Calculate model stats
% caliboptions.whichcombos=1 (CalibrateBIHAModel_PType's default): the stats commands compute only the targeted (function, statistic,
% restriction, ptype or grouped) combinations, through simoptions.whichcombos (and, for AllStats, a per-combination simoptions.whichstats)
% with a trailing type dimension, built by SetupTargetMoments_InfHorz (caliboptions.selectors). =0: every statistic of every targeted
% function. The three commands take differently shaped selectors, so each gets its own copy of simoptions; the conditional restrictions
% stay in simoptions for all three.
if usingallstats==1
    simoptions_AllStats=simoptions;
    if caliboptions.whichcombos==1
        simoptions_AllStats.whichcombos=caliboptions.selectors.AllStats.whichcombos;
        simoptions_AllStats.whichstats=caliboptions.selectors.AllStats.whichstats;
    else % caliboptions.whichcombos=0: every statistic of every targeted function
        simoptions_AllStats.whichstats=ones(1,7);
    end
    AllStats=EvalFnOnAgentDist_AllStats_InfHorz_PType(StationaryDist,Policy, FnsToEvaluate_AllStats,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,simoptions_AllStats);
end
if usingautocorr==1
    simoptions_AutoCorr=simoptions;
    if isfield(simoptions,'timehorizons')
        simoptions_AutoCorr.timehorizons=union(simoptions.timehorizons,caliboptions.autocorrtimehorizons); % the horizons the targets name (their _kK suffixes), plus any the user asked for
    else
        simoptions_AutoCorr.timehorizons=caliboptions.autocorrtimehorizons;
    end
    if caliboptions.whichcombos==1
        simoptions_AutoCorr.whichcombos=caliboptions.selectors.AutoCorr.whichcombos; % [nFns,1+nRestr,N_i+1] (no per-combination whichstats; the command reports its byproducts)
    end
    AutoCorrTransProbs=EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz_PType(StationaryDist,Policy,FnsToEvaluate_AutoCorrStats,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,pi_z,simoptions_AutoCorr);
end
if usingcrosssec==1
    simoptions_CrossSec=simoptions;
    if caliboptions.whichcombos==1
        simoptions_CrossSec.whichcombos=caliboptions.selectors.CrossSec.whichcombos; % pair-shaped, [nFns,nFns,1+nRestr,N_i+1]
    end
    CrossSectionCovarCorr=EvalFnOnAgentDist_CrossSectionCovarCorr_InfHorz_PType(StationaryDist,Policy,FnsToEvaluate_CrossSecStats,Parameters,n_d,n_a,n_z,Names_i,d_grid,a_grid,z_gridvals,simoptions_CrossSec);
end


%% Get current values of the target moments as a vector
% Each kind: walk the names of the target (two to five levels, the empty cells skipped) into the command's output, which has the same
% nesting (restrictions, permanent types, MoreInequality, CovarianceWith/CorrelationWith and the matrices alike). An AutoCorr statistic
% with a _kK horizon suffix is read from the command's .tperiodsK.(statistic), grouped or inside the type's struct. A target that is
% NaN in full (SetupTargetMoments_InfHorz warned about it and selected nothing for it) is skipped and filled with NaN.
currentmomentvec=zeros(size(targetmomentvec));
sofar=0;
for kindc=1:3
    if kindc==1
        kindusing=usingallstats;
    elseif kindc==2
        kindusing=usingautocorr;
    else
        kindusing=usingcrosssec;
    end
    if kindusing==1
        if kindc==1
            kindnames=allstatmomentnames; kindcum=allstatcummomentsizes; kindout=AllStats;
        elseif kindc==2
            kindnames=autocorrmomentnames; kindcum=autocorrcummomentsizes; kindout=AutoCorrTransProbs;
        else
            kindnames=crosssecmomentnames; kindcum=crossseccummomentsizes; kindout=CrossSectionCovarCorr;
        end
        for cc=1:size(kindnames,1)
            if cc==1
                idx=sofar+1:sofar+kindcum(1);
            else
                idx=sofar+kindcum(cc-1)+1:sofar+kindcum(cc);
            end
            if all(isnan(targetmomentvec(idx)))
                currentmomentvec(idx)=NaN;
                continue
            end
            names=kindnames(cc,:);
            names=names(~cellfun(@isempty,names));
            if kindc==2
                tk=regexp(names{end},'_k(\d+)$','tokens','once'); % a horizon suffix: stat_kK is the command's .tperiodsK.(stat)
                if ~isempty(tk)
                    names=[names(1:end-1),{['tperiods',tk{1}]},{names{end}(1:end-length(tk{1})-2)}];
                end
            end
            temp=kindout;
            for kk=1:numel(names)
                temp=temp.(names{kk});
            end
            currentmomentvec(idx)=temp(:);
        end
        sofar=sofar+kindcum(end);
    end
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
    Obj1=Obj1/caliboptions.nCalibParamsNonGE; % This is done so that the tolerances for convergence are sensible (the number of calibrated parameter names, not counting the GE prices, so the same as in the nested objective) [was length(CalibParamNames), which in joint includes the GE prices]

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
        if isstruct(Parameters.(GEPriceParamNames{ii})) % a GE price that depends on ptype
            for jj=1:length(Names_i)
                fprintf('	%s.%s: %8.4f \n',GEPriceParamNames{ii},Names_i{jj},Parameters.(GEPriceParamNames{ii}).(Names_i{jj}))
            end
        else
            fprintf('	%s: %8.4f \n',GEPriceParamNames{ii},Parameters.(GEPriceParamNames{ii}))
        end
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
    gg_c=0;
    for ii=1:length(GEeqnNames)
        if heteroagentoptions.GEptype(ii)==0
            gg_c=gg_c+1;
            fprintf('	%s: %8.4f \n',GEeqnNames{ii},GeneralEqmConditionsVec(gg_c))
        else % one per ptype
            for jj=1:length(Names_i)
                gg_c=gg_c+1;
                fprintf('	%s (%s): %8.4f \n',GEeqnNames{ii},Names_i{jj},GeneralEqmConditionsVec(gg_c))
            end
        end
    end
end


end
