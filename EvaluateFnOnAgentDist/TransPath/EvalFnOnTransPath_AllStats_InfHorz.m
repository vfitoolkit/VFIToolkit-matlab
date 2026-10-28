function AllStatsPath=EvalFnOnTransPath_AllStats_InfHorz(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,transpathoptions,simoptions)
% AllStatsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).
% No z is n_z=0 (z_grid=[]). iid e shocks are given in simoptions (n_e, e_grid, pi_e), as for EvalFnOnAgentDist_AllStats_InfHorz;
% e_grid may vary over the path. AgentDistPath is [n_a,(n_z),(n_e),T] and PolicyPath [rows,n_a,(n_z),(n_e),T].
% Semi-exogenous states are given in simoptions (n_semiz, semiz_grid, SemiExoStateFn or pi_semiz, l_dsemiz), as for
% EvalFnOnAgentDist_AllStats_InfHorz; then AgentDistPath is [n_a,n_semiz,(n_z),(n_e),T], PolicyPath [rows,n_a,n_semiz,(n_z),(n_e),T]
% and FnsToEvaluate inputs are (d,aprime,a,semiz,z,e,...).

if ~exist('simoptions','var')
    % If simoptions is not given, just use all the defaults
    simoptions.npoints=100;
    simoptions.nquantiles=20;
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    % Model solution
    simoptions.gridinterplayer=0;
    % Model setup
    simoptions.experienceasset=0;
    simoptions.experienceassetz=0;
    simoptions.experienceassete=0;
    simoptions.experienceassetze=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.n_semiz=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    % Model setup
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20;
    end
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    end
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    % Model solution
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
    if ~isfield(simoptions,'experienceassetz')
        simoptions.experienceassetz=0;
    end
    if ~isfield(simoptions,'experienceassete')
        simoptions.experienceassete=0;
    end
    if ~isfield(simoptions,'experienceassetze')
        simoptions.experienceassetze=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
end

l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
% Read here, because ExogShockSetup_InfHorz_TPath below removes n_e from simoptions
n_e=simoptions.n_e;
N_e=prod(n_e);
if N_e==0
    l_e=0;
else
    l_e=length(n_e);
end
if N_e>0 && simoptions.experienceasset>=1
    error('EvalFnOnTransPath_AllStats_InfHorz: experienceasset together with an e variable is not yet implemented')
end
% (z,e) are kept as one exogenous dimension, z first, which is a singleton when there is neither
N_ze=max(N_z,1)*max(N_e,1);
if N_z==0 && N_e==0
    n_ze=0;
elseif N_z>0 && N_e==0
    n_ze=n_z;
elseif N_z==0 && N_e>0
    n_ze=n_e;
elseif N_z>0 && N_e>0
    n_ze=[n_z,n_e];
end
% Semi-exogenous states (simoptions.n_semiz, with semiz_grid, and SemiExoStateFn or pi_semiz, l_dsemiz) go in front of
% (z,e), semiz fastest: from here on the exogenous dimension (n_ze, N_ze) is (semiz,z,e). FnsToEvaluate inputs are
% (d,aprime,a,semiz,z,e,...).
N_semiz=prod(simoptions.n_semiz);
N_zeonly=N_ze; % (z,e) alone, without semiz
if N_semiz==0
    l_semiz=0;
else
    l_semiz=length(simoptions.n_semiz);
    if n_d(1)==0
        error('EvalFnOnTransPath_AllStats_InfHorz: semi-exogenous states (simoptions.n_semiz) require a decision variable: the last simoptions.l_dsemiz decision variables determine the semiz transitions')
    end
    if simoptions.experienceasset>=1
        error('EvalFnOnTransPath_AllStats_InfHorz: semi-exogenous states (simoptions.n_semiz) are not yet implemented together with an experience asset')
    end
    N_ze=N_semiz*N_ze;
    if N_z==0 && N_e==0
        n_ze=simoptions.n_semiz;
    else
        n_ze=[simoptions.n_semiz,n_ze];
    end
end

%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
FnsToEvaluateNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnsToEvaluateNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvaluateNames{ff}));
    if length(temp)>(l_d+l_a+l_a+l_semiz+l_z+l_e)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_semiz+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,semiz,z,e)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(FnsToEvaluateNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=FnsToEvaluateNames;

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,ParamPathNames,{},transpathoptions);

%% The _tminus1 and _tplus1 values come from transpathoptions.initialvalues
% Checked here rather than at first use, so that a missing initialvalues names itself
% instead of surfacing as 'Unrecognized field name' from inside the loop over t.
if (use_tminus1price==1 || use_tminus1params==1 || use_tminus1AggVars==1) && ~isfield(transpathoptions,'initialvalues')
    error('EvalFnOnTransPath_AllStats_InfHorz: a _tminus1 input is used, so transpathoptions.initialvalues must be given (it supplies the period-1 values)')
end

%%
if all(size(d_grid)==[prod(n_d),length(n_d)])
    d_gridvals=d_grid; % already joint-grid
else
    d_gridvals=CreateGridvals(n_d,d_grid,1); % convert stacked-column to joint-grid
end
a_gridvals=CreateGridvals(n_a,a_grid,1);
if simoptions.experienceasset==0
    if simoptions.gridinterplayer==1
        if isscalar(n_a)
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a)',a_grid(1:n_a(1)),linspace(1,N_a,N_aprime)');
            aprime_grid=temp;
            n_aprime=N_aprime;
        else
            N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
            aprime_grid=[temp; a_grid(n_a(1)+1:end)];
            n_aprime=[N_a1prime,n_a(2:end)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
        simoptions.policyind2val_finegridinput=1; % aprime_gridvals contains the fine grid for the first asset (tells PolicyInd2Val_InfHorz_TPath)
    else
        aprime_gridvals=a_gridvals;
    end
elseif simoptions.experienceasset>=1
    % omit a2 from aprime_gridvals
    if simoptions.gridinterplayer==1
        N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
        temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
        if length(n_a)==2
            aprime_grid=temp; % omit a2
            n_aprime=N_a1prime; % omit a2
        elseif length(n_a)>2 % more than one a1
            aprime_grid=[temp; a_grid(n_a(1)+1:end-1)];
            n_aprime=[N_a1prime,n_a(2:end-1)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
        simoptions.policyind2val_finegridinput=1; % aprime_gridvals contains the fine grid for the first asset (tells PolicyInd2Val_InfHorz_TPath)
    else
        aprime_gridvals=CreateGridvals(n_a(1:end-1),a_grid(1:sum(n_a(1:end-1))),1); % omit a2
    end

end
% The TPath variant of the setup, so that a z_grid which varies along the path is picked up
% (the setup writes zpathtrivial and z_gridvals_T back into transpathoptions)
% ze_gridvals is the (z,e) grid (z varies first), and transpathoptions.zepathtrivial=0 puts its path in
% transpathoptions.ze_gridvals_T (when either z or e varies over the path).
[~, ~, ~, ~, ~, ~, ze_gridvals, transpathoptions, simoptions]=ExogShockSetup_InfHorz_TPath(n_z,z_grid,[],Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,1);
if N_semiz>0
    % The semiz grid (fixed over the path), put in front of the (z,e) grid, semiz fastest
    [semiz_gridvals, ~, transpathoptions, simoptions]=SemiExogShockSetup_InfHorz_TPath(n_d,d_grid,Parameters,PricePathNames,ParamPath,ParamPathNames,ParamPathSizeVec,T,transpathoptions,simoptions,1);
    if N_z==0 && N_e==0
        ze_gridvals=semiz_gridvals;
    else
        ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(ze_gridvals,N_semiz,1)];
    end
end


%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    for rr=1:length(CondlRestnFnNames)
        AllStatsPath.(CondlRestnFnNames{rr}).RestrictedSampleMass=zeros(1,T); % preallocate
    end
end


%%
% (PolicyInd2Val_InfHorz_TPath takes the exogenous sizes, (semiz,z,e) together, as its n_z input)
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_ze,T,d_grid,aprime_gridvals,simoptions,1); % [l_d+l_aprime,N_a,N_ze,T]
AgentDistPath=reshape(AgentDistPath,[N_a*N_ze,T]);

% preallocate
for ff=1:length(FnsToEvaluateNames)
    AllStatsPath.(FnsToEvaluateNames{ff}).Mean=zeros(1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).Median=zeros(1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).RatioMeanToMedian=zeros(1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).Variance=zeros(1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).StdDeviation=zeros(1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).LorenzCurve=zeros(simoptions.npoints,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).Gini=zeros(1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).QuantileCutoffs=zeros(simoptions.nquantiles+1,T);
    AllStatsPath.(FnsToEvaluateNames{ff}).QuantileMeans=zeros(simoptions.nquantiles,T);
end


%%
for tt=1:T
    % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
    % (setting them after period tt is written in would make every _tminus1 equal to period tt)
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            if tt>1
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            else
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
    end
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=AllStatsPath.(tminus1AggVarsNames{pp}).Mean(tt-1);
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if transpathoptions.zepathtrivial==0
        if N_semiz==0
            ze_gridvals=transpathoptions.ze_gridvals_T(:,:,tt);
        else
            ze_gridvals=[repmat(semiz_gridvals,N_zeonly,1),repelem(transpathoptions.ze_gridvals_T(:,:,tt),N_semiz,1)]; % (semiz,z,e), semiz fastest
        end
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            % Period T is the final stationary eqm, so the price after it is the same as at T
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
    end

    PolicyValues=PolicyValuesPath(:,:,:,tt);
    AgentDist=AgentDistPath(:,tt);

    if N_z==0 && N_e==0 && N_semiz==0
        AllStats=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AllStats_noz(AgentDist, PolicyValues, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,FnsToEvaluateNames, n_a, a_gridvals,simoptions);
    else
        AllStats=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AllStats(AgentDist, PolicyValues, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,FnsToEvaluateNames, n_a, n_ze, a_gridvals, ze_gridvals,simoptions);
    end

    if useCondlRest==1
        for rr=1:length(CondlRestnFnNames)
            AllStatsPath.(CondlRestnFnNames{rr}).RestrictedSampleMass(tt)=AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass;
        end
    end

    for ff=1:length(FnsToEvaluateNames)
        AllStatsPath.(FnsToEvaluateNames{ff}).Mean(tt)=AllStats.(FnsToEvaluateNames{ff}).Mean;
        AllStatsPath.(FnsToEvaluateNames{ff}).Median(tt)=AllStats.(FnsToEvaluateNames{ff}).Median;
        AllStatsPath.(FnsToEvaluateNames{ff}).RatioMeanToMedian(tt)=AllStats.(FnsToEvaluateNames{ff}).RatioMeanToMedian;
        AllStatsPath.(FnsToEvaluateNames{ff}).Variance(tt)=AllStats.(FnsToEvaluateNames{ff}).Variance;
        AllStatsPath.(FnsToEvaluateNames{ff}).StdDeviation(tt)=AllStats.(FnsToEvaluateNames{ff}).StdDeviation;
        AllStatsPath.(FnsToEvaluateNames{ff}).LorenzCurve(:,tt)=AllStats.(FnsToEvaluateNames{ff}).LorenzCurve;
        AllStatsPath.(FnsToEvaluateNames{ff}).Gini(tt)=AllStats.(FnsToEvaluateNames{ff}).Gini;
        AllStatsPath.(FnsToEvaluateNames{ff}).QuantileCutoffs(:,tt)=AllStats.(FnsToEvaluateNames{ff}).QuantileCutoffs;
        AllStatsPath.(FnsToEvaluateNames{ff}).QuantileMeans(:,tt)=AllStats.(FnsToEvaluateNames{ff}).QuantileMeans;

        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                if AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass>0
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Mean(tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Mean;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Median(tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Median;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).RatioMeanToMedian(tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).RatioMeanToMedian;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Variance(tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Variance;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).StdDeviation(tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).StdDeviation;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).LorenzCurve(:,tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).LorenzCurve;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Gini(tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).Gini;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).QuantileCutoffs(:,tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).QuantileCutoffs;
                    AllStatsPath.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).QuantileMeans(:,tt)=AllStats.(CondlRestnFnNames{rr}).(FnsToEvaluateNames{ff}).QuantileMeans;
                end
            end
        end
    end
end

if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        if any(AllStatsPath.(CondlRestnFnNames{rr}).RestrictedSampleMass==0) && simoptions.warnzerorestrictedmass==2
            warning('One of the conditional restrictions evaluates to a zero mass in some time periods of the transition path')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass in some time periods \n'])
        end
    end
end

end
