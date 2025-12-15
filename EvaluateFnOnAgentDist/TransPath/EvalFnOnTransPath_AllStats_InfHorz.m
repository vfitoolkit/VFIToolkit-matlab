function AllStatsPath=EvalFnOnTransPath_AllStats_InfHorz(FnsToEvaluate,AgentDistPath,PolicyPath,PricePath,ParamPath, Parameters, T, n_d, n_a, n_z, d_grid, a_grid,z_grid,simoptions)
% AllStatsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

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
    simoptions.n_e=0;
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
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
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
l_z=length(n_z);

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%%
% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_StructToMatrix(PricePath,ParamPath,T);

%%
FnsToEvaluateNames=fieldnames(FnsToEvaluate);
for ff=1:length(FnsToEvaluateNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvaluateNames{ff}));
    if length(temp)>(l_d+l_a+l_a+l_z)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(FnsToEvaluateNames{ff});
end
% For the subfunctions we want the following
simoptions.outputasstructure=0;
simoptions.AggVarNames=FnsToEvaluateNames;

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate)
    [tplus1priceNames,tminus1priceNames,~,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames);
    % tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk.
    % But omit tminus1AggVarsNames as AggVars are anyway not allowed to take AggVars as inputs.
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(simoptions.initialvalues,tminus1priceNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

%%
d_gridvals=CreateGridvals(n_d,d_grid,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if simoptions.experienceasset==0
    if simoptions.gridinterplayer==1
        if isscalar(n_a)
            N_aprime=N_a+(N_a-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,N_a,N_a)',a_grid(1:n_a(1)),linspace(1,N_a,N_aprime)');
            aprime_grid=temp;
            n_aprime=n_a;
        else
            N_a1prime=n_a(1)+(n_a(1)-1)*simoptions.ngridinterp;
            temp=interp1(linspace(1,n_a(1),n_a(1))',a_grid(1:n_a(1)),linspace(1,n_a(1),N_a1prime)');
            aprime_grid=[temp; a_grid(n_a(1)+1:end)];
            n_aprime=[N_a1prime,n_a(2:end)];
        end
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    else
        aprime_gridvals=a_gridvals;
    end
elseif simoptions.experienceasset==1
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
    else
        aprime_gridvals=CreateGridvals(n_a(1:end-1),a_grid(1:sum(n_a(1:end-1))),1); % omit a2
    end

end
z_gridvals=CreateGridvals(n_z,z_grid,1);


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
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyPath,n_d,n_a,n_z,T,d_gridvals,aprime_gridvals,simoptions,1);
AgentDistPath=reshape(AgentDistPath,[N_a,N_z,T]);

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
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=simoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
        end
    end
    
    PolicyValues=PolicyValuesPath(:,:,:,tt);
    AgentDist=AgentDistPath(:,:,tt);
    
    AllStats=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AllStats(AgentDist(:), PolicyValues, FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames,FnsToEvaluateNames, n_a, n_z, a_gridvals, z_gridvals,simoptions);
    
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
                AllStatsPath.(CondlRestnFnNames{rr}).RestrictedSampleMass(tt)=AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass;

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
        if any(AllStatsPath.(CondlRestnFnNames{rr}).RestrictedSampleMass==0)
            warning('One of the conditional restrictions evaluates to a zero mass in some time periods of the transition path')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass in some time periods \n'])
        end
    end
end

end
