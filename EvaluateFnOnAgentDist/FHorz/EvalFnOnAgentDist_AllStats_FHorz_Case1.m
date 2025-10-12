function AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)

if ~exist('simoptions','var')
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.gridinterplayer=0;
else
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    end
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
end


%%
l_a=length(n_a);
N_a=prod(n_a);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

%% I want to do some things now, so that they can be used in setting up conditional restrictions
AllStats=struct();

a_gridvals=CreateGridvals(n_a,a_grid,1);
if N_z==0
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)
else
    StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
end

% Figure out l_daprime from PolicyValues
l_daprime=size(PolicyValues,1);


%% Implement new way of handling FnsToEvaluate

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'outputasstructure')
    if simoptions.outputasstructure==1
        FnsToEvaluateStruct=1;
        FnsToEvalNames=simoptions.FnsToEvalNames;
    elseif simoptions.outputasstructure==0
        FnsToEvaluateStruct=0;
    end
end


%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it). 
if isfield(simoptions,'conditionalrestrictions')    
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),1);
    RestrictionStruct=struct();

    % For each conditional restriction, create a 'restricted stationary distribution'
    for rr=1:length(CondlRestnFnNames)
        % The current conditional restriction function
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a+l_z)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames={};
        end

        if N_z==0
            CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,CondlRestnFnParamNames,N_j,2); % j in 2nd dimension: (a,j,l_d+l_a), so we want j to be after N_a
            RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]));
        else
            CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,CondlRestnFnParamNames,N_j,3); % j in 3rd dimension: (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
            RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J));
        end

        RestrictedStationaryDistVec=StationaryDist;
        RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction
        
        % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
        restrictedsamplemass(rr)=sum(RestrictedStationaryDistVec(:));
        RestrictionStruct(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass(rr);

        if restrictedsamplemass(rr)==0
            warning('One of the conditional restrictions evaluates to a zero mass')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Just return this and hopefully it is clear to the user
        else
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Seems likely this would be something user might want
        end

    end
end


%%
if N_z==0
    % StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    % PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,N_j,2); % j in 2nd dimension: (a,j,l_d+l_a), so we want j to be after N_a
        Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
        AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

        %% If there are any conditional restrictions then deal with these
        % Evaluate AllStats, but conditional on the restriction being one.
        if useCondlRest==1
            % Evaluate the conditinal restrictions:
            % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
            for rr=1:length(CondlRestnFnNames)
                if restrictedsamplemass(rr)>0
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                end
            end
        end
    end
else % N_z
    % StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    % PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        % Values=nan(N_a,N_z,N_j,'gpuArray');
        CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,N_j,3); % j in 3rd dimension: (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
        Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
        AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

        %% If there are any conditional restrictions then deal with these
        % Evaluate AllStats, but conditional on the restriction being one.
        if useCondlRest==1
            % Evaluate the conditinal restrictions:
            % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
            for rr=1:length(CondlRestnFnNames)
                if restrictedsamplemass(rr)>0
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                end
            end
        end
    end
end


end
