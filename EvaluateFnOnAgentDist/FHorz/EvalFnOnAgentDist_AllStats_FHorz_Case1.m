function AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(StationaryDist,Policy, FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% simoptions.whichcombos ([numFnsToEvaluate, 1+number of conditional restrictions]) selects which (fn, restriction) combinations are
% computed, and simoptions.whichstats may be given per combination ([numFnsToEvaluate, 1+number of restrictions, 7]); see below.

if ~exist('simoptions','var')
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    % Model setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
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
    % Model setup
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
end
% Note: FnsToEvaluate commands deliberately do not create n_e and n_semiz in simoptions

if gpuDeviceCount==0
    error('AllStats requires a GPU')
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
    RestrictionMask=cell(length(CondlRestnFnNames),1); % each restriction kept as a logical mask over the grid (1 byte per point); the restricted weights are formed from it where used

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


        % Keep the restricted mass, and the mask
        restrictedsamplemass(rr)=sum(StationaryDist(:).*RestrictionValues(:)); % mass that satisfies the restriction
        RestrictionMask{rr}=RestrictionValues;

        if restrictedsamplemass(rr)==0 && simoptions.warnzerorestrictedmass==2
            warning('One of the conditional restrictions evaluates to a zero mass')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Just return this and hopefully it is clear to the user
        else
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr); % Seems likely this would be something user might want
        end

    end
end

%% simoptions.whichcombos and per-combination simoptions.whichstats
% whichcombos: [numFnsToEvaluate, 1+number of conditional restrictions] of zeros/ones ([numFnsToEvaluate,1] without restrictions):
% page 1 is the unrestricted stats, pages 2:end the restrictions in the fieldnames order of simoptions.conditionalrestrictions. Ones
% are computed, zeros skipped (their output fields are simply absent; RestrictedSampleMass is always filled). Default all ones. A
% vector of length numFnsToEvaluate with restrictions is applied to every page. Intended for calibration/estimation.
% whichstats: the usual 1x7 vector, or [numFnsToEvaluate, 1+number of restrictions, 7] giving a whichstats vector for every
% combination ([numFnsToEvaluate,7] is applied to every page). A combination asking for no statistic is skipped like a whichcombos zero.
numFnsToEvaluate=length(FnsToEvaluate);
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,nwhichpages);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if isvector(whichcombos) && numel(whichcombos)==numFnsToEvaluate
        whichcombos=repmat(whichcombos(:),[1,nwhichpages]); % one entry per function: apply to every page
    end
    if ~isequal(size(whichcombos),[numFnsToEvaluate,nwhichpages])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(nwhichpages),'] (number of FnsToEvaluate, 1+number of conditional restrictions)'])
    end
    whichcombos=double(whichcombos);
end
wsG=simoptions.whichstats;
if isvector(wsG) && numel(wsG)==7
    whichstatsG=repmat(reshape(wsG,[1,1,7]),[numFnsToEvaluate,nwhichpages,1]);
elseif ismatrix(wsG) && isequal(size(wsG),[numFnsToEvaluate,7])
    whichstatsG=repmat(reshape(wsG,[numFnsToEvaluate,1,7]),[1,nwhichpages,1]);
elseif isequal(size(wsG,1:3),[numFnsToEvaluate,nwhichpages,7])
    whichstatsG=wsG;
else
    error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(numFnsToEvaluate),',',num2str(nwhichpages),',7] (number of FnsToEvaluate, 1+number of conditional restrictions, 7), or [',num2str(numFnsToEvaluate),',7]'])
end
whichcombos=whichcombos.*any(whichstatsG,3); % a combination with no statistic requested is skipped altogether


%%
if N_z==0
    % StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    % PolicyValuesPermute=permute(PolicyValues,[2,3,1]); % (N_a,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        if ~any(whichcombos(ff,:)) % no combination of this function is wanted
            continue
        end
        CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,N_j,2); % j in 2nd dimension: (a,j,l_d+l_a), so we want j to be after N_a
        Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,0,a_gridvals,[]);
        if useCondlRest==1
            % The conditional restrictions only change the weights, so sort Values once and reuse the sort for the unrestricted stats and for every restriction
            % (only the points with positive mass are sorted; every restricted distribution is zero outside these)
            positivemass=(StationaryDist(:)>0);
            [SortedValues,SortedValues_index]=sort(Values(positivemass));
            SortedStationaryDist=StationaryDist(positivemass);
            SortedStationaryDist=SortedStationaryDist(SortedValues_index);
        end
        if whichcombos(ff,1)==1 % the unrestricted stats of this function are wanted
            ws1=reshape(whichstatsG(ff,1,:),[1,7]);
            if useCondlRest==0
                AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws1);
            else
                AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(SortedValues,SortedStationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws1);
            end
        end

        %% If there are any conditional restrictions then deal with these
        % Evaluate AllStats, but conditional on the restriction being one.
        if useCondlRest==1
            % Evaluate the conditional restrictions:
            % Only change is to use the restricted distribution (the mask times the agent distribution, normalised) as the agent distribution
            for rr=1:length(CondlRestnFnNames)
                if whichcombos(ff,1+rr)==1 && restrictedsamplemass(rr)>0 % this restriction is wanted for this function and has mass
                    RestrictedStationaryDist=StationaryDist(positivemass).*RestrictionMask{rr}(positivemass)/restrictedsamplemass(rr); % the restricted mass on the points with mass, normalised to one
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(SortedValues,RestrictedStationaryDist(SortedValues_index),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,reshape(whichstatsG(ff,1+rr,:),[1,7]));
                end
            end
        end
    end
else % N_z
    % StationaryDist=reshape(StationaryDist,[N_a,N_z,N_j]);
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    % PolicyValuesPermute=permute(PolicyValues,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)

    for ff=1:length(FnsToEvaluate)
        if ~any(whichcombos(ff,:)) % no combination of this function is wanted
            continue
        end
        % Values=nan(N_a,N_z,N_j,'gpuArray');
        CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,N_j,3); % j in 3rd dimension: (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
        Values=EvalFnOnAgentDist_Grid_J(FnsToEvaluate{ff},CellOverAgeOfParamValues,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals_J);
        if useCondlRest==1
            % The conditional restrictions only change the weights, so sort Values once and reuse the sort for the unrestricted stats and for every restriction
            % (only the points with positive mass are sorted; every restricted distribution is zero outside these)
            positivemass=(StationaryDist(:)>0);
            [SortedValues,SortedValues_index]=sort(Values(positivemass));
            SortedStationaryDist=StationaryDist(positivemass);
            SortedStationaryDist=SortedStationaryDist(SortedValues_index);
        end
        if whichcombos(ff,1)==1 % the unrestricted stats of this function are wanted
            ws1=reshape(whichstatsG(ff,1,:),[1,7]);
            if useCondlRest==0
                AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws1);
            else
                AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(SortedValues,SortedStationaryDist,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws1);
            end
        end

        %% If there are any conditional restrictions then deal with these
        % Evaluate AllStats, but conditional on the restriction being one.
        if useCondlRest==1
            % Evaluate the conditional restrictions:
            % Only change is to use the restricted distribution (the mask times the agent distribution, normalised) as the agent distribution
            for rr=1:length(CondlRestnFnNames)
                if whichcombos(ff,1+rr)==1 && restrictedsamplemass(rr)>0 % this restriction is wanted for this function and has mass
                    RestrictedStationaryDist=StationaryDist(positivemass).*RestrictionMask{rr}(positivemass)/restrictedsamplemass(rr); % the restricted mass on the points with mass, normalised to one
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(SortedValues,RestrictedStationaryDist(SortedValues_index),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,reshape(whichstatsG(ff,1+rr,:),[1,7]));
                end
            end
        end
    end
end


end
