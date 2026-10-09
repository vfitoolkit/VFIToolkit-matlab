function AllStats=EvalFnOnAgentDist_AllStats_InfHorz(StationaryDist, Policy, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions)
% Returns a wide variety of statistics
%
% simoptions optional inputs
%
% simoptions.whichcombos ([numFnsToEvaluate, 1+number of conditional restrictions]) selects which (fn, restriction) combinations are
% computed, and simoptions.whichstats may be given per combination ([numFnsToEvaluate, 1+number of restrictions, 7]); see below.

%%
if ~exist('simoptions','var')
    simoptions.npoints=100;
    simoptions.nquantiles=20;
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.gridinterplayer=0;
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.n_semiz=0;
else
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
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
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

if gpuDeviceCount==0
    error('AllStats requires a GPU')
end

% if n_d(1)==0
%     l_d=0;
% else
%     l_d=length(n_d);
% end
l_a=length(n_a);

N_a=prod(n_a);

a_gridvals=CreateGridvals(n_a,a_grid,1);
% Switch to z_gridvals (folding e and semiz into z if appropriate)
[n_z,z_gridvals,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_InfHorz(n_z,z_grid,simoptions,Parameters);

AllStats=struct();

%% I want to do some things now, so that they can be used in setting up conditional restrictions
StationaryDistVec=reshape(StationaryDist,[N_a*max(N_z,1),1]);

% Make sure things are on the gpu (they should already be)
StationaryDistVec=gpuArray(StationaryDistVec);
Policy=gpuArray(Policy);
% Switch to PolicyValues, and permute
PolicyValues=PolicyInd2Val_InfHorz(Policy,n_d,n_a,n_z,d_grid,a_grid,simoptions);
if N_z==0
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a]),[2,1]); %[N_a,l_d+l_a]
else
    PolicyValuesPermute=permute(reshape(PolicyValues,[size(PolicyValues,1),N_a,N_z]),[2,3,1]); %[N_a,N_z,l_d+l_a]
end
l_daprime=size(PolicyValues,1);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
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
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a+l_z)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames={};
        end
        CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames);

        RestrictionValues=logical(EvalFnOnAgentDist_Grid(CondlRestnFn, CondlRestnFnParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals));
        RestrictionValues=reshape(RestrictionValues,[N_a*max(N_z,1),1]); % (N_z is zero when there is no z)

        RestrictedStationaryDistVec=StationaryDistVec;
        RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction

        % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
        restrictedsamplemass(rr)=sum(RestrictedStationaryDistVec);
        RestrictionStruct(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass(rr);

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
% (As EvalFnOnAgentDist_AllStats_FHorz_Case1.)
numFnsToEvaluate=length(FnsToEvaluate);
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,nwhichpages); % whichcombos here is [nFns, 1+nRestr]
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
for ff=1:length(FnsToEvalNames)
    if ~any(whichcombos(ff,:)) % no combination of this function is wanted
        continue
    end
    FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names);
    Values=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermute,l_daprime,n_a,n_z,a_gridvals,z_gridvals);
    Values=reshape(Values,[N_a*max(N_z,1),1]);

    if whichcombos(ff,1)==1 % the unrestricted stats of this function are wanted
        AllStats.(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,StationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,reshape(whichstatsG(ff,1,:),[1,7]));
    end

    %% If there are any conditional restrictions then deal with these
    % Evaluate AllStats, but conditional on the restriction being one.
    if useCondlRest==1
        % Evaluate the conditional restrictions:
        % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
        for rr=1:length(CondlRestnFnNames)
            if whichcombos(ff,1+rr)==1 && restrictedsamplemass(rr)>0 % this restriction is wanted for this function and has mass
                AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,reshape(whichstatsG(ff,1+rr,:),[1,7]));
            end
        end
    end

end



end
