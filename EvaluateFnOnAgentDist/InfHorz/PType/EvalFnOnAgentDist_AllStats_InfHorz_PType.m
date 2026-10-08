function AllStats=EvalFnOnAgentDist_AllStats_InfHorz_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,Names_i,d_grid, a_grid, z_grid, simoptions)
% Reports a variety of stats, both grouped and by PType.
%
% simoptions.whichcombos ([numFnsToEvaluate, 1+number of conditional restrictions], optionally with a trailing type dimension of N_i+1) selects
% which (fn, restriction[, ptype or grouped]) combinations are computed, and simoptions.whichstats may be given per combination; see below.
%
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
%
%
% simoptions.verbose=1 will give feedback
% simoptions.verboseparams=1 will give further feedback on the param values of each permanent type
%
% Rest of this description describes how those inputs not already used for
% ValueFnIter_PType or StationaryDist_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
% The stationary distribution be a structure and will contain both the
% weights/distribution across the permanent types, as well as a pdf for the
% stationary distribution of each specific permanent type.
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.

% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate)
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i=cell(1,N_i);
    for ii=1:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
    simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu (off by default)
    simoptions.groupusingtdigest=0; % if you are ptypestorecpu=1 and groupptypesforstats=1, you might also need to use groupusingtdigest=1 if you get out of memory errors
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu (off by default)
    end
    if ~isfield(simoptions,'groupusingtdigest')
        simoptions.groupusingtdigest=0; % if you are ptypestorecpu=1 and groupptypesforstats=1, you might also need to use groupusingtdigest=1 if you get out of memory errors
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=100;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
    end
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    elseif simoptions.npoints==0
        error('simoptions.npoints must be a positive (non-zero) integer')
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if ~isfield(simoptions,'whichstats')
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    end
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
end

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end

% Set default of grouping all the PTypes together when reporting statistics
% AllStats reports both
% simoptions.groupptypesforstats=0;
% and
% simoptions.groupptypesforstats=1;


% Preallocate a few things
MeanVec=nan(numFnsToEvaluate,N_i); % Note, these need to be nan so we can omitnan to ignore ptypes for who that FnToEvaluate is not relevant
StdDevVec=zeros(numFnsToEvaluate,N_i);
minvaluevec=nan(numFnsToEvaluate,N_i);
maxvaluevec=nan(numFnsToEvaluate,N_i);
AllStats=struct();


% Preallocate
if simoptions.groupusingtdigest==1 % Things are being stored on cpu but solved on gpu
    % Following few lines relate to the digest
    delta=10000;
    merge_nsofar=zeros(1,numFnsToEvaluate); % Keep count
    merge_nsofar2=zeros(1,numFnsToEvaluate); % Keep count

    AllCMerge=struct();
    Alldigestweightsmerge=struct();
    for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
        AllCMerge.(FnsToEvalNames{ff})=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
        Alldigestweightsmerge.(FnsToEvalNames{ff})=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
    end
else
    AllValues=struct();
    AllWeights=struct();
    for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
        AllValues.(FnsToEvalNames{ff})=[];
        AllWeights.(FnsToEvalNames{ff})=[];
    end
end

FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');


%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(N_i,length(CondlRestnFnNames));
    % RestrictionStruct_ii=struct();

    if simoptions.groupusingtdigest==1 % Things are being stored on cpu but solved on gpu
        error('Have not implemented simoptions.groupusingtdigest==1 together with simoptions.conditionalrestrictions')
    else
        AllRestrictedWeights=struct(); % Only used if useCondlRest==1
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            for rr=1:length(CondlRestnFnNames)
                AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=[];
            end
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
% A trailing type dimension may be added: whichcombos [.., N_i+1] and whichstats [.., N_i+1, 7] select per permanent type, in the
% order of Names_i, with the last slot the grouped stats. The shapes without it apply to every ptype and to the grouped stats alike.
% The grouped Mean, StdDeviation, Minimum and Maximum are built from every ptype's, so a grouped slot that is on also computes (and
% reports) that combination for every ptype. (As EvalFnOnAgentDist_AllStats_FHorz_Case1_PType.)
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
if ~isfield(simoptions,'whichcombos')
    whichcombosAll=ones(numFnsToEvaluate,nwhichpages,N_i+1);
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if isvector(whichcombos) && numel(whichcombos)==numFnsToEvaluate
        whichcombos=repmat(whichcombos(:),[1,nwhichpages]); % one entry per function: apply to every page
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,nwhichpages])
        whichcombos=repmat(whichcombos,[1,1,N_i+1]); % no type dimension: apply to every ptype and to the grouped stats
    end
    if ~isequal(size(whichcombos,1:3),[numFnsToEvaluate,nwhichpages,N_i+1])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(nwhichpages),'] (number of FnsToEvaluate, 1+number of conditional restrictions), optionally with a trailing type dimension of ',num2str(N_i+1),' (one slot per permanent type in the order of Names_i, then the grouped stats)'])
    end
    whichcombosAll=double(whichcombos);
end
whichcombosG=whichcombosAll(:,:,N_i+1); % the grouped stats; whichcombosAll(:,:,ii) is ptype ii
wsG=simoptions.whichstats;
whichstatsAll=[]; % set when whichstats carries the trailing type dimension
if isvector(wsG) && numel(wsG)==7
    whichstatsG=repmat(reshape(wsG,[1,1,7]),[numFnsToEvaluate,nwhichpages,1]);
elseif ismatrix(wsG) && isequal(size(wsG),[numFnsToEvaluate,7])
    whichstatsG=repmat(reshape(wsG,[numFnsToEvaluate,1,7]),[1,nwhichpages,1]);
elseif ndims(wsG)==3 && isequal(size(wsG,1:3),[numFnsToEvaluate,nwhichpages,7])
    whichstatsG=wsG;
elseif ndims(wsG)==4 && isequal(size(wsG,1:4),[numFnsToEvaluate,nwhichpages,N_i+1,7])
    whichstatsAll=wsG;
    whichstatsG=reshape(wsG(:,:,N_i+1,:),[numFnsToEvaluate,nwhichpages,7]); % the last slot is the grouped stats
else
    error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(numFnsToEvaluate),',',num2str(nwhichpages),',7] (number of FnsToEvaluate, 1+number of conditional restrictions, 7), or [',num2str(numFnsToEvaluate),',7], or with a trailing type dimension [',num2str(numFnsToEvaluate),',',num2str(nwhichpages),',',num2str(N_i+1),',7] (one slot per permanent type in the order of Names_i, then the grouped stats)'])
end
whichcombosG=whichcombosG.*any(whichstatsG,3); % a grouped combination with no statistic requested is skipped altogether

%% NOTE GROUPING ONLY WORKS IF THE GRIDS ARE THE SAME SIZES FOR EACH AGENT (for whom a given FnsToEvaluate is being calculated)
% (mainly because otherwise would have to deal with simoptions.agegroupings being different for each agent and this requires more complex code)
% Will throw an error if this is not the case

% If grouping, we have ValuesOnDist and StationaryDist that contain
% everything we will need. Now we just have to compute them.
% Note that I do not currently allow the following simoptions to differ by PType

for ii=1:N_i
    iistr=Names_i{ii};

    % This ptype's selection: its own whichcombos slot, plus whatever the grouped stats need. The grouped Mean, StdDeviation, Minimum
    % and Maximum are built from every ptype's, so a grouped slot that is on forces this ptype's computation of that combination with
    % at least the grouped whichstats (the result is reported, as a byproduct).
    if isempty(whichstatsAll)
        whichstats_ii=whichstatsG;
    else
        whichstats_ii=reshape(whichstatsAll(:,:,ii,:),[numFnsToEvaluate,nwhichpages,7]); % this ptype's slot
    end
    wsForced=whichstatsG.*repmat(whichcombosG,[1,1,7]); % the grouped whichstats where the grouped slot is on
    lor=(whichstats_ii(:,:,4)==1 | whichstats_ii(:,:,4)==2 | wsForced(:,:,4)==1 | wsForced(:,:,4)==2);
    whichstats_ii=max(whichstats_ii,wsForced);
    ws4=whichstats_ii(:,:,4);
    ws4(lor)=1; % Gini/Lorenz codes: a Lorenz-giving 1 or 2 on either side wins over the Gini-only 3 (AllStats uses 1)
    whichstats_ii(:,:,4)=ws4;
    whichcombos_ii=max(whichcombosAll(:,:,ii),whichcombosG).*any(whichstats_ii,3); % a combination asking for no statistic is skipped

    tic;
    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    PolicyIndexes_temp=gpuArray(Policy.(iistr));

    %% Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end

    % A few other things we can do in outer loop
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=length(n_d_temp);
    end
    l_a_temp=length(n_a_temp);

    N_a_temp=prod(n_a_temp);

    % Switch to z_gridvals_temp (folding e and semiz into z if appropriate)
    if ~isfield(simoptions_temp,'alreadygridvals')
        simoptions_temp.alreadygridvals=0;
    end
    if ~isfield(simoptions_temp,'alreadygridvals_semiexo')
        simoptions_temp.alreadygridvals_semiexo=0;
    end
    [n_z_temp,z_gridvals_temp,N_z_temp,l_z_temp,simoptions_temp]=CreateGridvals_FnsToEvaluate_InfHorz(n_z_temp,z_grid_temp,simoptions_temp,Parameters_temp);
    if N_z_temp==0
        N_z_temp=1; % Just makes things easier below
    end

    % Switch to PolicyVals
    PolicyValues_temp=PolicyInd2Val_InfHorz(PolicyIndexes_temp,n_d_temp,n_a_temp,n_z_temp,d_grid_temp,a_grid_temp,simoptions_temp);
    % Note: must collapse n_a (and n_z) into N_a (and N_z) before the permute, as a_gridvals is
    % the joint grid over N_a. [Otherwise, with two endogenous states, the assets stay split and
    % do not match a_gridvals; only coincides when there is a single endogenous state.]
    % Note: use l_z_temp (not N_z_temp) to detect no-z, as N_z_temp was just set to 1 above
    if l_z_temp==0
        PolicyValuesPermute_temp=permute(reshape(PolicyValues_temp,[size(PolicyValues_temp,1),N_a_temp]),[2,1]); %[N_a,l_d+l_a]
    else
        PolicyValuesPermute_temp=permute(reshape(PolicyValues_temp,[size(PolicyValues_temp,1),N_a_temp,N_z_temp]),[2,3,1]); %[N_a,N_z,l_d+l_a]
    end

    l_daprime_temp=size(PolicyValues_temp,1); % Note, do this off of value not indexes, so that things like gridinterplayer have already been handled

    a_gridvals_temp=CreateGridvals(n_a_temp,a_grid_temp,1);

    [FnsToEvaluate_temp,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;



    %% Some things that don't need to go in the loop over FnsToEvalaute
    StationaryDist_ii=gpuArray(reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp,1])); % Note: does not impose *StationaryDist.ptweights(ii)
    
    % Eliminate all the zero-weighted points (this doesn't really save runtime for the exact calculation and often can increase it, but
    % for the createDigest it slashes the runtime. So since we want it then we may as well do it now.)
    temp=logical(StationaryDist_ii~=0);
    % StationaryDist_ii=StationaryDist_ii(temp); % This has to happen after the conditional restriction dist is calculated

    %% Evaluate conditional restrictions for this PType (note: these use simoptions not simoptions_temp)
    if useCondlRest==1
        RestrictionStruct_ii=struct();

        % For each conditional restriction, create a 'restricted stationary distribution'
        for rr=1:length(CondlRestnFnNames)
            CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
            % Get parameter names for Conditional Restriction functions
            tempnames=getAnonymousFnInputNames(CondlRestnFn);
            if length(tempnames)>(l_d_temp+l_a_temp+l_a_temp+l_z_temp)
                CondlRestnFnParamNames={tempnames{l_d_temp+l_a_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
            else
                CondlRestnFnParamNames={};
            end
            CondlRestnFnParamsCell=CreateCellFromParams(Parameters_temp,CondlRestnFnParamNames); % the parameters of this ptype

            RestrictionValues=logical(EvalFnOnAgentDist_Grid(CondlRestnFn, CondlRestnFnParamsCell,PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,n_z_temp,a_gridvals_temp,z_gridvals_temp));
            RestrictionValues=reshape(RestrictionValues,[N_a_temp*N_z_temp,1]);

            RestrictedStationaryDistVec=StationaryDist_ii;
            RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction
            RestrictedStationaryDistVec=RestrictedStationaryDistVec(temp); % This has already been done to StationaryDist_ii, so have to do it to Restricted Agent Dist

            % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
            restrictedsamplemass(ii,rr)=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass(ii,rr); % Normalize to mass of 1
            % Note: if the restriction is zero mass for this ptype the restricted stats are NaN (0/0), which is the
            % correct behaviour: the conditional moment of a group that does not exist is unknown (and the grouped
            % stats are then also NaN).
            % Store for later
            RestrictionStruct_ii(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec;

            if restrictedsamplemass(ii,rr)==0 && simoptions.warnzerorestrictedmass==2
                warning(['One of the conditional restrictions evaluates to a zero mass for permanent type ',iistr])
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
                AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.(iistr)=restrictedsamplemass(ii,rr); % Just return this and hopefully it is clear to the user
            else
                AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.(iistr)=restrictedsamplemass(ii,rr); % Seems likely this would be something user might want
            end

        end
    end


    %%
    StationaryDist_ii=StationaryDist_ii(temp);

    %%
    for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
        if FnsAndPTypeIndicator_ii(ff)==1 && any(whichcombos_ii(ff,:)) % If this function is relevant to this ptype (and some stat of it is wanted for this ptype: its own slot, or forced by the grouped slot)

            % Get parameter names for current FnsToEvaluate functions
            tempnames=getAnonymousFnInputNames(FnsToEvaluate_temp.(FnsToEvalNames{ff}));
            if length(tempnames)>(l_d_temp+l_a_temp+l_a_temp+l_z_temp)
                FnsToEvaluateParamNames={tempnames{l_d_temp+l_a_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
            else
                FnsToEvaluateParamNames={};
            end
            FnsToEvaluateParamsCell=CreateCellFromParams(Parameters_temp,FnsToEvaluateParamNames); % the parameters of this ptype

            %% We have set up the current PType, now do some calculations for it.
            simoptions_temp.keepoutputasmatrix=1;
            ValuesOnGrid_ii=EvalFnOnAgentDist_Grid(FnsToEvaluate_temp.(FnsToEvalNames{ff}), FnsToEvaluateParamsCell, PolicyValuesPermute_temp, l_daprime_temp, n_a_temp, n_z_temp, a_gridvals_temp, z_gridvals_temp);

            ValuesOnGrid_ii=reshape(ValuesOnGrid_ii,[N_a_temp*N_z_temp,1]);

            % StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp,1]); % Note: does not impose *StationaryDist.ptweights(ii)

            % Eliminate all the zero-weighted points (this doesn't really save runtime for the exact calculation and often can increase it, but
            % for the createDigest it slashes the runtime. So since we want it then we may as well do it now.)
            % temp=logical(StationaryDist_ii~=0);
            % StationaryDist_ii=StationaryDist_ii(temp);
            ValuesOnGrid_ii=ValuesOnGrid_ii(temp);

            % I want to use unique to make it easier to put the different agent
            % ptypes together (as all the matrices are typically smaller).
            % May as well do it before doing the StatsFromWeightedGrid
            [SortedValues,~,sortindex]=unique(ValuesOnGrid_ii); % Note: unique() automatically also sorts
            SortedWeights=accumarray(sortindex,StationaryDist_ii,[],@sum);

            %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and ff)
            if whichcombos_ii(ff,1)==1 % the unrestricted stats of this function are wanted for this ptype (its own slot, or forced by the grouped slot)
                ws1_ii=reshape(whichstats_ii(ff,1,:),[1,7]); % whichstats of this ptype for the unrestricted stats (its own, plus what the grouped stats need)
                AllStats.(FnsToEvalNames{ff}).(iistr)=StatsFromWeightedGrid(SortedValues,SortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws1_ii); % 1 is presorted
            end

            %% If using conditional restrictions, do those
            if useCondlRest==1
                for rr=1:length(CondlRestnFnNames)
                    if whichcombos_ii(ff,1+rr)==1 % this restriction is wanted for this function for this ptype (its own slot, or forced by the grouped slot)
                        RestrictedSortedWeights=accumarray(sortindex,RestrictionStruct_ii(rr).RestrictedStationaryDistVec,[],@sum); % This has already been done to SortedValues, so have to do it to Restricted Agent Dist
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr)=StatsFromWeightedGrid(SortedValues,RestrictedSortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,reshape(whichstats_ii(ff,1+rr,:),[1,7])); % 2 is presorted but with zero weights (the points outside the restriction): they must be dropped, or a zero-weight first point hides negative values from the Gini/Lorenz/shares check
                        % If doing grouped stats, store RestrictedSortedWeights
                        if whichcombosG(ff,1+rr)==0 % the grouped stats of this restriction are not wanted: nothing to append
                        elseif simoptions_temp.groupusingtdigest==1
                            error('Code should never get here (should have thrown an error earlier')
                        else
                            % Population-weighted, un-normalised restricted mass of this ptype.
                            % restrictedsamplemass(ii,rr) is needed because RestrictedSortedWeights was
                            % normalized to mass one above: without it every ptype enters with its full
                            % population weight regardless of how little of it meets the restriction.
                            % The zero-mass branch is needed because such a ptype had its
                            % RestrictedStationaryDistVec normalized 0/0 above and is all NaN; those NaNs
                            % would propagate into the pooled weights and make every grouped restricted
                            % statistic NaN. A ptype absent from the group contributes nothing to it.
                            if restrictedsamplemass(ii,rr)>0
                                RestrictedWeights_ii=RestrictedSortedWeights*(StationaryDist.ptweights(ii)*restrictedsamplemass(ii,rr));
                            else
                                RestrictedWeights_ii=zeros(size(RestrictedSortedWeights),'like',RestrictedSortedWeights);
                            end
                            if simoptions.ptypestorecpu==1
                                AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=[AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}); gather(RestrictedWeights_ii)];
                            else
                                AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=[AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}); RestrictedWeights_ii];
                            end
                            % Note: later once we have all the ii do AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}) renormalized by sum(ptweights.*restrictedsamplemass(:,rr))
                        end
                    end
                end
            end

            %% For later, put the mean and std dev in a convenient place
            if whichcombos_ii(ff,1)==1 && ws1_ii(1)==1
                MeanVec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).Mean;
            end
            if whichcombos_ii(ff,1)==1 && ws1_ii(3)==1
                StdDevVec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).StdDeviation;
            end
            % Do the same with the minimum and maximum
            if whichcombos_ii(ff,1)==1 && ws1_ii(5)==1 && StationaryDist.ptweights(ii)>0 % a ptype of zero mass is not in the population, so it must not set the grouped min/max (its entries stay NaN, which min()/max() ignore)
                minvaluevec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).Minimum;
                maxvaluevec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).Maximum;
            end

            if ~any(whichcombosG(ff,:)) % no grouped stat of this function is wanted: nothing to pool
            elseif simoptions_temp.groupusingtdigest==1
                Cmerge=AllCMerge.(FnsToEvalNames{ff});
                digestweightsmerge=Alldigestweightsmerge.(FnsToEvalNames{ff});

                %% Create digest (if unique() was not enough to make them small)
                [C_ii,digestweights_ii,~]=createDigest(SortedValues, SortedWeights,delta,1); % 1 is presorted

                merge_nsofar2(ff)=merge_nsofar(ff)+length(C_ii);
                Cmerge(merge_nsofar(ff)+1:merge_nsofar2(ff))=C_ii;
                digestweightsmerge(merge_nsofar(ff)+1:merge_nsofar2(ff))=digestweights_ii*StationaryDist.ptweights(ii);
                merge_nsofar(ff)=merge_nsofar2(ff);

                AllCMerge.(FnsToEvalNames{ff})=Cmerge;
                Alldigestweightsmerge.(FnsToEvalNames{ff})=digestweightsmerge;
            else
                if simoptions.ptypestorecpu==1
                    AllValues.(FnsToEvalNames{ff})=[AllValues.(FnsToEvalNames{ff}); gather(SortedValues)];
                    AllWeights.(FnsToEvalNames{ff})=[AllWeights.(FnsToEvalNames{ff}); gather(SortedWeights)*gather(StationaryDist.ptweights(ii))];
                else
                    AllValues.(FnsToEvalNames{ff})=[AllValues.(FnsToEvalNames{ff}); SortedValues];
                    AllWeights.(FnsToEvalNames{ff})=[AllWeights.(FnsToEvalNames{ff}); SortedWeights*StationaryDist.ptweights(ii)];
                end
            end
        end
    end
end



%% Now for the grouped stats, putting the ptypes together
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        % Population mass in the restriction. Note the per-ptype RestrictedSampleMass.(iistr) above are WITHIN-type shares, so this is
        % their ptweights-weighted sum, not their plain sum (which is not a mass at all and can exceed one). Always filled, whichcombos or not.
        AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.TotalAllPTypes=sum(StationaryDist.ptweights(:).*restrictedsamplemass(:,rr));
        if AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.TotalAllPTypes==0 && simoptions.warnzerorestrictedmass>=1
            warning('One of the conditional restrictions evaluates to a zero mass (for every permanent type)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end
for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
    if ~any(whichcombosG(ff,:)) % no grouped stat of this function is wanted: nothing was pooled, nothing to compute
        continue
    end
    ws1=reshape(whichstatsG(ff,1,:),[1,7]); % whichstats of the grouped unrestricted stats of this function

    if simoptions_temp.groupusingtdigest==1
        Cmerge=AllCMerge.(FnsToEvalNames{ff});
        digestweightsmerge=Alldigestweightsmerge.(FnsToEvalNames{ff});
        % Clean off the zeros at the end of Cmerge (that exist because of how we preallocate 'too much' for Cmerge); same for digestweightsmerge.
        Cmerge=Cmerge(1:merge_nsofar(ff));
        digestweightsmerge=digestweightsmerge(1:merge_nsofar(ff));

        % Merge the digests
        [C_kk,digestweights_kk,~]=mergeDigest(Cmerge, digestweightsmerge, delta);

        if whichcombosG(ff,1)==1 % the grouped unrestricted stats of this function are wanted
            tempStats=StatsFromWeightedGrid(C_kk,digestweights_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws1);
            allstatnames=fieldnames(tempStats);
        end
    else
        % Do unique() before we calculate stats
        [AllValues.(FnsToEvalNames{ff}),~,sortindex]=unique(AllValues.(FnsToEvalNames{ff}));
        AllWeights.(FnsToEvalNames{ff})=accumarray(sortindex,AllWeights.(FnsToEvalNames{ff}),[],@sum);

        if whichcombosG(ff,1)==1 % the grouped unrestricted stats of this function are wanted
            tempStats=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}),AllWeights.(FnsToEvalNames{ff}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,ws1); % 2: sorted, but a ptype of zero mass contributes zero weights
            allstatnames=fieldnames(tempStats);
        end
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                if whichcombosG(ff,1+rr)==0 % the grouped stats of this restriction are not wanted for this function
                    continue
                end
                AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=accumarray(sortindex,AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})/sum(StationaryDist.ptweights(:).*restrictedsamplemass(:,rr)),[],@sum);
                tempStatsRestricted=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}),AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,reshape(whichstatsG(ff,1+rr,:),[1,7])); % 2: sorted, with zero weights (outside the restriction)
                % Following is necessary as just AllStats=StatsFromWeightedGrid() overwrote the existing subfields
                rallstatnames=fieldnames(tempStatsRestricted);
                for aa=1:length(rallstatnames)
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(rallstatnames{aa})=tempStatsRestricted.(rallstatnames{aa});
                end
            end
        end
    end
    if whichcombosG(ff,1)==1 % store the grouped unrestricted stats of this function
        % Following is necessary as just AllStats=StatsFromWeightedGrid() overwrote the existing subfields
        % allstatnames=fieldnames(tempStats);
        for aa=1:length(allstatnames)
            AllStats.(FnsToEvalNames{ff}).(allstatnames{aa})=tempStats.(allstatnames{aa});
        end


        % Grouped mean and standard deviation are overwritten on a more direct calculation that does not involve the digests
        SigmaNxi=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights)'); % The sum of the masses of the relevant types

        % Mean
        if ws1(1)==1
            AllStats.(FnsToEvalNames{ff}).Mean=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*MeanVec(ff,:))/SigmaNxi;
        end

        % Standard Deviation
        if ws1(3)==1
            if N_i==1
                AllStats.(FnsToEvalNames{ff}).StdDeviation=StdDevVec(ff,:);
            else
                temp2=zeros(N_i,1);
                for ii=2:N_i
                    if FnsAndPTypeIndicator(ff,ii)==1
                        temp=MeanVec(ff,1:(ii-1))-MeanVec(ff,ii); % This bit with temp is just to handle numerical rounding errors where temp evaluated to negative with order -15
                        if any(temp<0) && all(temp>10^(-12))
                            temp=max(temp,0);
                        end
                        temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator(ff,1:(ii-1)).*(StationaryDist.ptweights(1:(ii-1))').*(temp.^2));
                    end
                end
                AllStats.(FnsToEvalNames{ff}).StdDeviation=sqrt(sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*(StdDevVec(ff,:).^2))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
            end
            AllStats.(FnsToEvalNames{ff}).Variance=(AllStats.(FnsToEvalNames{ff}).StdDeviation)^2;
        end

        % With t-Digests the pooled extremes are only approximate, so the minimum and maximum are taken directly from the ptypes (and
        % overwritten); without them the pooled stats are exact, and the minimum and maximum are those of the pooled population, as the
        % other stats (the min/max over the ptypes differs from them, as the tolerance is then a mass within each ptype, not within the population)
        if ws1(5)==1 && simoptions.groupusingtdigest==1
            AllStats.(FnsToEvalNames{ff}).Maximum=max(maxvaluevec(ff,:));
            AllStats.(FnsToEvalNames{ff}).Minimum=min(minvaluevec(ff,:));
        end
    end
end




end
