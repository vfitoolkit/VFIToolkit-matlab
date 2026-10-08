function AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
% simoptions.whichcombos ([numFnsToEvaluate, 1+number of conditional restrictions], optionally with a trailing type dimension of N_i+1) selects
% which (fn, restriction[, ptype or grouped]) combinations are computed, and simoptions.whichstats may be given per combination; see below.
% Loop order: the ptype setup is done once (pass 0: policy values, grids, parameters, and the conditional restrictions as logical
% masks). Then for each FnsToEvaluate: the ptype loop evaluates it, computes the per-ptype stats, and appends each ptype to one pooled
% cell, which is pooled across ptypes as soon as the ptype loop ends. Memory use is therefore bounded by a small multiple of the agent
% distribution plus one function's pooled cell, and does not grow with the number of FnsToEvaluate or conditional restrictions. (The
% earlier design kept every function's cell for every ptype until the end.)
% Reports a variety of stats, both grouped and by PType.
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
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    % simoptions.conditionalrestrictions  % Evaluate AllStats, but conditional on the restriction being equal to one (not zero).
    simoptions.gridinterplayer=0;
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0;
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
        simoptions.verboseparams=0;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
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
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
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

FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');

%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(N_i,length(CondlRestnFnNames));

    if simoptions.groupusingtdigest==1
        error('Have not implemented simoptions.groupusingtdigest==1 together with simoptions.conditionalrestrictions')
    end
end
%% simoptions.whichcombos and per-combination simoptions.whichstats
% whichcombos: [numFnsToEvaluate, 1+number of conditional restrictions] of zeros/ones ([numFnsToEvaluate,1] without restrictions):
% page 1 is the unrestricted stats, pages 2:end the restrictions in the fieldnames order of simoptions.conditionalrestrictions. Ones
% are computed, zeros skipped (their output fields are simply absent; RestrictedSampleMass is always filled). Default all ones. A
% vector of length numFnsToEvaluate with restrictions is applied to every page. Intended for calibration/estimation.
% whichstats: the usual 1x7 vector, or [numFnsToEvaluate, 1+number of restrictions, 7] giving a whichstats vector for every
% combination ([numFnsToEvaluate,7] is applied to every page). A combination asking for no statistic is skipped like a whichcombos zero.
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end
% A trailing type dimension may be added (2026-10-07): whichcombos [.., N_i+1] and whichstats [.., N_i+1, 7] select per permanent
% type, in the order of Names_i, with the last slot the grouped stats. The shapes without it apply to every ptype and to the grouped
% stats alike. The grouped Mean, StdDeviation, Minimum and Maximum are built from every ptype's, so a grouped slot that is on also
% computes (and reports) that combination for every ptype.
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

%% Pass 0: per-type setup, done once and kept for the loop over FnsToEvaluate
PT=struct();
RestrictionMask=cell(N_i,1); % becomes cell(N_i,nRestr) on first use; logical over the (a,z,j) grid of each ptype
for ii=1:N_i
    iistr=Names_i{ii};

    tic;

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(iistr));
        % StationaryDist_temp=gpuArray(StationaryDist.(iistr));
    else
        PolicyIndexes_temp=Policy.(iistr);
        % StationaryDist_temp=StationaryDist.(iistr);
    end

    %% Go through everything which might be dependent on fixed type (PType)
    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end

    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    % Exogenous shocks
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);

    % Parameters
    Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.

    if simoptions_temp.verboseparams==1
        fprintf('Parameter values for the current permanent type \n')
        Parameters_temp
    end

    % A few other things we can do in outer loop
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    N_a_temp=prod(n_a_temp);

    a_gridvals_temp=CreateGridvals(n_a_temp,a_grid_temp,1);
    % Turn (semiz,z,e) into z_gridvals_J_temp as FnsToEvalute do not distinguish them
    [n_z_temp,z_gridvals_J_temp,N_z_temp,l_z_temp,simoptions_temp]=CreateGridvals_FnsToEvaluate_FHorz(n_z_temp,z_grid_temp,N_j_temp,simoptions_temp,Parameters_temp);
    if N_z_temp==0
        N_z_temp=1; % Just makes things easier below
    end

    % Switch to PolicyVals
    PolicyValues_temp=PolicyInd2Val_FHorz(PolicyIndexes_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,simoptions_temp,1);
    if l_z_temp==0
        PolicyValuesPermute_temp=permute(PolicyValues_temp,[2,3,1]); % (N_a,N_j,l_daprime)
    else
        PolicyValuesPermute_temp=permute(PolicyValues_temp,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
    end
    l_daprime_temp=size(PolicyValues_temp,1);

    [FnsToEvaluate_temp,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

    StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp*N_j_temp,1]); % Note: does not impose *StationaryDist.ptweights(ii)

    %% Evaluate conditional restrictions for this PType, kept as logical masks (note: these use simoptions not simoptions_temp)
    if useCondlRest==1
        for rr=1:length(CondlRestnFnNames)
            % The current conditional restriction function
            CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
            % Get parameter names for Conditional Restriction functions
            temp2=getAnonymousFnInputNames(CondlRestnFn);
            if length(temp2)>(l_daprime_temp+l_a_temp+l_z_temp)
                CondlRestnFnParamNames={temp2{l_daprime_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
            else
                CondlRestnFnParamNames={};
            end

            if l_z_temp==0
                CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters_temp,CondlRestnFnParamNames,N_j_temp,2); % j in 2nd dimension: (a,j,l_d+l_a), so we want j to be after N_a
                RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CellOverAgeOfParamValues,PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,0,a_gridvals_temp,[]));
            else
                CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters_temp,CondlRestnFnParamNames,N_j_temp,3); % j in 3rd dimension: (a,z,j,l_d+l_a), so we want j to be after N_a and N_z
                RestrictionValues=logical(EvalFnOnAgentDist_Grid_J(CondlRestnFn,CellOverAgeOfParamValues,PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,n_z_temp,a_gridvals_temp,z_gridvals_J_temp));
            end
            RestrictionValues=reshape(RestrictionValues,[N_a_temp*N_z_temp*N_j_temp,1]);

            restrictedsamplemass(ii,rr)=sum(StationaryDist_ii.*RestrictionValues); % mass within this ptype that satisfies the restriction
            if simoptions.ptypestorecpu==1
                RestrictionMask{ii,rr}=gather(RestrictionValues);
            else
                RestrictionMask{ii,rr}=RestrictionValues;
            end
            % Note: if the restriction is zero mass for this ptype its restricted stats are NaN (0/0 below), which is the correct
            % behaviour: the conditional moment of a group that does not exist is unknown (and the grouped stats are then also NaN).
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.(iistr)=restrictedsamplemass(ii,rr); % Seems likely this would be something user might want
            % (a type with no mass under the restriction is not a warning: the population-level check is after the loop over types)
        end
    end

    % Keep the per-type setup for the loop over FnsToEvaluate
    PT(ii).simoptions_temp=simoptions_temp;
    PT(ii).Parameters_temp=Parameters_temp;
    PT(ii).n_a_temp=n_a_temp;
    PT(ii).n_z_temp=n_z_temp;
    PT(ii).N_j_temp=N_j_temp;
    PT(ii).l_a_temp=l_a_temp;
    PT(ii).l_z_temp=l_z_temp;
    PT(ii).N_a_temp=N_a_temp;
    PT(ii).N_z_temp=N_z_temp;
    PT(ii).a_gridvals_temp=a_gridvals_temp;
    PT(ii).z_gridvals_J_temp=z_gridvals_J_temp;
    PT(ii).l_daprime_temp=l_daprime_temp;
    PT(ii).FnsToEvaluate_temp=FnsToEvaluate_temp;
    PT(ii).FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii;
    if simoptions.ptypestorecpu==1
        PT(ii).PolicyValuesPermute_temp=gather(PolicyValuesPermute_temp);
    else
        PT(ii).PolicyValuesPermute_temp=PolicyValuesPermute_temp;
    end
end % end ii over N_i (pass 0)
clear PolicyValues_temp PolicyValuesPermute_temp PolicyIndexes_temp RestrictionValues StationaryDist_ii temp

%% Now for the grouped stats, putting the ptypes together
if useCondlRest==1
    for rr=1:length(CondlRestnFnNames)
        % Population mass in the restriction: the per-ptype RestrictedSampleMass.(iistr) above are WITHIN-type shares, so this is their
        % ptweights-weighted sum (not their plain sum, which is not a mass at all and can exceed one). Always filled, whichcombos or not.
        AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.TotalAllPTypes=sum(StationaryDist.ptweights(:).*restrictedsamplemass(:,rr));
        if AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.TotalAllPTypes==0
            warning('One of the conditional restrictions evaluates to a zero mass (for every permanent type)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
        end
    end
end
if simoptions.groupusingtdigest==1
    delta=10000;
end

%% Main loop: for each FnsToEvaluate, the ptypes (per-ptype stats, appended to one pooled cell), then the pooled cell
for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
    if ~any(whichcombosAll(ff,:,:),'all') % no combination of this function is wanted for any ptype or for the grouped stats
        continue
    end
    ws1=reshape(whichstatsG(ff,1,:),[1,7]); % whichstats of the grouped unrestricted stats of this function

    % The pooled cell for this function
    PoolValues=[];
    PoolWeights=[];
    if useCondlRest==1
        PoolRestrWeights=cell(length(CondlRestnFnNames),1);
        for rr=1:length(CondlRestnFnNames)
            PoolRestrWeights{rr}=[];
        end
    end
    if simoptions.groupusingtdigest==1
        Cmerge=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
        digestweightsmerge=zeros(5000*N_i,1);
        merge_nsofar=0;
    end

    for ii=1:N_i
        % This ptype's selection: its own whichcombos slot, plus whatever the grouped stats need. The grouped Mean, StdDeviation, Minimum
        % and Maximum are built from every ptype's, so a grouped slot that is on forces this ptype's computation of that combination with
        % at least the grouped whichstats (the result is reported, as a byproduct).
        if isempty(whichstatsAll)
            whichstats_ii=whichstatsG;
        else
            whichstats_ii=reshape(whichstatsAll(:,:,ii,:),[numFnsToEvaluate,nwhichpages,7]); % this ptype's slot
        end
        wsForced=whichstatsG.*repmat(whichcombosG,[1,1,7]); % the grouped whichstats where the grouped slot is on
        wsForced(:,:,1)=max(wsForced(:,:,1),wsForced(:,:,3)); % the grouped StdDeviation is rebuilt below from every ptype's Mean and StdDeviation, so a grouped StdDeviation needs the ptype Means too (before 2026-10-07 MeanVec stayed NaN and so did the grouped StdDeviation when only it was asked for)
        lor=(whichstats_ii(:,:,4)==1 | whichstats_ii(:,:,4)==2 | wsForced(:,:,4)==1 | wsForced(:,:,4)==2);
        whichstats_ii=max(whichstats_ii,wsForced);
        ws4=whichstats_ii(:,:,4);
        ws4(lor)=1; % Gini/Lorenz codes: a Lorenz-giving 1 or 2 on either side wins over the Gini-only 3 (AllStats uses 1)
        whichstats_ii(:,:,4)=ws4;
        whichcombos_ii=max(whichcombosAll(:,:,ii),whichcombosG).*any(whichstats_ii,3); % a combination asking for no statistic is skipped
        iistr=Names_i{ii};
        simoptions_temp=PT(ii).simoptions_temp;
        Parameters_temp=PT(ii).Parameters_temp;
        n_a_temp=PT(ii).n_a_temp;
        n_z_temp=PT(ii).n_z_temp;
        N_j_temp=PT(ii).N_j_temp;
        l_a_temp=PT(ii).l_a_temp;
        l_z_temp=PT(ii).l_z_temp;
        N_a_temp=PT(ii).N_a_temp;
        N_z_temp=PT(ii).N_z_temp;
        a_gridvals_temp=PT(ii).a_gridvals_temp;
        z_gridvals_J_temp=PT(ii).z_gridvals_J_temp;
        l_daprime_temp=PT(ii).l_daprime_temp;
        FnsToEvaluate_temp=PT(ii).FnsToEvaluate_temp;
        FnsAndPTypeIndicator_ii=PT(ii).FnsAndPTypeIndicator_ii;
        if simoptions.ptypestorecpu==1
            PolicyValuesPermute_temp=gpuArray(PT(ii).PolicyValuesPermute_temp);
        else
            PolicyValuesPermute_temp=PT(ii).PolicyValuesPermute_temp;
        end

        if FnsAndPTypeIndicator_ii(ff)==1 && any(whichcombos_ii(ff,:)) % If this function is relevant to this ptype (and some stat of it is wanted for this ptype: its own slot, or forced by the grouped slot)
            StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp*N_j_temp,1]); % Note: does not impose *StationaryDist.ptweights(ii)
            temp=logical(StationaryDist_ii~=0); % the points with mass (this doesn't really save runtime for the exact calculation, but it slashes it for createDigest)
            StationaryDist_ii=StationaryDist_ii(temp);

            % Get parameter names for current FnsToEvaluate functions
            tempnames=getAnonymousFnInputNames(FnsToEvaluate_temp.(FnsToEvalNames{ff}));
            if length(tempnames)>(l_daprime_temp+l_a_temp+l_z_temp)
                FnsToEvaluateParamNames={tempnames{l_daprime_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
            else
                FnsToEvaluateParamNames={};
            end
            if l_z_temp==0
                CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters_temp,FnsToEvaluateParamNames,N_j_temp,2);
            else
                CellOverAgeOfParamValues=CreateCellOverAgeFromParams(Parameters_temp,FnsToEvaluateParamNames,N_j_temp,3);
            end

            %% We have set up the current PType, now do some calculations for it.
            simoptions_temp.keepoutputasmatrix=1;
            ValuesOnGrid_ii=EvalFnOnAgentDist_Grid_J(FnsToEvaluate_temp.(FnsToEvalNames{ff}),CellOverAgeOfParamValues,PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,n_z_temp,a_gridvals_temp,z_gridvals_J_temp);
            ValuesOnGrid_ii=reshape(ValuesOnGrid_ii,[N_a_temp*N_z_temp*N_j_temp,1]);
            ValuesOnGrid_ii=ValuesOnGrid_ii(temp);

            % I want to use unique to make it easier to put the different agent
            % ptypes together (as all the matrices are typically smaller).
            % May as well do it before doing the StatsFromWeightedGrid
            [SortedValues,~,sortindex]=unique(ValuesOnGrid_ii);
            SortedWeights=accumarray(sortindex,StationaryDist_ii,[],@sum);

            %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and kk)
            if whichcombos_ii(ff,1)==1 % the unrestricted stats of this function are wanted for this ptype (its own slot, or forced by the grouped slot)
                ws1_ii=reshape(whichstats_ii(ff,1,:),[1,7]); % whichstats of this ptype for the unrestricted stats (its own, plus what the grouped stats need)
            AllStats.(FnsToEvalNames{ff}).(iistr)=StatsFromWeightedGrid(SortedValues,SortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws1_ii); % 1 is presorted
            end % whichcombos_ii(ff,1)

            %% If using conditional restrictions, do those (the restricted weights come from the stored masks)
            if useCondlRest==1
                for rr=1:length(CondlRestnFnNames)
                    if whichcombos_ii(ff,1+rr)==1 % this restriction is wanted for this function for this ptype (its own slot, or forced by the grouped slot)
                        if simoptions.ptypestorecpu==1
                            RestrictionMask_iirr=gpuArray(RestrictionMask{ii,rr});
                        else
                            RestrictionMask_iirr=RestrictionMask{ii,rr};
                        end
                        RestrictedSortedWeights=accumarray(sortindex,StationaryDist_ii.*RestrictionMask_iirr(temp),[],@sum); % the restricted mass of this ptype on its sorted support (un-normalised; zero if the restriction has zero mass for this ptype)
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr)=StatsFromWeightedGrid(SortedValues,RestrictedSortedWeights/restrictedsamplemass(ii,rr),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,reshape(whichstats_ii(ff,1+rr,:),[1,7])); % normalised to mass one (0/0 is NaN for a zero-mass restriction, see above); 2: sorted, with zero weights (outside the restriction), which must be dropped or a zero-weight first point hides negative values from the Gini/Lorenz/shares check

                        % Append to the pooled cell: ptweights(ii) times the un-normalised restricted mass, so each ptype enters the grouped restricted stats by its
                        % population mass in the restriction (later normalised by sum(ptweights.*restrictedsamplemass(:,rr))); a zero-mass ptype contributes zeros
                        if whichcombosG(ff,1+rr)==0 % the grouped stats of this restriction are not wanted: nothing to append
                        elseif simoptions.ptypestorecpu==1
                            PoolRestrWeights{rr}=[PoolRestrWeights{rr}; gather(RestrictedSortedWeights)*gather(StationaryDist.ptweights(ii))];
                        else
                            PoolRestrWeights{rr}=[PoolRestrWeights{rr}; RestrictedSortedWeights*StationaryDist.ptweights(ii)];
                        end
                    end % whichcombos_ii(ff,1+rr)
                end
            end

            if whichcombos_ii(ff,1)==1 % (ws1_ii was set above)
            %% For later, put the mean and std dev in a convenient place
            if ws1_ii(1)==1
                MeanVec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).Mean;
            end
            if ws1_ii(3)==1
                StdDevVec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).StdDeviation;
            end
            % Do the same with the minimum and maximum
            if ws1_ii(5)==1 && StationaryDist.ptweights(ii)>0 % a ptype of zero mass is not in the population, so it must not set the grouped min/max (its entries stay NaN, which min()/max() ignore)
                minvaluevec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).Minimum;
                maxvaluevec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(iistr).Maximum;
            end
            end % whichcombos_ii(ff,1)

            % Append this ptype to the pooled cell
            if ~any(whichcombosG(ff,:)) % no grouped stat of this function is wanted: nothing to pool
            elseif simoptions.groupusingtdigest==1
                [C_ii,digestweights_ii,~]=createDigest(SortedValues, SortedWeights,delta,1); % 1 is presorted
                Cmerge(merge_nsofar+1:merge_nsofar+length(C_ii))=C_ii;
                digestweightsmerge(merge_nsofar+1:merge_nsofar+length(C_ii))=digestweights_ii*StationaryDist.ptweights(ii);
                merge_nsofar=merge_nsofar+length(C_ii);
            elseif simoptions.ptypestorecpu==1
                PoolValues=[PoolValues; gather(SortedValues)];
                PoolWeights=[PoolWeights; gather(SortedWeights)*gather(StationaryDist.ptweights(ii))];
            else
                PoolValues=[PoolValues; SortedValues];
                PoolWeights=[PoolWeights; SortedWeights*StationaryDist.ptweights(ii)];
            end
        end % this function is relevant to this ptype
    end % end ii over N_i

    %% Pool this function's cell across the ptypes and compute the grouped stats
    if ~any(whichcombosG(ff,:)) % no grouped stat of this function is wanted: nothing was pooled, nothing to compute
    elseif simoptions.groupusingtdigest==1
        Cmerge=Cmerge(1:merge_nsofar);
        digestweightsmerge=digestweightsmerge(1:merge_nsofar);
        [C_kk,digestweights_kk,~]=mergeDigest(Cmerge, digestweightsmerge, delta);
        if whichcombosG(ff,1)==1 % the grouped unrestricted stats of this function are wanted
            tempStats=StatsFromWeightedGrid(C_kk,digestweights_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws1);
            allstatnames=fieldnames(tempStats);
        end % whichcombosG(ff,1)
    else
        % Do unique() before we calculate stats
        [PoolValues,~,sortindex]=unique(PoolValues);
        PoolWeights=accumarray(sortindex,PoolWeights,[],@sum);
        if whichcombosG(ff,1)==1 % the grouped unrestricted stats of this function are wanted
            tempStats=StatsFromWeightedGrid(PoolValues,PoolWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,ws1); % 2: sorted, but a ptype of zero mass contributes zero weights
            allstatnames=fieldnames(tempStats);
        end % whichcombosG(ff,1)
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                if whichcombosG(ff,1+rr)==1 % the grouped stats of this restriction are wanted for this function
                    PoolRestrWeights{rr}=accumarray(sortindex,PoolRestrWeights{rr}/sum(StationaryDist.ptweights(:).*restrictedsamplemass(:,rr)),[],@sum); % normalised by the population mass in the restriction
                    tempStatsRestricted=StatsFromWeightedGrid(PoolValues,PoolRestrWeights{rr},simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,reshape(whichstatsG(ff,1+rr,:),[1,7])); % 2: sorted, with zero weights (outside the restriction)
                    % Following is necessary as just AllStats=StatsFromWeightedGrid() overwrote the existing subfields
                    rallstatnames=fieldnames(tempStatsRestricted);
                    for aa=1:length(rallstatnames)
                        AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(rallstatnames{aa})=tempStatsRestricted.(rallstatnames{aa});
                    end
                end % whichcombosG(ff,1+rr)
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
    end % whichcombosG(ff,1)
end % end ff over FnsToEvaluate


end
