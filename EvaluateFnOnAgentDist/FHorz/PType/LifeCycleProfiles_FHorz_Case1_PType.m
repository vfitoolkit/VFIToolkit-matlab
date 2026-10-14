function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
% Loop order: the ptype setup is done once (pass 0: policy values, grids, parameters, and the conditional restrictions as
% logical masks). Then for each FnsToEvaluate: pass 1 over ptypes evaluates it on the grid; pass 2 over age groups, with the
% ptype loop innermost, computes the per-ptype stats and appends each ptype to one pooled cell, which is pooled across ptypes
% as soon as its ptype loop ends. Memory use is therefore bounded by a small multiple of the agent distribution and does not
% grow with the number of FnsToEvaluate, age groups or conditional restrictions. (The earlier design kept every
% (fn,agegroup) cell for every ptype until the end, and ran out of memory on large models with conditional restrictions.)
% simoptions.lowmemory is accepted but ignored: there is a single code path.
% simoptions.whichcombos selects which (FnsToEvaluate, age group[, conditional restriction][, permanent type or grouped]) combinations are computed; see below.
% simoptions.whichstats may also be given per (FnsToEvaluate, age group[, conditional restriction][, permanent type or grouped]) combination; see below.
%
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
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
% The stationary distribution must be a structure and will contain both the
% weights/distribution across the permanent types, as well as a pdf for the
% stationary distribution of each specific permanent type.
%
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows
% for each type, or as a structure with different fields for each type.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.

% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate; note that this cannot be done for
% vfoptions, simoptions, etc as it then becomes impossible to tell that the vector/matrix is because of PType and not something else)
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

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
    simoptions.lowmemory=0; % accepted but ignored: with the ptype-innermost loop order memory use does not depend on the number of FnsToEvaluate (kept so existing codes that set it still run)
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.alreadygridvals_semiexo=0;
    defaultagegroupings=1;
    if isstruct(N_j)
        N_j_max=0;
        for ii=1:N_i
            simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
            N_j_max=max(N_j_max,N_j.(Names_i{ii}));
        end
    else
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
        N_j_max=N_j;
    end
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
    simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, N_i+1, 7]
    simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    simoptions.groupusingtdigest=0; % if you are ptypestorecpu=1 and groupptypesforstats=1, you might also need to use groupusingtdigest=1 if you get out of memory errors
    simoptions.ntimeperiods=1; % the statistics of the sum/average over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    simoptions.logwhichcombos=0; % the combinations whose statistics are of the log (same shapes as whichcombos without the type dimension, or a scalar)
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.gridinterplayer=0;
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % accepted but ignored (see above)
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
    if ~isfield(simoptions,'agegroupings')
        defaultagegroupings=1;
        if isstruct(N_j)
            N_j_max=0;
            for ii=1:N_i
                simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
                N_j_max=max(N_j_max,N_j.(Names_i{ii}));
            end
        else
            simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
            N_j_max=N_j;
        end
    else
        defaultagegroupings=0;
        N_j_max=length(simoptions.agegroupings);
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
    if ~isfield(simoptions,'agejshifter')
        simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
    end
    if ~isfield(simoptions,'whichstats')
        if ~isstruct(N_j)
            if any(simoptions.agegroupings(2:end)-simoptions.agegroupings(1:end-1)>4)
                % if some agegroupings are 'large', use the slower but lower memory versions
                simoptions.whichstats=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, N_i+1, 7]
            else
                simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, N_i+1, 7]
            end
        else
            for ii=1:N_i
                temp=simoptions.agegroupings.(Names_i{ii});
                if any(temp(2:end)-temp(1:end-1)>4)
                    % if some agegroupings are 'large', use the slower but lower memory versions
                    simoptions.whichstats.(Names_i{ii})=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
                else
                    simoptions.whichstats.(Names_i{ii})=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
                end
            end
        end
    end
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    end
    if ~isfield(simoptions,'groupusingtdigest')
        simoptions.groupusingtdigest=0; % if you are ptypestorecpu=1 and groupptypesforstats=1, you might also need to use groupusingtdigest=1 if you get out of memory errors
    end
    if ~isfield(simoptions,'ntimeperiods')
        simoptions.ntimeperiods=1; % the statistics of the sum/average over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    end
    if ~isfield(simoptions,'ntimeperiodstype')
        simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    end
    if ~isfield(simoptions,'logwhichcombos')
        simoptions.logwhichcombos=0; % the combinations whose statistics are of the log (same shapes as whichcombos without the type dimension, or a scalar)
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
end

%% Setup to allow different N_j (and different agejshifter)
if isstruct(simoptions.agegroupings)
    ngroups=zeros(N_i,1);
    for ii=1:N_i
        ngroups(ii)=length(simoptions.agegroupings.(Names_i{ii}));
    end
else
    ngroups=length(simoptions.agegroupings)*ones(N_i,1);
end
maxngroups=max(ngroups);
if isstruct(simoptions.agejshifter) % if using agejshifter
    tempagejshifter=simoptions.agejshifter;
    simoptions=rmfield(simoptions,'agejshifter');
    simoptions.agejshifter=zeros(N_i,1);
    for ii=1:N_i
        simoptions.agejshifter(ii)=tempagejshifter.(Names_i{ii});
    end
    simoptions.agejshifter=simoptions.agejshifter-min(simoptions.agejshifter); % put them all relative to the minimum
    % Need an alternative version of N_j_max for grouped stats
    if isstruct(N_j)
        N_j_max2=0;
        for ii=1:N_i
            N_j_max2=max(N_j_max2,simoptions.agejshifter(ii)+N_j.(Names_i{ii}));
        end
    end
elseif isscalar(simoptions.agejshifter) % not using agejshifter
    simoptions.agejshifter=zeros(N_i,1);
    N_j_max2=N_j_max;
else % have inputted as a vector
    simoptions.agejshifter=simoptions.agejshifter-min(simoptions.agejshifter); % put them all relative to the minimum
    if isstruct(N_j)
        N_j_max2=0;
        for ii=1:N_i
            N_j_max2=max(N_j_max2,simoptions.agejshifter(ii)+N_j.(Names_i{ii}));
        end
    else
        N_j_max2=N_j_max;
    end
end
% You cannot use agejshifter together with any age grouping other than just every period
if max(simoptions.agejshifter)>0 && defaultagegroupings==0
    error('You cannot use agejshifter together with any age grouping other than the default (each period seperately)')
end

jgroupstr=cell(1,maxngroups);
for jj=1:maxngroups
    if jj<10
        jgroupstr{jj}=['agej00',num2str(jj)];
    elseif jj<100
        jgroupstr{jj}=['agej0',num2str(jj)];
    elseif jj<1000
        jgroupstr{jj}=['agej',num2str(jj)];
    end
end


%%
if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    error('You can only use PType when FnsToEvaluate is a structure')
end


% Preallocate a few things
minvaluevec=nan(numFnsToEvaluate,N_i,maxngroups);
maxvaluevec=nan(numFnsToEvaluate,N_i,maxngroups);
MeanVec=nan(numFnsToEvaluate,N_i,maxngroups);
StdDevVec=nan(numFnsToEvaluate,N_i,maxngroups);
GroupMassVec=zeros(N_i,maxngroups); % mass of each ptype in each age group (its StationaryDist includes its own age weights, which can differ by ptype)
AgeConditionalStats=struct();


FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');

%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(N_i,N_j,length(CondlRestnFnNames));

    if simoptions.groupusingtdigest==1
        error('Have not implemented simoptions.groupusingtdigest==1 together with simoptions.conditionalrestrictions')
    end

end

AgeMasses=zeros(N_i,N_j_max,'gpuArray'); % Only ends up used if using simoptions.conditionalrestrictions
if useCondlRest==1 && isstruct(N_j)
    error('LifeCycleProfiles: Have not implemented combination of doing conditional restrictions with N_j being a structure (differing across agents)')
end

%% simoptions.whichcombos: which (FnsToEvaluate, age group[, conditional restriction]) combinations to compute
% Size [numFnsToEvaluate, number of age groups] without conditional restrictions, and [numFnsToEvaluate, number of age groups,
% 1+number of restrictions] with them: page 1 is the unrestricted stats, pages 2:end the restrictions in the fieldnames order of
% simoptions.conditionalrestrictions. Ones are computed, zeros are skipped; the output has the same fields either way and the
% skipped entries stay NaN. RestrictedSampleMass is always filled. Default is all ones (compute everything). A 2D input when
% there are restrictions is applied to every page. Intended for calibration/estimation, which only needs the targeted combinations.
if useCondlRest==1
    nwhichpages=1+length(CondlRestnFnNames);
else
    nwhichpages=1;
end

%% Multi-period sums/averages (simoptions.ntimeperiods, simoptions.ntimeperiodstype) and logs (simoptions.logwhichcombos)
% As in LifeCycleProfiles_FHorz_Case1: the M>1 outputs are fnname_MperiodAverage (or _MperiodSum) and the logs fnname_log and
% fnname_MperiodAverage_log, each with the grouped stats and one field per ptype, as for fnname. whichcombos and whichstats may
% have length(ntimeperiods)*numFnsToEvaluate rows, in blocks of numFnsToEvaluate in the order of ntimeperiods (numFnsToEvaluate
% rows apply to every M), with the trailing type dimension as usual; logwhichcombos has the shapes of whichcombos without the
% type dimension (it applies to every ptype and to the grouped stats alike), or is a scalar. M>1 needs agegroupings=1:1:N_j,
% the same N_j for every ptype, no agejshifter, and simoptions.pi_z (z transitions; may depend on ptype as pi_z does). The
% per-ptype window statistics are those of LifeCycleProfiles_FHorz_Case1; the grouped ones pool the exact path lists of every
% ptype (each ptype weighted by ptweights times its mass at the start age, or its restricted mass there).
% Internally: the M=1 rows (with an extra function fnname_log, the log of fnname, for each function with a log row) are handed
% to the single-period code below as simoptions.whichcombos (and whichstats), and the window rows are kept for the window section.
ntimeperiods=gather(simoptions.ntimeperiods(:)'); % the order is kept: it is the order of the row blocks of whichcombos
if isempty(ntimeperiods) || ~isnumeric(ntimeperiods) || any(ntimeperiods<1) || any(ntimeperiods~=round(ntimeperiods))
    error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods must be a vector of positive integers (the number of periods summed/averaged over, 1 is the single period)')
end
if length(unique(ntimeperiods))<length(ntimeperiods)
    error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods must not contain repeated values')
end
if ~(ischar(simoptions.ntimeperiodstype) || isstring(simoptions.ntimeperiodstype)) || ~any(strcmp(simoptions.ntimeperiodstype,{'average','sum'}))
    error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiodstype must be ''average'' or ''sum''')
end
nblocks=length(ntimeperiods);
has1=any(ntimeperiods==1);
windowM=ntimeperiods(ntimeperiods>1);
nW=length(windowM);
if strcmp(simoptions.ntimeperiodstype,'average')
    windowstr='periodAverage';
else
    windowstr='periodSum';
end
logwcIn=simoptions.logwhichcombos;
if ~(isnumeric(logwcIn) || islogical(logwcIn)) || any(logwcIn(:)~=0 & logwcIn(:)~=1)
    error('simoptions.logwhichcombos must contain only zeros and ones')
end
uselog=any(logwcIn(:));
numFnsBase=numFnsToEvaluate; % the functions given (the log rows of M=1 are added as extra functions below)
fnlogbase=zeros(1,numFnsToEvaluate); % fnlogbase(ff) is the function an extra function is the log of (0 for the given ones)
skipFields=zeros(numFnsToEvaluate,nwhichpages); % (fn, page) whose fields are removed at the end: nothing of them is computed (only with the new options)
if nW>0
    if isstruct(N_j)
        error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods with M>1 needs the same N_j for every permanent type')
    end
    if any(ntimeperiods>N_j)
        error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods cannot exceed N_j (no window of that many periods fits in the life cycle)')
    end
    if isstruct(simoptions.agejshifter) || any(simoptions.agejshifter~=0)
        error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods with M>1 is not implemented with simoptions.agejshifter')
    end
    if isstruct(simoptions.agegroupings) || ~isequal(reshape(gather(simoptions.agegroupings),1,[]),1:1:N_j)
        error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods with M>1 (multi-period sums/averages) can only be used with simoptions.agegroupings=1:1:N_j (the default); the statistics of W_j are by window start age j')
    end
    if simoptions.groupusingtdigest==1
        error('LifeCycleProfiles_FHorz_Case1_PType: simoptions.ntimeperiods with M>1 is not implemented with simoptions.groupusingtdigest=1')
    end
end
if nblocks>1 || ~has1 || uselog
    nrowswc=nblocks*numFnsBase;
    % whichcombos, every form, to [nrowswc, number of age groups, 1+number of restrictions, N_i+1]
    if ~isfield(simoptions,'whichcombos')
        wcRows=ones(nrowswc,maxngroups,nwhichpages,N_i+1);
    else
        wcRows=simoptions.whichcombos;
        if ~(isnumeric(wcRows) || islogical(wcRows)) || any(wcRows(:)~=0 & wcRows(:)~=1)
            error('simoptions.whichcombos must contain only zeros and ones')
        end
        nrowsin=size(wcRows,1); % numFnsBase, or nrowswc
        if ismatrix(wcRows) && isequal(size(wcRows),[nrowsin,maxngroups]) && (nrowsin==numFnsBase || nrowsin==nrowswc)
            wcRows=repmat(wcRows,[1,1,nwhichpages]); % 2D input: apply to every page
        end
        if ndims(wcRows)<=3 && isequal(size(wcRows,1:3),[nrowsin,maxngroups,nwhichpages]) && (nrowsin==numFnsBase || nrowsin==nrowswc)
            wcRows=repmat(wcRows,[1,1,1,N_i+1]); % no type dimension: apply to every ptype and to the grouped stats
        end
        if nblocks>1 && nrowsin==numFnsBase
            wcRows=repmat(wcRows,[nblocks,1,1,1]); % one selection for every M
        end
        if ~isequal(size(wcRows,1:4),[nrowswc,maxngroups,nwhichpages,N_i+1])
            error(['simoptions.whichcombos must be of size [',num2str(nrowswc),',',num2str(maxngroups),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions), optionally with a trailing type dimension of ',num2str(N_i+1),' (one slot per permanent type in the order of Names_i, then the grouped stats), or have ',num2str(numFnsBase),' rows to apply to every M'])
        end
        wcRows=double(wcRows);
    end
    % whichstats: a 1x7 vector or a structure of them by ptype stays as it is; an array, every form, to [nrowswc, number of age groups, 1+number of restrictions, N_i+1, 7]
    wsIn=simoptions.whichstats;
    wsIsArray=~isstruct(wsIn) && ~(isvector(wsIn) && numel(wsIn)==7);
    if wsIsArray
        nrowsin=size(wsIn,1);
        if ~(nrowsin==numFnsBase || nrowsin==nrowswc)
            error(['simoptions.whichstats must be a 1x7 vector, or have ',num2str(nrowswc),' rows (length of simoptions.ntimeperiods times the number of FnsToEvaluate) or ',num2str(numFnsBase),' rows (applied to every M)'])
        end
        if ndims(wsIn)==3 && isequal(size(wsIn),[nrowsin,maxngroups,7])
            wsRows=repmat(reshape(wsIn,[nrowsin,maxngroups,1,1,7]),[1,1,nwhichpages,N_i+1,1]);
        elseif ndims(wsIn)<=4 && isequal(size(wsIn,1:4),[nrowsin,maxngroups,nwhichpages,7])
            wsRows=repmat(reshape(wsIn,[nrowsin,maxngroups,nwhichpages,1,7]),[1,1,1,N_i+1,1]);
        elseif isequal(size(wsIn,1:5),[nrowsin,maxngroups,nwhichpages,N_i+1,7])
            wsRows=wsIn;
        else
            error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(nrowswc),',',num2str(maxngroups),',',num2str(nwhichpages),',7], or [',num2str(nrowswc),',',num2str(maxngroups),',7], or with a trailing type dimension [',num2str(nrowswc),',',num2str(maxngroups),',',num2str(nwhichpages),',',num2str(N_i+1),',7], or any of these with ',num2str(numFnsBase),' rows to apply to every M'])
        end
        if nblocks>1 && nrowsin==numFnsBase
            wsRows=repmat(wsRows,[nblocks,1,1,1,1]); % one whichstats for every M
        end
    end
    % logwhichcombos: scalar, [rows, number of age groups] or [rows, number of age groups, 1+number of restrictions] (no type dimension)
    if isscalar(logwcIn)
        logRows=logwcIn*ones(nrowswc,maxngroups,nwhichpages);
    else
        logRows=logwcIn;
        nrowsin=size(logRows,1);
        if ismatrix(logRows) && nwhichpages>1 && isequal(size(logRows),[nrowsin,maxngroups]) && (nrowsin==numFnsBase || nrowsin==nrowswc)
            logRows=repmat(logRows,[1,1,nwhichpages]); % 2D input: apply to every page
        end
        if nblocks>1 && nrowsin==numFnsBase
            logRows=repmat(logRows,[nblocks,1,1]); % one selection for every M
        end
        if ~isequal(size(logRows),[nrowswc,maxngroups,nwhichpages]) && ~(nwhichpages==1 && isequal(size(logRows),[nrowswc,maxngroups]))
            error(['simoptions.logwhichcombos must be a scalar 0 or 1, or of size [',num2str(nrowswc),',',num2str(maxngroups),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions; it has no type dimension, it applies to every ptype and to the grouped stats), or with ',num2str(numFnsBase),' rows to apply to every M, or the same 2-D form as whichcombos'])
        end
        logRows=double(logRows);
    end
    % The M=1 block (to the single-period code) and the window blocks
    if has1
        bb=find(ntimeperiods==1);
        wc1=wcRows((bb-1)*numFnsBase+1:bb*numFnsBase,:,:,:);
        logwc1=logRows((bb-1)*numFnsBase+1:bb*numFnsBase,:,:);
        if wsIsArray
            ws1=wsRows((bb-1)*numFnsBase+1:bb*numFnsBase,:,:,:,:);
        end
    else
        wc1=zeros(numFnsBase,maxngroups,nwhichpages,N_i+1);
        logwc1=zeros(numFnsBase,maxngroups,nwhichpages);
        if wsIsArray
            ws1=zeros(numFnsBase,maxngroups,nwhichpages,N_i+1,7);
        end
    end
    wcLog1=wc1.*logwc1; % the M=1 combinations of the log (every ptype slot and the grouped one)
    wc1=wc1.*(1-logwc1);
    if nW>0
        wcW=zeros(numFnsBase,N_j,nwhichpages,N_i+1,nW); % the window combinations of the level
        wcLogW=zeros(numFnsBase,N_j,nwhichpages,N_i+1,nW); % and of the log
        wsW=zeros(numFnsBase,N_j,nwhichpages,N_i+1,7,nW); % their whichstats
        for lw=1:nW
            bb=find(ntimeperiods==windowM(lw));
            wcblock=wcRows((bb-1)*numFnsBase+1:bb*numFnsBase,:,:,:);
            logblock=logRows((bb-1)*numFnsBase+1:bb*numFnsBase,:,:);
            wcLogW(:,:,:,:,lw)=wcblock.*logblock;
            wcW(:,:,:,:,lw)=wcblock.*(1-logblock);
            if wsIsArray
                wsW(:,:,:,:,:,lw)=wsRows((bb-1)*numFnsBase+1:bb*numFnsBase,:,:,:,:);
            elseif isstruct(wsIn)
                for ii=1:N_i
                    wsW(:,:,:,ii,:,lw)=repmat(reshape(wsIn.(Names_i{ii}),[1,1,1,1,7]),[numFnsBase,N_j,nwhichpages,1,1]);
                end
                wsW(:,:,:,N_i+1,:,lw)=repmat(reshape(wsIn.(Names_i{1}),[1,1,1,1,7]),[numFnsBase,N_j,nwhichpages,1,1]); % the grouped stats use the first ptype's whichstats (as for M=1)
            else
                wsW(:,:,:,:,:,lw)=repmat(reshape(wsIn,[1,1,1,1,7]),[numFnsBase,N_j,nwhichpages,N_i+1,1]);
            end
            anyws=any(wsW(:,:,:,:,:,lw),5);
            wcW(:,:,:,:,lw)=wcW(:,:,:,:,lw).*anyws; % a combination with no statistic requested is skipped altogether
            wcLogW(:,:,:,:,lw)=wcLogW(:,:,:,:,lw).*anyws;
            wcW(:,N_j-windowM(lw)+2:N_j,:,:,lw)=0; % no window of M periods starts after age N_j-M+1
            wcLogW(:,N_j-windowM(lw)+2:N_j,:,:,lw)=0;
            if simoptions.groupptypesforstats==0
                wcW(:,:,:,N_i+1,lw)=0; % no grouped stats
                wcLogW(:,:,:,N_i+1,lw)=0;
            end
        end
    end
    % The M=1 log rows are extra functions (fnname_log) whose values are the log of the function's values
    for ff=1:numFnsBase
        if any(wcLog1(ff,:,:,:),'all')
            logname=[FnsToEvalNames{ff},'_log'];
            if isfield(FnsToEvaluate,logname)
                error(['LifeCycleProfiles_FHorz_Case1_PType: simoptions.logwhichcombos makes an output ',logname,', but FnsToEvaluate already has a function of that name'])
            end
            FnsToEvaluate.(logname)=FnsToEvaluate.(FnsToEvalNames{ff});
            wc1(end+1,:,:,:)=wcLog1(ff,:,:,:); %#ok<AGROW>
            if wsIsArray
                ws1(end+1,:,:,:,:)=ws1(ff,:,:,:,:); %#ok<AGROW>
            end
            fnlogbase(end+1)=ff; %#ok<AGROW>
        end
    end
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
    % The (fn, page) with nothing computed have no fields (removed at the end): the given functions when M=1 is not wanted or all
    % of their M=1 selections on the page are of the log, and the log functions where they are not selected
    skipFields=zeros(numFnsToEvaluate,nwhichpages);
    for ff=1:numFnsToEvaluate
        for pp=1:nwhichpages
            if fnlogbase(ff)==0
                skipFields(ff,pp)=(~has1) || (any(wcLog1(ff,:,pp,:),'all') && ~any(wc1(ff,:,pp,:),'all'));
            else
                skipFields(ff,pp)=~any(wc1(ff,:,pp,:),'all');
            end
        end
    end
    simoptions.whichcombos=wc1;
    if wsIsArray
        simoptions.whichstats=ws1;
    end
    % (re)preallocate for the extra functions
    minvaluevec=nan(numFnsToEvaluate,N_i,maxngroups);
    maxvaluevec=nan(numFnsToEvaluate,N_i,maxngroups);
    MeanVec=nan(numFnsToEvaluate,N_i,maxngroups);
    StdDevVec=nan(numFnsToEvaluate,N_i,maxngroups);
    FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');
end

% A trailing type dimension may be added (2026-10-07): [.., N_i+1] selects per permanent type, in the order of Names_i, with the
% last slot the grouped stats. The shapes without it apply to every ptype and to the grouped stats alike. The grouped Mean,
% StdDeviation, Minimum and Maximum are built from every ptype's, so a grouped slot that is on also computes (and reports) that
% combination for every ptype. simoptions.whichstats takes the same trailing dimension ([.., N_i+1, 7]); see below.
if ~isfield(simoptions,'whichcombos')
    whichcombosAll=ones(numFnsToEvaluate,maxngroups,nwhichpages,N_i+1); % whichcombos here is [nFns, nAgeGroups, 1+nRestr, N_i+1]
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && isequal(size(whichcombos),[numFnsToEvaluate,maxngroups])
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % 2D input: apply to every page
    end
    if ndims(whichcombos)<=3 && isequal(size(whichcombos,1:3),[numFnsToEvaluate,maxngroups,nwhichpages])
        whichcombos=repmat(whichcombos,[1,1,1,N_i+1]); % no type dimension: apply to every ptype and to the grouped stats
    end
    if ~isequal(size(whichcombos,1:4),[numFnsToEvaluate,maxngroups,nwhichpages,N_i+1])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(maxngroups),',',num2str(nwhichpages),'] (number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions), optionally with a trailing type dimension of ',num2str(N_i+1),' (one slot per permanent type in the order of Names_i, then the grouped stats)'])
    end
    whichcombosAll=double(whichcombos);
end
whichcombosG=whichcombosAll(:,:,:,N_i+1); % whichcombosG(ff,jj,1) is the grouped unrestricted stats, whichcombosG(ff,jj,1+rr) the grouped stats of restriction rr; whichcombosAll(:,:,:,ii) is ptype ii

%% simoptions.whichstats, per combination
% whichstats is either the usual 1x7 vector (or a structure with a vector per ptype), or an array of size
% [numFnsToEvaluate, number of age groups, 1+number of restrictions, 7] giving a whichstats vector for every (fn, age group, page)
% combination (same pages as whichcombos). [numFnsToEvaluate, number of age groups, 7] is also accepted: without restrictions it is
% the same thing, with restrictions it is applied to every page. Entries keep their meaning from StatsFromWeightedGrid. An output
% field exists for a (fn, page[, ptype]) if any of its age groups asks for that statistic; the age groups that do not stay NaN.
% A combination that asks for no statistic at all is skipped like a whichcombos zero.
if isstruct(simoptions.whichstats)
    wsG=simoptions.whichstats.(Names_i{1}); % the grouped stats use the first ptype's whichstats (as before)
else
    wsG=simoptions.whichstats;
end
whichstatsAll=[]; % set when whichstats carries the trailing type dimension (then each ptype takes its own slot in pass 0)
if isvector(wsG) && numel(wsG)==7
    whichstatsG=repmat(reshape(wsG,[1,1,1,7]),[numFnsToEvaluate,maxngroups,nwhichpages,1]);
elseif ndims(wsG)==3 && isequal(size(wsG),[numFnsToEvaluate,maxngroups,7])
    whichstatsG=repmat(reshape(wsG,[numFnsToEvaluate,maxngroups,1,7]),[1,1,nwhichpages,1]);
elseif ndims(wsG)==4 && isequal(size(wsG,1:4),[numFnsToEvaluate,maxngroups,nwhichpages,7])
    whichstatsG=wsG;
elseif ndims(wsG)==5 && isequal(size(wsG,1:5),[numFnsToEvaluate,maxngroups,nwhichpages,N_i+1,7])
    whichstatsAll=wsG;
    whichstatsG=reshape(wsG(:,:,:,N_i+1,:),[numFnsToEvaluate,maxngroups,nwhichpages,7]); % the last slot is the grouped stats
else
    error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(numFnsToEvaluate),',',num2str(maxngroups),',',num2str(nwhichpages),',7] (number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions, 7), or [',num2str(numFnsToEvaluate),',',num2str(maxngroups),',7], or with a trailing type dimension [',num2str(numFnsToEvaluate),',',num2str(maxngroups),',',num2str(nwhichpages),',',num2str(N_i+1),',7] (one slot per permanent type in the order of Names_i, then the grouped stats)'])
end
whichcombosG=whichcombosG.*any(whichstatsG,4); % a grouped combination with no statistic requested is skipped altogether

if useCondlRest==1
    % Preallocate various things for the stats (as many will have jj as a dimension)
    % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
    for ff=1:numFnsToEvaluate
        for rr=1:length(CondlRestnFnNames)
            % whichstats for this (fn, restriction) across its age groups: a statistic is preallocated if any age group asks for it (the Lorenz curve if any age group has 1<=whichstats(4)<3)
            wsP=max(reshape(whichstatsG(ff,:,1+rr,:),[maxngroups,7]),[],1);
            ws4=reshape(whichstatsG(ff,:,1+rr,4),[1,maxngroups]);
            if any(ws4>=1 & ws4<3)
                wsP(4)=1;
            end
            if wsP(1)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(1)==1 && wsP(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(3)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(4)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
                if wsP(4)<3
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
                end
            end
            if wsP(5)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(6)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(7)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
        end
    end
end

%% Pass 0: per-type setup, done once and kept for the ff loop
PT=struct();
RestrictionMask=cell(N_i,1); % becomes cell(N_i,nRestr) on first use
for ii=1:N_i
        iistr=Names_i{ii};

        % First set up simoptions
        simoptions_temp=PType_Options(simoptions,iistr); % Note: already check for existence of simoptions and created it if it was not inputted

        if simoptions_temp.verbose==1
            fprintf('Permanent type: %i of %i \n',ii, N_i)
        end
        if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
            PolicyIndexes_temp=gpuArray(Policy.(iistr)); % Essentially just assuming simoptions.ptypestorecpu=1 as well
            % StationaryDist_temp=gpuArray(StationaryDist.(iistr));
        else
            PolicyIndexes_temp=Policy.(iistr);
            % StationaryDist_temp=StationaryDist.(iistr);
        end

        %% Go through everything which might be dependent on fixed type (PType)
        [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

        if isstruct(N_j)
            N_j_temp=N_j.(iistr);
        else
            N_j_temp=N_j;
        end

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

        [~,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
        FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

        StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

        AgeMasses(ii,simoptions.agejshifter(ii)+(1:N_j_temp))=sum(StationaryDist_ii,1); % I think this is right, but haven't tested yet

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
            RestrictionValues=reshape(RestrictionValues,[N_a_temp*N_z_temp,N_j_temp]);
            restrictedsamplemass(ii,:,rr)=sum(StationaryDist_ii.*RestrictionValues,1); % mass within ptype ii at each age that satisfies the restriction (already includes the age weights)
            if simoptions.ptypestorecpu==1
                RestrictionMask{ii,rr}=gather(RestrictionValues);
            else
                RestrictionMask{ii,rr}=RestrictionValues;
            end

            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.(iistr)=restrictedsamplemass(ii,:,rr); % Seems likely this would be something user might want
            if all(restrictedsamplemass(ii,:,rr)==0) && simoptions.warnzerorestrictedmass==2
                warning(['One of the conditional restrictions evaluates to a zero mass (at all j) for permanent type ',iistr])
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass for this permanent type \n'])
            end
        end
    end

    % Keep the per-type setup for the ff loop
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
    PT(ii).FnsAndPTypeIndicator_ii=FnsAndPTypeIndicator_ii;
    % whichstats for this ptype, per (fn, age group, page) combination; see the whichstats section above
    if ~isempty(whichstatsAll) % whichstats with the trailing type dimension: this ptype's slot
        PT(ii).whichstatsArr=reshape(whichstatsAll(:,:,:,ii,:),[numFnsToEvaluate,maxngroups,nwhichpages,7]);
    else
        wsT=simoptions_temp.whichstats;
        if isvector(wsT) && numel(wsT)==7
            PT(ii).whichstatsArr=repmat(reshape(wsT,[1,1,1,7]),[numFnsToEvaluate,maxngroups,nwhichpages,1]);
        elseif ndims(wsT)==3 && isequal(size(wsT),[numFnsToEvaluate,maxngroups,7])
            PT(ii).whichstatsArr=repmat(reshape(wsT,[numFnsToEvaluate,maxngroups,1,7]),[1,1,nwhichpages,1]);
        elseif ndims(wsT)==4 && isequal(size(wsT,1:4),[numFnsToEvaluate,maxngroups,nwhichpages,7])
            PT(ii).whichstatsArr=wsT;
        else
            error(['simoptions.whichstats for ptype ',iistr,' must be a 1x7 vector, or of size [',num2str(numFnsToEvaluate),',',num2str(maxngroups),',',num2str(nwhichpages),',7], or [',num2str(numFnsToEvaluate),',',num2str(maxngroups),',7] (see the whichstats section)'])
        end
    end
    % This ptype's effective selection: its own whichcombos slot, plus whatever the grouped stats need. The grouped Mean, StdDeviation,
    % Minimum and Maximum are built from every ptype's, so a grouped slot that is on forces this ptype's computation of that
    % combination with at least the grouped whichstats (the result is reported, as a byproduct).
    wsForced=whichstatsG.*repmat(whichcombosG,[1,1,1,7]); % the grouped whichstats where the grouped slot is on
    wsForced(:,:,:,1)=max(wsForced(:,:,:,1),wsForced(:,:,:,3)); % the grouped StdDeviation is rebuilt below from every ptype's Mean and StdDeviation, so a grouped StdDeviation needs the ptype Means too (before 2026-10-07 MeanVec stayed NaN and so did the grouped StdDeviation when only it was asked for)
    lor=(PT(ii).whichstatsArr(:,:,:,4)==1 | PT(ii).whichstatsArr(:,:,:,4)==2 | wsForced(:,:,:,4)==1 | wsForced(:,:,:,4)==2);
    PT(ii).whichstatsArr=max(PT(ii).whichstatsArr,wsForced);
    ws4=PT(ii).whichstatsArr(:,:,:,4);
    ws4(lor)=2; % Gini/Lorenz codes: a Lorenz-giving 1 or 2 on either side wins over the Gini-only 3
    PT(ii).whichstatsArr(:,:,:,4)=ws4;
    PT(ii).whichcombos=max(whichcombosAll(:,:,:,ii),whichcombosG).*any(PT(ii).whichstatsArr,4); % a combination asking for no statistic is skipped
    if simoptions.ptypestorecpu==1
        PT(ii).PolicyValuesPermute_temp=gather(PolicyValuesPermute_temp);
    else
        PT(ii).PolicyValuesPermute_temp=PolicyValuesPermute_temp;
    end
end % end ii over N_i (pass 0)
clear PolicyValues_temp PolicyValuesPermute_temp PolicyIndexes_temp RestrictionValues StationaryDist_ii

    %% Now we compute the grouped stats
    if simoptions_temp.verbose==1
        fprintf('Permanent type: Grouped Stats \n')
    end
    % Preallocate various things for the stats (as many will have jj as a dimension)
    % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
    % If we can put these together, it must be the case that whichstats is same for all permanent types
    if isstruct(simoptions.whichstats)
        simoptions.whichstats=simoptions.whichstats.(Names_i{1}); % just use the first one
    end
    if simoptions.groupptypesforstats==1 % These grouped top-level fields are only ever filled when grouping (see below); if we don't group, skip preallocating them so they are not left as NaN placeholders next to the per-ptype fields
        for ff=1:numFnsToEvaluate
            % whichstats for this fn across its age groups: a statistic is preallocated if any age group asks for it (the Lorenz curve if any age group has 1<=whichstats(4)<3)
            wsP=max(reshape(whichstatsG(ff,:,1,:),[maxngroups,7]),[],1);
            ws4=reshape(whichstatsG(ff,:,1,4),[1,maxngroups]);
            if any(ws4>=1 & ws4<3)
                wsP(4)=1;
            end
            if wsP(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,N_j_max2,'gpuArray'); % Note: N_j_max2=length(simoptions.agegroupings) in basic setup, will be different when N_j or agejshifter varies by PType
            end
            if wsP(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,N_j_max2,'gpuArray');
                if wsP(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,N_j_max2,'gpuArray');
                end
            end
            if wsP(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,N_j_max2,'gpuArray');
            end
            if wsP(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,N_j_max2,'gpuArray');
                if wsP(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,N_j_max2,'gpuArray');
                end
            end
            if wsP(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,N_j_max2,'gpuArray');
            end
            if wsP(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,N_j_max2,'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,N_j_max2,'gpuArray');
            end
            if wsP(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,N_j_max2,'gpuArray');
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,N_j_max2,'gpuArray');
            end
            if useCondlRest==1
                for rr=1:length(CondlRestnFnNames)
                    wsP=max(reshape(whichstatsG(ff,:,1+rr,:),[maxngroups,7]),[],1); % as above, for this restriction
                    ws4=reshape(whichstatsG(ff,:,1+rr,4),[1,maxngroups]);
                    if any(ws4>=1 & ws4<3)
                        wsP(4)=1;
                    end
                    if wsP(1)==1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean=nan(1,N_j_max2,'gpuArray');
                    end
                    if wsP(2)==1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median=nan(1,N_j_max2,'gpuArray');
                        if wsP(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,N_j_max2,'gpuArray');
                        end
                    end
                    if wsP(3)==1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation=nan(1,N_j_max2,'gpuArray');
                    end
                    if wsP(4)>=1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini=nan(1,N_j_max2,'gpuArray');
                        if wsP(4)<3
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,N_j_max2,'gpuArray');
                        end
                    end
                    if wsP(5)==1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum=nan(1,N_j_max2,'gpuArray');
                    end
                    if wsP(6)>=1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,N_j_max2,'gpuArray'); % Includes the min and max values
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,N_j_max2,'gpuArray');
                    end
                    if wsP(7)==1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,N_j_max2,'gpuArray');
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,N_j_max2,'gpuArray');
                    end
                end
            end
        end
    end

if simoptions.groupusingtdigest==1
    delta=10000;
end

%% Main loop: ff outer; pass 1 over ptypes evaluates ff; pass 2 over age groups with ptypes innermost pools each (ff,agegroup) cell at once
for ff=1:numFnsToEvaluate

    %% Pass 1: evaluate ff on the grid for every ptype (kept for all ptypes, this ff only)
    ValuesOnGrid_ff=cell(N_i,1);
    for ii=1:N_i
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
        FnsAndPTypeIndicator_ii=PT(ii).FnsAndPTypeIndicator_ii;
        whichstatsArr_ii=PT(ii).whichstatsArr;
        if simoptions.ptypestorecpu==1
            PolicyValuesPermute_temp=gpuArray(PT(ii).PolicyValuesPermute_temp);
        else
            PolicyValuesPermute_temp=PT(ii).PolicyValuesPermute_temp;
        end

        if FnsAndPTypeIndicator_ii(ff)==1 % If this function is relevant to this ptype
            if any(PT(ii).whichcombos(ff,:,:),'all') % skip the evaluation if no combination of this function is wanted for this ptype (its own slot or the grouped one) (its output fields are still preallocated below, and stay NaN)
                % Get parameter names for current FnsToEvaluate functions
                if isstruct(FnsToEvaluate.(FnsToEvalNames{ff}))
                    tempfn=FnsToEvaluate.(FnsToEvalNames{ff}).(iistr);
                else
                    tempfn=FnsToEvaluate.(FnsToEvalNames{ff});
                end
                tempnames=getAnonymousFnInputNames(tempfn);
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
                simoptions_temp.keepoutputasmatrix=2;
                ValuesOnGrid_ffii=EvalFnOnAgentDist_Grid_J(tempfn,CellOverAgeOfParamValues,PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,n_z_temp,a_gridvals_temp,z_gridvals_J_temp);

                ValuesOnGrid_ffii=reshape(ValuesOnGrid_ffii,[N_a_temp*N_z_temp,N_j_temp]);
                if fnlogbase(ff)>0 % a log row (simoptions.logwhichcombos): the statistics are of the log of the function
                    % it must be positive on every point with positive mass in the populations this ptype is used in (its own slot, and the grouped one)
                    StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp,N_j_temp]);
                    usedmass_ii=false(N_a_temp*N_z_temp,N_j_temp,'gpuArray');
                    agegr_ii=simoptions_temp.agegroupings;
                    for kk=1:length(agegr_ii)
                        j1=agegr_ii(kk);
                        if kk<length(agegr_ii)
                            jend=agegr_ii(kk+1)-1;
                        else
                            jend=N_j_temp;
                        end
                        kks=kk+simoptions.agejshifter(ii);
                        if PT(ii).whichcombos(ff,kks,1)==1
                            usedmass_ii(:,j1:jend)=usedmass_ii(:,j1:jend) | gpuArray(StationaryDist_ii(:,j1:jend)>0);
                        end
                        if useCondlRest==1
                            for rr=1:length(CondlRestnFnNames)
                                if PT(ii).whichcombos(ff,kks,1+rr)==1
                                    usedmass_ii(:,j1:jend)=usedmass_ii(:,j1:jend) | gpuArray(StationaryDist_ii(:,j1:jend).*RestrictionMask{ii,rr}(:,j1:jend)>0);
                                end
                            end
                        end
                    end
                    if any(ValuesOnGrid_ffii(usedmass_ii)<=0)
                        error(['LifeCycleProfiles_FHorz_Case1_PType: simoptions.logwhichcombos asks for the log of ',FnsToEvalNames{fnlogbase(ff)},', but for permanent type ',iistr,' it is not positive on some points with positive mass (use a conditional restriction, or a floor in the FnsToEvaluate)'])
                    end
                    ValuesOnGrid_ffii(ValuesOnGrid_ffii<=0)=1; % points with no mass in any population used (their weight is zero), log(1)=0 keeps them finite
                    ValuesOnGrid_ffii=log(ValuesOnGrid_ffii);
                end
            else
                ValuesOnGrid_ffii=[];
            end
                % StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

                % Note, eliminating zero weights and unique() cannot be done yet as they need to be conditional on j
                % (otherwise lose the j dimension if I just apply them now)

                % Preallocate various things for the stats (as many will have jj as a dimension)
                % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).
                % whichstats of this ptype for this fn across its age groups: a statistic is preallocated if any age group asks for it (the Lorenz curve if any age group has 1<=whichstats(4)<3)
                wsP=max(reshape(whichstatsArr_ii(ff,:,1,:),[maxngroups,7]),[],1);
                ws4=reshape(whichstatsArr_ii(ff,:,1,4),[1,maxngroups]);
                if any(ws4>=1 & ws4<3)
                    wsP(4)=1;
                end
                if wsP(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Mean=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if wsP(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Median=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    if wsP(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).RatioMeanToMedian=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    end
                end
                if wsP(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Variance=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).StdDeviation=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if wsP(4)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Gini=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    if wsP(4)<3
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).LorenzCurve=nan(simoptions_temp.npoints,length(simoptions_temp.agegroupings),'gpuArray');
                    end
                end
                if wsP(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Minimum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Maximum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if wsP(6)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,length(simoptions_temp.agegroupings),'gpuArray'); % Includes the min and max values
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).QuantileMeans=nan(simoptions_temp.nquantiles,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if wsP(7)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Top1share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Top5share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Top10share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Bottom50share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile50th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile90th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile95th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile99th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if useCondlRest==1
                    for rr=1:length(CondlRestnFnNames)
                        wsP=max(reshape(whichstatsArr_ii(ff,:,1+rr,:),[maxngroups,7]),[],1); % as above, for this restriction
                        ws4=reshape(whichstatsArr_ii(ff,:,1+rr,4),[1,maxngroups]);
                        if any(ws4>=1 & ws4<3)
                            wsP(4)=1;
                        end
                        if wsP(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Mean=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if wsP(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Median=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            if wsP(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).RatioMeanToMedian=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            end
                        end
                        if wsP(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Variance=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).StdDeviation=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if wsP(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Gini=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            if wsP(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).LorenzCurve=nan(simoptions_temp.npoints,length(simoptions_temp.agegroupings),'gpuArray');
                            end
                        end
                        if wsP(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Minimum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Maximum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if wsP(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,length(simoptions_temp.agegroupings),'gpuArray'); % Includes the min and max values
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).QuantileMeans=nan(simoptions_temp.nquantiles,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if wsP(7)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Top1share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Top5share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Top10share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Bottom50share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile50th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile90th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile95th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile99th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                    end
                end

            if simoptions.ptypestorecpu==1
                ValuesOnGrid_ff{ii}=gather(ValuesOnGrid_ffii);
            else
                ValuesOnGrid_ff{ii}=ValuesOnGrid_ffii;
            end
        end
    end % end ii over N_i (pass 1)
    clear ValuesOnGrid_ffii PolicyValuesPermute_temp

    %% Pass 2: age groups, ptypes innermost; pool each (ff,agegroup) cell as soon as its ptype loop ends
    for jjs=1:maxngroups % jjs is the (agejshifter-shifted) age group index used for the grouped stats
        if ~any(whichcombosAll(ff,jjs,:,:),'all') % no combination of this (function, age group) is wanted for any ptype or for the grouped stats
            continue
        end
        % The pooled cell for this (ff,jjs)
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
            iistr=Names_i{ii};
            simoptions_temp=PT(ii).simoptions_temp;
            N_j_temp=PT(ii).N_j_temp;
            N_a_temp=PT(ii).N_a_temp;
            N_z_temp=PT(ii).N_z_temp;
            FnsAndPTypeIndicator_ii=PT(ii).FnsAndPTypeIndicator_ii;
            whichstatsArr_ii=PT(ii).whichstatsArr;
            whichcombos_ii=PT(ii).whichcombos; % this ptype's selection: its own slot, plus what the grouped stats need

            if FnsAndPTypeIndicator_ii(ff)==1 % If this function is relevant to this ptype
                jj=jjs-simoptions.agejshifter(ii); % this ptype's own age group index
                if jj>=1 && jj<=length(simoptions_temp.agegroupings) && any(whichcombos_ii(ff,jjs,:)) % this ptype has age group jjs, and some stat of it is wanted for this ptype (its own slot, or forced by the grouped slot; otherwise its values were not even evaluated in pass 1)
                    StationaryDist_ii=reshape(StationaryDist.(iistr),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

                    j1=simoptions_temp.agegroupings(jj);
                    if jj<length(simoptions_temp.agegroupings)
                        jend=simoptions_temp.agegroupings(jj+1)-1;
                    else
                        jend=N_j_temp;
                    end
                    % Where we store them depends on
                    jjageshifted=jj+simoptions.agejshifter(ii);

                    % Calculate the individual stats
                    StationaryDistVec_jj=reshape(StationaryDist_ii(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]);
                    if simoptions.ptypestorecpu==1
                        Values_jj=gpuArray(reshape(ValuesOnGrid_ff{ii}(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]));
                    else
                        Values_jj=reshape(ValuesOnGrid_ff{ii}(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]);
                    end

                    % Eliminate all the zero-weighted points (this doesn't really save runtime for the exact calculation and often can increase it, but
                    % for the createDigest it slashes the runtime. So since we want it then we may as well do it now.)
                    temp=logical(StationaryDistVec_jj==0); % NOTE: This and the next line could in principle be done outside all of these loops (just looping over j)
                    StationaryDistVec_jj=StationaryDistVec_jj(~temp);
                    Values_jj=Values_jj(~temp);

                    % I want to use unique to make it easier to put the different agent
                    % ptypes together (as all the matrices are typically smaller).
                    % May as well do it before doing the StatsFromWeightedGrid
                    [SortedValues_jj,~,sortindex]=unique(Values_jj);
                    SortedWeights_jj=accumarray(sortindex,StationaryDistVec_jj,[],@sum);

                    groupmass_jj=sum(SortedWeights_jj(:)); % mass of this ptype in this age group
                    GroupMassVec(ii,jjageshifted)=gather(groupmass_jj);
                    SortedWeights_jj=SortedWeights_jj/groupmass_jj; % Normalize conditional on jj (is later reweighted by ptweights(ii)*groupmass_jj for groupstats)

                    %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and ff)
                    if whichcombos_ii(ff,jjs,1)==1 % the unrestricted stats of this (function, age group) are wanted for this ptype (its own slot, or forced by the grouped slot)
                    ws=reshape(whichstatsArr_ii(ff,jjs,1,:),[1,7]); % whichstats of this ptype for this (fn, age group)
                    tempStats=StatsFromWeightedGrid(SortedValues_jj,SortedWeights_jj,simoptions_temp.npoints,simoptions_temp.nquantiles,simoptions_temp.tolerance,1,ws); % 1 is presorted

                    % Now store these based on jj
                    if ws(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Mean(jj)=tempStats.Mean;
                    end
                    if ws(2)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Median(jj)=tempStats.Median;
                        if ws(1)==1
                            AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).RatioMeanToMedian(jj)=tempStats.RatioMeanToMedian;
                        end
                    end
                    if ws(3)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Variance(jj)=tempStats.Variance;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).StdDeviation(jj)=tempStats.StdDeviation;
                    end
                    if ws(4)>=1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Gini(jj)=tempStats.Gini;
                        if ws(4)<3
                            AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).LorenzCurve(:,jj)=tempStats.LorenzCurve;
                        end
                    end
                    if ws(5)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Minimum(jj)=tempStats.Minimum;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).Maximum(jj)=tempStats.Maximum;
                    end
                    if ws(6)>=1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).QuantileCutoffs(:,jj)=tempStats.QuantileCutoffs;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).QuantileMeans(:,jj)=tempStats.QuantileMeans;
                    end
                    if ws(7)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Top1share(jj)=tempStats.MoreInequality.Top1share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Top5share(jj)=tempStats.MoreInequality.Top5share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Top10share(jj)=tempStats.MoreInequality.Top10share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Bottom50share(jj)=tempStats.MoreInequality.Bottom50share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile50th(jj)=tempStats.MoreInequality.Percentile50th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile90th(jj)=tempStats.MoreInequality.Percentile90th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile95th(jj)=tempStats.MoreInequality.Percentile95th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile99th(jj)=tempStats.MoreInequality.Percentile99th;
                    end

                    % For later, put the mean and std dev in a convenient place. These are instead done on jjageshifted (so they can be grouped across ptypes later)
                    if ws(1)==1
                        MeanVec(ff,ii,jjageshifted)=tempStats.Mean;
                    end
                    if ws(3)==1
                        StdDevVec(ff,ii,jjageshifted)=tempStats.StdDeviation;
                    end
                    % Do the same with the minimum and maximum
                    if ws(5)==1 && StationaryDist.ptweights(ii)>0 % a ptype of zero mass is not in the population, so it must not set the grouped min/max (its entries stay NaN, which min()/max() ignore)
                        minvaluevec(ff,ii,jjageshifted)=tempStats.Minimum;
                        maxvaluevec(ff,ii,jjageshifted)=tempStats.Maximum;
                    end
                    end % whichcombos_ii(ff,jjs,1)

                    if simoptions.groupptypesforstats==1 && any(whichcombosG(ff,jjs,:)) % (some grouped stat of this (function, age group) is wanted)
                        % Append this ptype to the pooled cell
                        if simoptions.groupusingtdigest==1
                            [C_jj,digestweights_jj,~]=createDigest(SortedValues_jj, SortedWeights_jj,delta,1); % 1 is presorted
                            Cmerge(merge_nsofar+1:merge_nsofar+length(C_jj))=C_jj;
                            digestweightsmerge(merge_nsofar+1:merge_nsofar+length(C_jj))=digestweights_jj*StationaryDist.ptweights(ii)*groupmass_jj; % weight of the ptype in the age group: ptweights times its mass there (the ptypes can have different age weights)
                            merge_nsofar=merge_nsofar+length(C_jj);
                        elseif simoptions.ptypestorecpu==1
                            PoolValues=[PoolValues; gather(SortedValues_jj)];
                            PoolWeights=[PoolWeights; gather(SortedWeights_jj)*gather(StationaryDist.ptweights(ii)*groupmass_jj)]; % weight of the ptype in the age group: ptweights times its mass there (the ptypes can have different age weights)
                        else
                            PoolValues=[PoolValues; SortedValues_jj];
                            PoolWeights=[PoolWeights; SortedWeights_jj*StationaryDist.ptweights(ii)*groupmass_jj]; % weight of the ptype in the age group: ptweights times its mass there (the ptypes can have different age weights)
                        end
                    end

                    %% If using conditional restrictions, do those (the restricted weights come from the stored mask)
                    if useCondlRest==1
                        for rr=1:length(CondlRestnFnNames)
                            if whichcombos_ii(ff,jjs,1+rr)==1 % this restriction is wanted for this (function, age group) for this ptype (its own slot, or forced by the grouped slot)
                            ws=reshape(whichstatsArr_ii(ff,jjs,1+rr,:),[1,7]); % whichstats of this ptype for this (fn, age group, restriction)
                            if sum(restrictedsamplemass(ii,j1:jend,rr))~=0
                                if simoptions.ptypestorecpu==1
                                    RestrictedSortedWeights=StationaryDist_ii(:,j1:jend).*gpuArray(RestrictionMask{ii,rr}(:,j1:jend));
                                else
                                    RestrictedSortedWeights=StationaryDist_ii(:,j1:jend).*RestrictionMask{ii,rr}(:,j1:jend);
                                end
                                RestrictedSortedWeights=reshape(RestrictedSortedWeights,[N_a_temp*N_z_temp*(jend-j1+1),1]); % restricted mass at each point (each age of the agegrouping weighted by its restricted mass)
                                RestrictedSortedWeights=RestrictedSortedWeights(~temp); % drop zeros masses (but ignoring the restrictions; this is just to match what was already done to SortedValues_jj)
                                RestrictedSortedWeights=accumarray(sortindex,RestrictedSortedWeights,[],@sum); % This has already been done to SortedValues, so have to do it to Restricted Agent Dist
                                RestrictedSortedWeights=RestrictedSortedWeights/sum(RestrictedSortedWeights(:)); % renormalize to 1

                                tempStatsRestricted=StatsFromWeightedGrid(SortedValues_jj,RestrictedSortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,ws);

                                % Now store these based on jj
                                if ws(1)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Mean(jj)=tempStatsRestricted.Mean;
                                end
                                if ws(2)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Median(jj)=tempStatsRestricted.Median;
                                    if ws(1)==1
                                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).RatioMeanToMedian(jj)=tempStatsRestricted.RatioMeanToMedian;
                                    end
                                end
                                if ws(3)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Variance(jj)=tempStatsRestricted.Variance;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).StdDeviation(jj)=tempStatsRestricted.StdDeviation;
                                end
                                if ws(4)>=1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Gini(jj)=tempStatsRestricted.Gini;
                                    if ws(4)<3
                                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).LorenzCurve(:,jj)=tempStatsRestricted.LorenzCurve;
                                    end
                                end
                                if ws(5)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Minimum(jj)=tempStatsRestricted.Minimum;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).Maximum(jj)=tempStatsRestricted.Maximum;
                                end
                                if ws(6)>=1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).QuantileCutoffs(:,jj)=tempStatsRestricted.QuantileCutoffs;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).QuantileMeans(:,jj)=tempStatsRestricted.QuantileMeans;
                                end
                                if ws(7)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Top1share(jj)=tempStatsRestricted.MoreInequality.Top1share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Top5share(jj)=tempStatsRestricted.MoreInequality.Top5share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Top10share(jj)=tempStatsRestricted.MoreInequality.Top10share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Bottom50share(jj)=tempStatsRestricted.MoreInequality.Bottom50share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile50th(jj)=tempStatsRestricted.MoreInequality.Percentile50th;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile90th(jj)=tempStatsRestricted.MoreInequality.Percentile90th;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile95th(jj)=tempStatsRestricted.MoreInequality.Percentile95th;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(iistr).MoreInequality.Percentile99th(jj)=tempStatsRestricted.MoreInequality.Percentile99th;
                                end
                            else
                                RestrictedSortedWeights=zeros(size(SortedValues_jj),'gpuArray'); % Need this size for groupstats later
                            end

                            % If doing grouped stats, append RestrictedSortedWeights to the pooled cell
                            if simoptions.groupptypesforstats==1 && whichcombosG(ff,jjs,1+rr)==1 % (the grouped stats of this restriction are wanted)
                                if simoptions.ptypestorecpu==1
                                    PoolRestrWeights{rr}=[PoolRestrWeights{rr}; gather(RestrictedSortedWeights)*gather(StationaryDist.ptweights(ii)*sum(restrictedsamplemass(ii,j1:jend,rr)))];
                                else
                                    PoolRestrWeights{rr}=[PoolRestrWeights{rr}; RestrictedSortedWeights*StationaryDist.ptweights(ii)*sum(restrictedsamplemass(ii,j1:jend,rr))];
                                end
                                % Weight of this ptype in the grouped stats is ptweights(ii) times its restricted mass in this agegrouping (restrictedsamplemass is mass within the ptype, so already includes the age weights)
                                % Note: later normalize by sum(sum(restrictedsamplemass(:,j1:jend,rr),2))
                            end
                            end % whichcombos_ii(ff,jjs,1+rr)
                        end
                    end
                end % this ptype has age group jjs
            end % this function is relevant to this ptype
        end % end ii over N_i (pass 2)

        %% Pool this (ff,jjs) cell across ptypes and compute the grouped stats
        if simoptions.groupptypesforstats==1 && any(whichcombosG(ff,jjs,:)) % (some grouped stat of this (function, age group) is wanted; nothing was appended to the pooled cell otherwise)
            jj=jjs;
            ws=reshape(whichstatsG(ff,jjs,1,:),[1,7]); % whichstats of the grouped stats for this (fn, age group)

            if simoptions.groupusingtdigest==1 % using t-Digests
                Cmerge=Cmerge(1:merge_nsofar);
                digestweightsmerge=digestweightsmerge(1:merge_nsofar);
                [C_ff,digestweights_ff,~]=mergeDigest(Cmerge, digestweightsmerge, delta);
                % digestweights_ff will sum to one, except if using different agejshifter across PTypes, so need to add a renormalization in case that is happening
                if sum(digestweights_ff)>0
                    digestweights_ff=digestweights_ff/sum(digestweights_ff);
                end
                if whichcombosG(ff,jjs,1)==1 % the grouped unrestricted stats of this (function, age group) are wanted (the pooled unique() above is needed regardless, the restricted weights are aligned to it)
                tempStats=StatsFromWeightedGrid(C_ff,digestweights_ff,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws);
                end % whichcombosG(ff,jjs,1)
            else % just using unique() of the values and weights
                [PoolValues,~,sortindex]=unique(PoolValues);
                PoolWeights=accumarray(sortindex,PoolWeights,[],@sum);
                % PoolWeights will sum to one, except if using different agejshifter across PTypes, so need to add a renormalization in case that is happening
                if sum(PoolWeights)>0
                    PoolWeights=PoolWeights/sum(PoolWeights);
                end
                if whichcombosG(ff,jjs,1)==1 % the grouped unrestricted stats of this (function, age group) are wanted (the pooled unique() above is needed regardless, the restricted weights are aligned to it)
                tempStats=StatsFromWeightedGrid(PoolValues,PoolWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,ws); % 2: sorted, but a ptype of zero mass contributes zero weights
                end % whichcombosG(ff,jjs,1)
            end
            if whichcombosG(ff,jjs,1)==1 % store the grouped unrestricted stats of this (function, age group)
                % Store them in AgeConditionalStats
                if ws(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=tempStats.Mean;
                end
                if ws(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Median(jj)=tempStats.Median;
                    if ws(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(jj)=tempStats.RatioMeanToMedian;
                    end
                end
                if ws(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=tempStats.Variance;
                    AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=tempStats.StdDeviation;
                end
                if ws(4)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Gini(jj)=tempStats.Gini;
                    if ws(4)<3
                        AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,jj)=tempStats.LorenzCurve;
                    end
                end
                if ws(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=tempStats.Minimum;
                    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=tempStats.Maximum;
                end
                if ws(6)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,jj)=tempStats.QuantileCutoffs;
                    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,jj)=tempStats.QuantileMeans;
                end
                if ws(7)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(jj)=tempStats.MoreInequality.Top1share;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(jj)=tempStats.MoreInequality.Top5share;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(jj)=tempStats.MoreInequality.Top10share;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(jj)=tempStats.MoreInequality.Bottom50share;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(jj)=tempStats.MoreInequality.Percentile50th;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(jj)=tempStats.MoreInequality.Percentile90th;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(jj)=tempStats.MoreInequality.Percentile95th;
                    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(jj)=tempStats.MoreInequality.Percentile99th;
                end

                % Grouped mean and standard deviation are overwritten on a more direct calculation that does not involve the digests
                % The weight of a ptype in the age group is ptweights times its mass in the age group (the ptypes can have different age weights)
                wvec=FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*GroupMassVec(:,jj)';
                relevant=(wvec>0); % the ptypes in the population of this age group for which the function is relevant
                SigmaNxi=sum(wvec(relevant)); % The sum of the masses of the relevant types in the age group

                % Mean
                if ws(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=sum(wvec(relevant).*MeanVec(ff,relevant,jj))/SigmaNxi;
                end

                % Standard Deviation: within-type variances plus the between-type part (sum over pairs of types of w_i*w_k*(m_i-m_k)^2, over SigmaNxi^2)
                if ws(3)==1
                    if N_i==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=StdDevVec(ff,:,jj);
                    else
                        temp2=zeros(N_i,1);
                        for ii=2:N_i
                            if relevant(ii)
                                temp=MeanVec(ff,1:(ii-1),jj)-MeanVec(ff,ii,jj); % This bit with temp is just to handle numerical rounding errors where temp evaluated to negative with order -15
                                if any(temp<0) && all(temp>10^(-12))
                                    temp=max(temp,0);
                                end
                                temp(~relevant(1:(ii-1)))=0; % (a ptype not in the population contributes nothing, and its NaN mean must not get in)
                                temp2(ii)=wvec(ii)*sum(wvec(1:(ii-1)).*(temp.^2));
                            end
                        end
                        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=sqrt(sum(wvec(relevant).*(StdDevVec(ff,relevant,jj).^2))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
                    end
                    AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=(AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj))^2;
                end

                % With t-Digests the pooled extremes are only approximate, so the minimum and maximum are taken directly from the ptypes (and
                % overwritten); without them the pooled stats are exact, and the minimum and maximum are those of the pooled population, as the
                % other stats (the min/max over the ptypes differs from them, as the tolerance is then a mass within each ptype, not within the population)
                if ws(5)==1 && simoptions.groupusingtdigest==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=max(maxvaluevec(ff,:,jj));
                    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=min(minvaluevec(ff,:,jj));
                end
            end % whichcombosG(ff,jjs,1)

                %% Deal with conditional restrictions
                if useCondlRest==1
                    j1=simoptions_temp.agegroupings(jj);
                    if jj<length(simoptions_temp.agegroupings)
                        jend=simoptions_temp.agegroupings(jj+1)-1;
                    else
                        jend=N_j_temp;
                    end

                    for rr=1:length(CondlRestnFnNames)
                        if whichcombosG(ff,jjs,1+rr)==1 % the grouped stats of this restriction are wanted for this (function, age group)
                        ws=reshape(whichstatsG(ff,jjs,1+rr,:),[1,7]); % whichstats of the grouped stats for this (fn, age group, restriction)

                        if sum(StationaryDist.ptweights.*sum(restrictedsamplemass(:,j1:jend,rr),2,'omitnan'))>0 % the population (not just some ptype) has restricted mass in this agegrouping [a ptype of zero mass can have restricted mass of its own]
                            % We need to load up each ii, and put them together
                            if simoptions.groupusingtdigest==1 % using t-Digests
                                error('You should not be able to get here in the code')
                            else % just using unique() of the values and weights
                                % [PoolValues,~,sortindex]=unique(PoolValues);
                                AllRestrictedWeights_rrffjj=accumarray(sortindex,PoolRestrWeights{rr}/sum(sum(restrictedsamplemass(:,j1:jend,rr),2)),[],@sum);
                                AllRestrictedWeights_rrffjj=AllRestrictedWeights_rrffjj/sum(AllRestrictedWeights_rrffjj(:));

                                % AllRestrictedWeights_rrffjj will sum to one, except if using different agejshifter across PTypes, so need to add a renormalization in case that is happening
                                if sum(AllRestrictedWeights_rrffjj)>0
                                    AllRestrictedWeights_rrffjj=AllRestrictedWeights_rrffjj/sum(AllRestrictedWeights_rrffjj);
                                end

                                tempStats2=StatsFromWeightedGrid(PoolValues,AllRestrictedWeights_rrffjj,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,ws); % 2: sorted, with zero weights (outside the restriction)
                            end
                            % Store them in AgeConditionalStats
                            if ws(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(jj)=tempStats2.Mean;
                            end
                            if ws(2)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(jj)=tempStats2.Median;
                                if ws(1)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(jj)=tempStats2.RatioMeanToMedian;
                                end
                            end
                            if ws(3)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(jj)=tempStats2.Variance;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(jj)=tempStats2.StdDeviation;
                            end
                            if ws(4)>=1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(jj)=tempStats2.Gini;
                                if ws(4)<3
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,jj)=tempStats2.LorenzCurve;
                                end
                            end
                            if ws(5)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(jj)=tempStats2.Minimum;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(jj)=tempStats2.Maximum;
                            end
                            if ws(6)>=1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,jj)=tempStats2.QuantileCutoffs;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,jj)=tempStats2.QuantileMeans;
                            end
                            if ws(7)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share(jj)=tempStats2.MoreInequality.Top1share;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share(jj)=tempStats2.MoreInequality.Top5share;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share(jj)=tempStats2.MoreInequality.Top10share;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share(jj)=tempStats2.MoreInequality.Bottom50share;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th(jj)=tempStats2.MoreInequality.Percentile50th;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th(jj)=tempStats2.MoreInequality.Percentile90th;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th(jj)=tempStats2.MoreInequality.Percentile95th;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th(jj)=tempStats2.MoreInequality.Percentile99th;
                            end

                            % For unrestricted stats, I do a more direct calculation of mean, std dev, min and max. But I don't bother with the conditional restriction stats.
                        end
                        end % whichcombosG(ff,jjs,1+rr)
                    end
                end
        end % groupptypesforstats
    end % end jjs over age groups
    clear ValuesOnGrid_ff
end % end ff over FnsToEvalNames

%% Multi-period sums/averages (simoptions.ntimeperiods with M>1)
% For each start age j with something wanted, each ptype is computed by LifeCycleProfiles_FHorz_Case1 with the selection of start
% age j only: its own window statistics (its slot of whichcombos), and, for the grouped stats, the exact path lists of the windows
% starting at j (simoptions.windowlists, internal: the M-period sums and their masses within the ptype). The grouped statistics are
% those of the pooled lists, each ptype weighted by ptweights times its mass at age j (the restricted mass for a restriction).
if nW>0
    wantWT=(wcW+wcLogW)>0; % [numFnsBase,N_j,nwhichpages,N_i+1,nW]
    WNames=cell(numFnsBase,nW,2); % the output names: level and log
    for ff=1:numFnsBase
        for lw=1:nW
            WNames{ff,lw,1}=[FnsToEvalNames{ff},'_',num2str(windowM(lw)),windowstr];
            WNames{ff,lw,2}=[FnsToEvalNames{ff},'_',num2str(windowM(lw)),windowstr,'_log'];
        end
    end
    % Preallocate: fields for each (row, page, slot) selected at some start age (a ptype only where the function is relevant to it)
    for ff=1:numFnsBase
        for lw=1:nW
            for kind=1:2
                for pp=1:nwhichpages
                    for slot=1:N_i+1
                        if slot<=N_i && FnsAndPTypeIndicator(ff,slot)==0
                            continue
                        end
                        if kind==1
                            selW=reshape(wcW(ff,:,pp,slot,lw),[1,N_j]);
                        else
                            selW=reshape(wcLogW(ff,:,pp,slot,lw),[1,N_j]);
                        end
                        if ~any(selW)
                            continue
                        end
                        wsP=max(reshape(wsW(ff,selW==1,pp,slot,:,lw),[sum(selW),7]),[],1); % a statistic is preallocated if a selected start age asks for it
                        ws4=reshape(wsW(ff,selW==1,pp,slot,4,lw),[1,sum(selW)]);
                        if any(ws4>=1 & ws4<3)
                            wsP(4)=1;
                        end
                        SS=struct();
                        if wsP(1)==1
                            SS.Mean=nan(1,N_j,'gpuArray');
                        end
                        if wsP(2)==1
                            SS.Median=nan(1,N_j,'gpuArray');
                        end
                        if wsP(1)==1 && wsP(2)==1
                            SS.RatioMeanToMedian=nan(1,N_j,'gpuArray');
                        end
                        if wsP(3)==1
                            SS.Variance=nan(1,N_j,'gpuArray');
                            SS.StdDeviation=nan(1,N_j,'gpuArray');
                        end
                        if wsP(4)>=1
                            SS.Gini=nan(1,N_j,'gpuArray');
                            if wsP(4)<3
                                SS.LorenzCurve=nan(simoptions.npoints,N_j,'gpuArray');
                            end
                        end
                        if wsP(5)==1
                            SS.Minimum=nan(1,N_j,'gpuArray');
                            SS.Maximum=nan(1,N_j,'gpuArray');
                        end
                        if wsP(6)>=1
                            SS.QuantileCutoffs=nan(simoptions.nquantiles+1,N_j,'gpuArray'); % Includes the min and max values
                            SS.QuantileMeans=nan(simoptions.nquantiles,N_j,'gpuArray');
                        end
                        if wsP(7)==1
                            SS.MoreInequality.Top1share=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Top5share=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Top10share=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Bottom50share=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Percentile50th=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Percentile90th=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Percentile95th=nan(1,N_j,'gpuArray');
                            SS.MoreInequality.Percentile99th=nan(1,N_j,'gpuArray');
                        end
                        if slot<=N_i
                            if pp==1
                                AgeConditionalStats.(WNames{ff,lw,kind}).(Names_i{slot})=SS;
                            else
                                AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind}).(Names_i{slot})=SS;
                            end
                        else
                            fnSS=fieldnames(SS);
                            for kk=1:length(fnSS)
                                if pp==1
                                    AgeConditionalStats.(WNames{ff,lw,kind}).(fnSS{kk})=SS.(fnSS{kk});
                                else
                                    AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind}).(fnSS{kk})=SS.(fnSS{kk});
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    for jj=1:N_j
        if ~any(wantWT(:,jj,:,:,:),'all')
            continue
        end
        % The pooled path lists of start age jj (grouped stats)
        PoolS=cell(numFnsBase,nW,nwhichpages);
        PoolM=cell(numFnsBase,nW,nwhichpages);
        for ii=1:N_i
            iistr=Names_i{ii};
            % the functions this ptype computes at start age jj: its own slot, and the grouped one (a ptype of zero weight adds nothing to it)
            wantT=reshape(any(any(wantWT(:,jj,:,ii,:),3),5),[numFnsBase,1]);
            if StationaryDist.ptweights(ii)>0
                wantL=reshape(any(any(wantWT(:,jj,:,N_i+1,:),3),5),[numFnsBase,1]);
            else
                wantL=false(numFnsBase,1);
            end
            fnsc=find(reshape(gather(FnsAndPTypeIndicator(1:numFnsBase,ii)),[numFnsBase,1])==1 & (wantT | wantL))';
            if isempty(fnsc)
                continue
            end
            nfc=length(fnsc);

            % This ptype's setup (as for the other PType commands that call the single-type command)
            simoptions_temp=PType_Options(simoptions,iistr);
            if useCondlRest==1 % PType_Options only keeps a structure-valued option with a field for this type: put the restrictions back
                simoptions_temp.conditionalrestrictions=struct();
                for rr=1:length(CondlRestnFnNames)
                    if isstruct(simoptions.conditionalrestrictions.(CondlRestnFnNames{rr}))
                        simoptions_temp.conditionalrestrictions.(CondlRestnFnNames{rr})=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr}).(iistr);
                    else
                        simoptions_temp.conditionalrestrictions.(CondlRestnFnNames{rr})=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
                    end
                end
            end
            if simoptions_temp.ptypestorecpu==1
                PolicyIndexes_temp=gpuArray(Policy.(iistr));
                StationaryDist_temp=gpuArray(StationaryDist.(iistr));
            else
                PolicyIndexes_temp=Policy.(iistr);
                StationaryDist_temp=StationaryDist.(iistr);
            end
            [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);
            if isfield(simoptions_temp,'pi_z')
                pi_z_in=simoptions_temp.pi_z;
            else
                pi_z_in=[];
            end
            [n_z_temp,z_grid_temp,pi_z_temp,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,pi_z_in,simoptions_temp,3);
            if ~isempty(pi_z_temp)
                simoptions_temp.pi_z=pi_z_temp;
            end
            Parameters_temp=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);
            if n_d_temp(1)==0
                l_d_temp=0;
            else
                l_d_temp=length(n_d_temp);
            end
            l_a_temp=length(n_a_temp);
            if prod(n_z_temp)==0
                l_z_temp=0;
            else
                l_z_temp=length(n_z_temp);
            end
            [FnsToEvaluate_temp,~,~,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
            Fc=struct();
            for kk=1:nfc
                Fc.(FnsToEvalNames{fnsc(kk)})=FnsToEvaluate_temp.(FnsToEvalNames{fnsc(kk)});
            end
            % The selection of start age jj: rows in blocks of the nfc functions, one block per window length
            wcc=zeros(nW*nfc,N_j,nwhichpages);
            lwc=zeros(nW*nfc,N_j,nwhichpages);
            wsc=zeros(nW*nfc,N_j,nwhichpages,7);
            wl=zeros(nfc,N_j,nwhichpages,nW);
            for lw=1:nW
                for kk=1:nfc
                    ff=fnsc(kk);
                    row=(lw-1)*nfc+kk;
                    wcc(row,jj,:)=reshape(wcW(ff,jj,:,ii,lw)+wcLogW(ff,jj,:,ii,lw),[1,1,nwhichpages]);
                    lwc(row,jj,:)=reshape(wcLogW(ff,jj,:,ii,lw),[1,1,nwhichpages]);
                    wsc(row,jj,:,:)=reshape(wsW(ff,jj,:,ii,:,lw),[1,1,nwhichpages,7]);
                    if StationaryDist.ptweights(ii)>0
                        wl(kk,jj,:,lw)=reshape(wantWT(ff,jj,:,N_i+1,lw),[1,1,nwhichpages]);
                    end
                end
            end
            simoptions_c=simoptions_temp;
            simoptions_c.ntimeperiods=windowM;
            simoptions_c.agegroupings=1:1:N_j;
            simoptions_c.whichcombos=wcc;
            simoptions_c.logwhichcombos=lwc;
            simoptions_c.whichstats=wsc;
            simoptions_c.windowlists=wl;
            simoptions_c.warnzerorestrictedmass=0; % (warned about above)
            simoptions_c.verbose=0;
            ACS_ii=LifeCycleProfiles_FHorz_Case1(StationaryDist_temp,PolicyIndexes_temp,Fc,Parameters_temp,[],n_d_temp,n_a_temp,n_z_temp,N_j,d_grid_temp,a_grid_temp,z_grid_temp,simoptions_c);

            for kk=1:nfc
                ff=fnsc(kk);
                for lw=1:nW
                    for pp=1:nwhichpages
                        % this ptype's own statistics at start age jj
                        for kind=1:2
                            if (kind==1 && wcW(ff,jj,pp,ii,lw)==0) || (kind==2 && wcLogW(ff,jj,pp,ii,lw)==0)
                                continue
                            end
                            if pp==1
                                src=ACS_ii.(WNames{ff,lw,kind});
                                dst=AgeConditionalStats.(WNames{ff,lw,kind}).(iistr);
                            else
                                src=ACS_ii.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind});
                                dst=AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind}).(iistr);
                            end
                            fl=fieldnames(src);
                            for kf=1:length(fl)
                                if strcmp(fl{kf},'MoreInequality')
                                    fm=fieldnames(src.MoreInequality);
                                    for km=1:length(fm)
                                        dst.MoreInequality.(fm{km})(jj)=src.MoreInequality.(fm{km})(jj);
                                    end
                                else
                                    dst.(fl{kf})(:,jj)=src.(fl{kf})(:,jj);
                                end
                            end
                            if pp==1
                                AgeConditionalStats.(WNames{ff,lw,kind}).(iistr)=dst;
                            else
                                AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind}).(iistr)=dst;
                            end
                        end
                        % this ptype's path list, for the grouped stats
                        if wl(kk,jj,pp,lw)==1 && ~isempty(ACS_ii.WindowLists{kk,lw,pp,jj})
                            if pp==1
                                startmass=AgeMasses(ii,jj); % the mass of this ptype at age jj (its distribution includes its age weights)
                            else
                                startmass=restrictedsamplemass(ii,jj,pp-1); % its restricted mass at age jj
                            end
                            [Su,~,sortindex]=unique(ACS_ii.WindowLists{kk,lw,pp,jj}.S); % merge equal sums within the ptype first (the pooled cell is smaller)
                            mu_ii=accumarray(sortindex,ACS_ii.WindowLists{kk,lw,pp,jj}.m,[],@sum);
                            PoolS{ff,lw,pp}=[PoolS{ff,lw,pp}; Su];
                            PoolM{ff,lw,pp}=[PoolM{ff,lw,pp}; mu_ii*(StationaryDist.ptweights(ii)*startmass)];
                        end
                    end
                end
            end
            clear ACS_ii
        end % ii

        % The grouped stats of start age jj, from the pooled path lists
        for ff=1:numFnsBase
            for lw=1:nW
                M=windowM(lw);
                for pp=1:nwhichpages
                    if isempty(PoolS{ff,lw,pp}) || sum(PoolM{ff,lw,pp})==0
                        continue % no ptype has mass there: stays NaN
                    end
                    for kind=1:2
                        if (kind==1 && wcW(ff,jj,pp,N_i+1,lw)==0) || (kind==2 && wcLogW(ff,jj,pp,N_i+1,lw)==0)
                            continue
                        end
                        vals=PoolS{ff,lw,pp};
                        if strcmp(simoptions.ntimeperiodstype,'average')
                            vals=vals/M;
                        end
                        if kind==2
                            if any(vals(PoolM{ff,lw,pp}>0)<=0)
                                error(['LifeCycleProfiles_FHorz_Case1_PType: simoptions.logwhichcombos asks for the log of the ',num2str(M),'-period sum/average of ',FnsToEvalNames{ff},' starting at age ',num2str(jj),', but it is not positive for some agents with positive mass (use a conditional restriction, or a floor in the FnsToEvaluate)'])
                            end
                            vals=log(vals);
                        end
                        [vals,~,sortindex]=unique(vals);
                        wpool=accumarray(sortindex,PoolM{ff,lw,pp},[],@sum);
                        wpool=wpool/sum(wpool);
                        ws=reshape(wsW(ff,jj,pp,N_i+1,:,lw),[1,7]);
                        tempStats=StatsFromWeightedGrid(vals,wpool,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,ws); % 2: sorted, a ptype of zero mass contributes zero weights

                        if pp==1
                            SS=AgeConditionalStats.(WNames{ff,lw,kind});
                        else
                            SS=AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind});
                        end
                        if ws(1)==1
                            SS.Mean(jj)=tempStats.Mean;
                        end
                        if ws(2)==1
                            SS.Median(jj)=tempStats.Median;
                            if ws(1)==1
                                SS.RatioMeanToMedian(jj)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if ws(3)==1
                            SS.Variance(jj)=tempStats.Variance;
                            SS.StdDeviation(jj)=tempStats.StdDeviation;
                        end
                        if ws(4)>=1
                            SS.Gini(jj)=tempStats.Gini;
                            if ws(4)<3
                                SS.LorenzCurve(:,jj)=tempStats.LorenzCurve;
                            end
                        end
                        if ws(5)==1
                            SS.Minimum(jj)=tempStats.Minimum;
                            SS.Maximum(jj)=tempStats.Maximum;
                        end
                        if ws(6)>=1
                            SS.QuantileCutoffs(:,jj)=tempStats.QuantileCutoffs;
                            SS.QuantileMeans(:,jj)=tempStats.QuantileMeans;
                        end
                        if ws(7)==1
                            SS.MoreInequality.Top1share(jj)=tempStats.MoreInequality.Top1share;
                            SS.MoreInequality.Top5share(jj)=tempStats.MoreInequality.Top5share;
                            SS.MoreInequality.Top10share(jj)=tempStats.MoreInequality.Top10share;
                            SS.MoreInequality.Bottom50share(jj)=tempStats.MoreInequality.Bottom50share;
                            SS.MoreInequality.Percentile50th(jj)=tempStats.MoreInequality.Percentile50th;
                            SS.MoreInequality.Percentile90th(jj)=tempStats.MoreInequality.Percentile90th;
                            SS.MoreInequality.Percentile95th(jj)=tempStats.MoreInequality.Percentile95th;
                            SS.MoreInequality.Percentile99th(jj)=tempStats.MoreInequality.Percentile99th;
                        end
                        if pp==1
                            AgeConditionalStats.(WNames{ff,lw,kind})=SS;
                        else
                            AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind})=SS;
                        end
                    end
                end
            end
        end
    end % jj
end

%% The (fn, page) with nothing computed (only with simoptions.ntimeperiods or simoptions.logwhichcombos): no fields
for ff=1:numFnsToEvaluate
    for pp=1:nwhichpages
        if skipFields(ff,pp)==1
            if pp==1 && isfield(AgeConditionalStats,FnsToEvalNames{ff})
                AgeConditionalStats=rmfield(AgeConditionalStats,FnsToEvalNames{ff});
            elseif pp>1 && isfield(AgeConditionalStats.(CondlRestnFnNames{pp-1}),FnsToEvalNames{ff})
                AgeConditionalStats.(CondlRestnFnNames{pp-1})=rmfield(AgeConditionalStats.(CondlRestnFnNames{pp-1}),FnsToEvalNames{ff});
            end
        end
    end
end


    if useCondlRest==1 % Store the restricted masses
        for rr=1:length(CondlRestnFnNames)
            if sum(sum(restrictedsamplemass(:,:,rr)))==0 && simoptions.warnzerorestrictedmass>=1
                warning('One of the conditional restrictions evaluates to a zero mass (for every permanent type, at all j)')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            end
            % Note: restrictedsamplemass(ii,j,rr) is the mass of ptype ii at age j that satisfies the restriction, as a share of ptype ii (so it already includes the age weights)
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.ByAge=sum(restrictedsamplemass(:,:,rr).*StationaryDist.ptweights,1); % Mass at each age that satisfies the restriction, as a share of the whole population (same as RestrictedSampleMass in LifeCycleProfiles_FHorz_Case1)
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.ByPType=sum(restrictedsamplemass(:,:,rr),2,'omitnan'); % Conditional on ptype, what fraction satisfy restriction ['omitnan' as ages beyond the N_j of a ptype are NaN]
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.Total=sum(StationaryDist.ptweights.*sum(restrictedsamplemass(:,:,rr),2,'omitnan')); % What fraction of the population satisfy restriction

        end
    end


end
