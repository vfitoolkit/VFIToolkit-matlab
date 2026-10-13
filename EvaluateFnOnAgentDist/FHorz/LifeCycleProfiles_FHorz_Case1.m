function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% Similar to SimLifeCycleProfiles but works from StationaryDist rather than
% simulating panel data. Where applicable it is faster and more accurate.
% options.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., options.agegroupings=1:10:N_j will divide into 10 year age bins and calculate stats for each of them
% simoptions.whichcombos ([numFnsToEvaluate, number of age groups, 1+number of conditional restrictions]) selects which combinations are
% computed, and simoptions.whichstats may be given per combination ([.., .., .., 7]); see the whichcombos section below.
% options.npoints can be used to determine how many points are used for the lorenz curve
% options.nquantiles can be used to change from reporting (age conditional) ventiles, to quartiles/deciles/percentiles/etc.
%
% Note that the quantile are what are typically reported as life-cycle profiles (or more precisely, the quantile cutoffs).
%
% Use simoptions.ntimeperiods (default 1) to get the same statistics of the sum or average of a function over M consecutive
% periods: the statistic of a person at age j is W_j=x_j+...+x_{j+M-1} (simoptions.ntimeperiodstype='sum'), or that divided
% by M (simoptions.ntimeperiodstype='average', the default). A vector, e.g. simoptions.ntimeperiods=[1,2,5], gives each of
% them (M=1 is the single-period statistic). Any M>1 needs simoptions.agegroupings=1:1:N_j (the default), and needs the
% transition probabilities of z as simoptions.pi_z (the same shapes as the pi_z input of StationaryDist_FHorz_Case1; not
% needed with simoptions.ExogShockFn); the e and semi-exogenous shocks are read from simoptions as usual. All of the
% statistics of W_j are exact (W_j depends on the path of the states from j to j+M-1, which is enumerated: the paths that
% reach the same state with the same partial sum are merged, there is no approximation; the cost grows with M as the number
% of distinct (state, partial sum) pairs, so a large M with many shocks can run out of memory). Conditional restrictions
% are evaluated at the window start age j. The M=1 outputs are under AgeConditionalStats.(fnname) as always; the M>1
% outputs are under AgeConditionalStats.([fnname,'_',M,'periodAverage']) (or 'periodSum'), e.g. earnings_2periodAverage,
% with the same fields (1 x N_j, the last M-1 entries NaN as the window runs past age N_j). An age with zero mass is NaN.
% With ntimeperiods of length L, simoptions.whichcombos and simoptions.whichstats may have L*numFnsToEvaluate rows, in
% blocks of numFnsToEvaluate in the order of simoptions.ntimeperiods (row (ll-1)*numFnsToEvaluate+ff is function ff with
% M=ntimeperiods(ll)); numFnsToEvaluate rows apply to every M.
% Use simoptions.logwhichcombos (zeros and ones, the same shapes as simoptions.whichcombos, or a scalar 0 or 1 for all; default
% 0) to get the statistics of the log of the function (M=1) or of the log of the M-period sum/average (log of the average,
% not the average of the logs) at the combinations where it is one: those are reported under [fnname,'_log'] (or
% [fnname,'_',M,'periodAverage_log']) instead of fnname, with the same fields. The function (or its sum) must be positive
% on every agent with positive mass in the populations used (an error says where it is not; use a conditional
% restriction, e.g. alive, or a floor in the FnsToEvaluate). The log of M=1 works with any agegroupings.
% The new outputs (M>1 and the logs) have fields only for the (row, restriction) that whichcombos selects at some age, and a
% function all of whose single-period selections on a page are of the log has only the fnname_log fields there.
%
% Output takes following form
% ngroups=length(options.agegroupings);
% AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups); % Includes the min and max values
% AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups);

%% Temporary warning due to changing function input order
if isempty(Parameters)
    warning('LifeCycleProfiles_FHorz_Case1 has changed the order of the fourth and fifth inputs (should now be something like Params,[] when previously it would have been [],Params)')
    warning('Annoying, but it makes inputs to LifeCycleProfiles_FHorz_Case1 have same order as those of similar functions')
end

%% Check which simoptions have been declared, set all others to defaults
if ~exist('simoptions','var')
    %If options is not given, just use all the defaults
    if isgpuarray(StationaryDist)
        simoptions.parallel=2;
    else
        simoptions.parallel=1;
    end
    simoptions.verbose=0;
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, 7]
    % Model Setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    simoptions.ntimeperiods=1; % the statistics of the sum/average over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    simoptions.logwhichcombos=0; % the combinations whose statistics are of the log (same shapes as whichcombos, or a scalar)
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
else
    %Check options for missing fields, if there are some fill them with the defaults
    if isgpuarray(StationaryDist) % simoptions.parallel is overwritten based on StationaryDist
        simoptions.parallel=2;
    else
        simoptions.parallel=1;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if ~isfield(simoptions,'agegroupings')
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if isfield(simoptions,'SampleRestrictionFn') % If using SampleRestrictionFn then need to set some things
        if ~isfield(simoptions,'SampleRestrictionFn_include')
            simoptions.SampleRestrictionFn_include=1; % By default, include observations that meet the sample restriction (if zero, then exclude observations meeting this criterion)
        end
        simoptions.SampleRestrictionFnParamNames=getAnonymousFnInputNames(simoptions.SampleRestrictionFn); % Note: we remove those relating to the state space later
    end
    if ~isfield(simoptions,'whichstats')
        if any(simoptions.agegroupings(2:end)-simoptions.agegroupings(1:end-1)>4)
            % if some agegroupings are 'large', use the slower but lower memory versions
            simoptions.whichstats=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, 7]
        else
            simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, 7]
        end
    end
    % Model Setup
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
    if ~isfield(simoptions,'ntimeperiods')
        simoptions.ntimeperiods=1; % the statistics of the sum/average over M consecutive periods, for each M in ntimeperiods (1 is the single period)
    end
    if ~isfield(simoptions,'ntimeperiodstype')
        simoptions.ntimeperiodstype='average'; % 'average' or 'sum' over the M periods
    end
    if ~isfield(simoptions,'logwhichcombos')
        simoptions.logwhichcombos=0; % the combinations whose statistics are of the log (same shapes as whichcombos, or a scalar)
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
end

% N_d=prod(n_d);
N_a=prod(n_a);
% N_z=prod(n_z);

l_a=length(n_a);

ngroups=length(simoptions.agegroupings);

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    numFnsToEvaluate=length(FnsToEvaluate);
end
numFnsBase=numFnsToEvaluate; % the functions given (the log rows of M=1 are added as extra functions below)

%% Multi-period sums/averages: simoptions.ntimeperiods (the window lengths M, 1 is the single period) and simoptions.ntimeperiodstype
ntimeperiods=gather(simoptions.ntimeperiods(:)'); % the order is kept: it is the order of the row blocks of whichcombos
if isempty(ntimeperiods) || ~isnumeric(ntimeperiods) || any(ntimeperiods<1) || any(ntimeperiods~=round(ntimeperiods))
    error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods must be a vector of positive integers (the number of periods summed/averaged over, 1 is the single period)')
end
if length(unique(ntimeperiods))<length(ntimeperiods)
    error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods must not contain repeated values')
end
if any(ntimeperiods>N_j)
    error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods cannot exceed N_j (no window of that many periods fits in the life cycle)')
end
if ~(ischar(simoptions.ntimeperiodstype) || isstring(simoptions.ntimeperiodstype)) || ~any(strcmp(simoptions.ntimeperiodstype,{'average','sum'}))
    error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiodstype must be ''average'' or ''sum''')
end
nblocks=length(ntimeperiods); % the number of row blocks of whichcombos
has1=any(ntimeperiods==1); % the single-period statistics are wanted
windowM=ntimeperiods(ntimeperiods>1); % the window lengths M>1, in the order given
nW=length(windowM);
if strcmp(simoptions.ntimeperiodstype,'average')
    windowstr='periodAverage';
else
    windowstr='periodSum';
end
if nW>0 && ~isequal(reshape(gather(simoptions.agegroupings),1,[]),1:1:N_j)
    error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods with M>1 (multi-period sums/averages) can only be used with simoptions.agegroupings=1:1:N_j (the default); the statistics of W_j are by window start age j')
end
if (nW>0 || any(simoptions.logwhichcombos(:))) && ~isstruct(FnsToEvaluate)
    error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods and simoptions.logwhichcombos need FnsToEvaluate as a structure (the output names are made from its field names)')
end

%%
if simoptions.parallel==1
    if nW>0 || any(simoptions.logwhichcombos(:))
        error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods with M>1 and simoptions.logwhichcombos are only implemented on the gpu')
    end
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_cpu(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
    return
end
% just make sure things are on gpu as they should be
StationaryDist=gpuArray(StationaryDist);
Policy=gpuArray(Policy);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate
% (the windows need the transition probabilities of the three separately: keep the shapes and simoptions as given)
n_zW=n_z;
simoptionsW=simoptions;
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);


%% I want to do some things now, so that they can be used in setting up conditional restrictions

a_gridvals=CreateGridvals(n_a,a_grid,1);
if N_z==0
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,1,3]); % (N_a,l_daprime,N_j)
else
    StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_z,l_daprime,N_j)
end

% Figure out l_daprime from PolicyValues
l_daprime=size(PolicyValues,1);


%% Implement new way of handling FnsToEvaluate

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    % FnsToEvalNames=fieldnames(FnsToEvaluate);
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
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    end
end

%% simoptions.whichcombos and per-combination simoptions.whichstats
% whichcombos: zeros/ones of size [numFnsToEvaluate, number of age groups] without conditional restrictions, and
% [numFnsToEvaluate, number of age groups, 1+number of restrictions] with them: page 1 is the unrestricted stats, pages 2:end the
% restrictions in the fieldnames order of simoptions.conditionalrestrictions. Ones are computed, zeros skipped; the output has the
% same fields either way and the skipped entries stay NaN; RestrictedSampleMass is always filled. Default all ones. A 2D input with
% restrictions is applied to every page. Intended for calibration/estimation, which only needs the targeted combinations.
% whichstats: the usual 1x7 vector, or [numFnsToEvaluate, number of age groups, 1+number of restrictions, 7] giving a whichstats
% vector for every combination ([numFnsToEvaluate, number of age groups, 7] is applied to every page). A field exists for a
% (fn, page) if any of its age groups asks for that statistic. A combination asking for no statistic is skipped like a whichcombos zero.
if isfield(simoptions,'conditionalrestrictions')
    nwhichpages=1+length(fieldnames(simoptions.conditionalrestrictions));
else
    nwhichpages=1;
end
% With simoptions.ntimeperiods of length L>1 the rows (of both) are L blocks of numFnsToEvaluate, one block per M in the order of
% simoptions.ntimeperiods; an input with numFnsToEvaluate rows applies to every M. Internally whichcombos and whichstatsArr are the
% M=1 block (zeros when 1 is not in ntimeperiods) and whichcombosW(:,:,:,lw), whichstatsW(:,:,:,:,lw) the block of windowM(lw).
nrowswc=nblocks*numFnsToEvaluate;
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,ngroups,nwhichpages); % whichcombos here is [nFns, nAgeGroups, 1+nRestr]
    if nW>0
        whichcombosW=ones(numFnsToEvaluate,ngroups,nwhichpages,nW);
    end
    if ~has1
        whichcombos=zeros(numFnsToEvaluate,ngroups,nwhichpages); % the single-period statistics are not wanted
    end
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    nrowsin=size(whichcombos,1); % numFnsToEvaluate, or nblocks*numFnsToEvaluate
    if ismatrix(whichcombos) && nwhichpages>1 && isequal(size(whichcombos),[nrowsin,ngroups]) && (nrowsin==numFnsToEvaluate || nrowsin==nrowswc)
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % 2D input with restrictions: apply to every page
    end
    if nblocks>1 && nrowsin==numFnsToEvaluate
        whichcombos=repmat(whichcombos,[nblocks,1,1]); % one selection for every M
    end
    if ~isequal(size(whichcombos,1:3),[nrowswc,ngroups,nwhichpages])
        if nblocks==1
            error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(ngroups),',',num2str(nwhichpages),'] (number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions; the third dimension is dropped when there are no conditional restrictions)'])
        else
            error(['simoptions.whichcombos must be of size [',num2str(nrowswc),',',num2str(ngroups),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, N_j, 1+number of conditional restrictions; the third dimension is dropped when there are no conditional restrictions), or have ',num2str(numFnsToEvaluate),' rows to apply to every M'])
        end
    end
    whichcombos=double(whichcombos);
    if nblocks>1 || ~has1
        whichcombosRows=whichcombos;
        if has1
            bb=find(ntimeperiods==1);
            whichcombos=whichcombosRows((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
        else
            whichcombos=zeros(numFnsToEvaluate,ngroups,nwhichpages);
        end
        whichcombosW=zeros(numFnsToEvaluate,ngroups,nwhichpages,nW);
        for lw=1:nW
            bb=find(ntimeperiods==windowM(lw));
            whichcombosW(:,:,:,lw)=whichcombosRows((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
        end
    end
end
wsG=simoptions.whichstats;
if isvector(wsG) && numel(wsG)==7
    whichstatsRows=repmat(reshape(wsG,[1,1,1,7]),[nrowswc,ngroups,nwhichpages,1]);
elseif ndims(wsG)==3 && (isequal(size(wsG),[numFnsToEvaluate,ngroups,7]) || isequal(size(wsG),[nrowswc,ngroups,7]))
    whichstatsRows=repmat(reshape(wsG,[size(wsG,1),ngroups,1,7]),[1,1,nwhichpages,1]);
elseif isequal(size(wsG,1:4),[numFnsToEvaluate,ngroups,nwhichpages,7]) || isequal(size(wsG,1:4),[nrowswc,ngroups,nwhichpages,7])
    whichstatsRows=wsG;
else
    if nblocks==1
        error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(numFnsToEvaluate),',',num2str(ngroups),',',num2str(nwhichpages),',7] (number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions, 7), or [',num2str(numFnsToEvaluate),',',num2str(ngroups),',7]'])
    else
        error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(nrowswc),',',num2str(ngroups),',',num2str(nwhichpages),',7] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, N_j, 1+number of conditional restrictions, 7), or [',num2str(nrowswc),',',num2str(ngroups),',7], or either with ',num2str(numFnsToEvaluate),' rows to apply to every M'])
    end
end
if nblocks>1 && size(whichstatsRows,1)==numFnsToEvaluate
    whichstatsRows=repmat(whichstatsRows,[nblocks,1,1,1]); % one whichstats for every M
end
if has1
    bb=find(ntimeperiods==1);
    whichstatsArr=whichstatsRows((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:,:);
else
    whichstatsArr=zeros(numFnsToEvaluate,ngroups,nwhichpages,7);
end
if nW>0
    whichstatsW=zeros(numFnsToEvaluate,ngroups,nwhichpages,7,nW);
    for lw=1:nW
        bb=find(ntimeperiods==windowM(lw));
        whichstatsW(:,:,:,:,lw)=whichstatsRows((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:,:);
    end
end

%% simoptions.logwhichcombos: the combinations (of whichcombos) whose statistics are of the log
% The same shapes as whichcombos (rows in blocks of simoptions.ntimeperiods), or a scalar 0/1 for every combination. A one at
% (row, age group, page) where whichcombos also has a one: the statistics of that row and page are of log(x) (M=1) or of
% log(W_j), the log of the M-period sum/average, and are reported under [rowname,'_log'] instead of rowname.
logwc=simoptions.logwhichcombos;
if ~(isnumeric(logwc) || islogical(logwc)) || any(logwc(:)~=0 & logwc(:)~=1)
    error('simoptions.logwhichcombos must contain only zeros and ones')
end
if isscalar(logwc)
    logwc=logwc*ones(nrowswc,ngroups,nwhichpages);
else
    nrowsin=size(logwc,1); % numFnsToEvaluate, or nblocks*numFnsToEvaluate
    if ismatrix(logwc) && nwhichpages>1 && isequal(size(logwc),[nrowsin,ngroups]) && (nrowsin==numFnsToEvaluate || nrowsin==nrowswc)
        logwc=repmat(logwc,[1,1,nwhichpages]); % 2D input with restrictions: apply to every page
    end
    if nblocks>1 && nrowsin==numFnsToEvaluate
        logwc=repmat(logwc,[nblocks,1,1]); % one selection for every M
    end
    if ~isequal(size(logwc,1:3),[nrowswc,ngroups,nwhichpages])
        error(['simoptions.logwhichcombos must be a scalar 0 or 1, or of the size of simoptions.whichcombos: [',num2str(nrowswc),',',num2str(ngroups),',',num2str(nwhichpages),'] (length of simoptions.ntimeperiods times the number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions), or with ',num2str(numFnsToEvaluate),' rows to apply to every M, or the same 2-D form as whichcombos'])
    end
    logwc=double(logwc);
end
uselog=any(logwc(:));
if nW>0
    wcLogW=zeros(numFnsToEvaluate,ngroups,nwhichpages,nW); % the window combinations of the log
end
fnlogbase=zeros(1,numFnsToEvaluate); % fnlogbase(ff) is the function an extra function is the log of (0 for the given ones)
logonlyBase=zeros(numFnsBase,nwhichpages); % a given function whose M=1 selections on a page are all of the log: it has no fields there (only fnname_log)
if uselog
    % Split each selection into its level part (whichcombos, whichcombosW keep only those) and its log part
    logwc1=zeros(numFnsToEvaluate,ngroups,nwhichpages); % the M=1 block of logwhichcombos
    if has1
        bb=find(ntimeperiods==1);
        logwc1=logwc((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
    end
    wcLog1=whichcombos.*logwc1; % the M=1 combinations of the log
    whichcombos=whichcombos.*(1-logwc1);
    for ff=1:numFnsBase
        for pp=1:nwhichpages
            logonlyBase(ff,pp)=any(wcLog1(ff,:,pp)) && ~any(whichcombos(ff,:,pp));
        end
    end
    for lw=1:nW
        bb=find(ntimeperiods==windowM(lw));
        wcLogW(:,:,:,lw)=whichcombosW(:,:,:,lw).*logwc((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:);
        whichcombosW(:,:,:,lw)=whichcombosW(:,:,:,lw).*(1-logwc((bb-1)*numFnsToEvaluate+1:bb*numFnsToEvaluate,:,:));
    end
    % The M=1 log rows are extra functions (fnname_log) whose values are the log of the function's values: they then go through
    % exactly the single-period computation, with the whichstats of the function
    for ff=1:numFnsBase
        if any(wcLog1(ff,:,:),'all')
            nF=length(FnsToEvalNames)+1;
            FnsToEvaluate{nF}=FnsToEvaluate{ff};
            FnsToEvaluateParamNames(nF).Names=FnsToEvaluateParamNames(ff).Names;
            FnsToEvalNames{nF}=[FnsToEvalNames{ff},'_log'];
            whichcombos(nF,:,:)=wcLog1(ff,:,:);
            whichstatsArr(nF,:,:,:)=whichstatsArr(ff,:,:,:);
            fnlogbase(nF)=ff;
        end
    end
    numFnsToEvaluate=length(FnsToEvalNames);
end
whichcombos=whichcombos.*any(whichstatsArr,4); % a combination with no statistic requested is skipped altogether
if nW>0
    for lw=1:nW
        whichcombosW(:,:,:,lw)=whichcombosW(:,:,:,lw).*any(whichstatsW(:,:,:,:,lw),4);
        wcLogW(:,:,:,lw)=wcLogW(:,:,:,lw).*any(whichstatsW(:,:,:,:,lw),4);
        whichcombosW(:,N_j-windowM(lw)+2:N_j,:,lw)=0; % no window of M periods starts after age N_j-M+1
        wcLogW(:,N_j-windowM(lw)+2:N_j,:,lw)=0;
    end
end


% Preallocate various things for the stats (as many will have jj as a dimension)
% Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
for ff=1:numFnsToEvaluate
    if (fnlogbase(ff)==0 && (~has1 || logonlyBase(ff,1))) || (fnlogbase(ff)>0 && ~any(whichcombos(ff,:,1))) % the single period is not wanted (or only its log), or this log row is not selected on this page
        continue
    end
    % whichstats for this fn across its age groups: a statistic is preallocated if any age group asks for it (the Lorenz curve if any age group has 1<=whichstats(4)<3)
    wsP=max(reshape(whichstatsArr(ff,:,1,:),[ngroups,7]),[],1);
    ws4=reshape(whichstatsArr(ff,:,1,4),[1,ngroups]);
    if any(ws4>=1 & ws4<3)
        wsP(4)=1;
    end
    if wsP(1)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(2)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(1)==1 && wsP(2)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(3)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(4)>=1
        AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
        if wsP(4)<3
            AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
        end
    end
    if wsP(5)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(6)>=1
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(7)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
end

%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),N_j);
    RestrictionMask=cell(length(CondlRestnFnNames),1); % each restriction kept as a logical mask over the (a,z) x j grid (1 byte per point); the restricted weights are formed from it where used

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
            RestrictionValues=zeros(N_a,N_j);
        else
            RestrictionValues=zeros(N_a,N_z,N_j);
        end
        for jj=1:N_j % Given the actual stats have to loop over j, I just do it here even though it could be done with EvalFnOnAgentDist_Grid_J instead
            % Get parameter values for Conditional Restriction functions
            CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames,jj);

            % Compute the restrictions
            if N_z==0
                RestrictionValues(:,jj)=logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,n_z,a_gridvals,[]));
            else
                RestrictionValues(:,:,jj)=logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj)));
            end
        end
        if N_z>0
            RestrictionValues=reshape(RestrictionValues,[N_a*N_z,N_j]);
        end


        % Keep the restricted mass at each age, and the mask
        restrictedsamplemass(rr,:)=sum(StationaryDist.*RestrictionValues,1); % mass at each age that satisfies the restriction (includes the age weights)
        RestrictionMask{rr}=logical(RestrictionValues);

        if all(restrictedsamplemass(rr,:)==0) && simoptions.warnzerorestrictedmass==2
            warning('One of the conditional restrictions evaluates to a zero mass (at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:); % Just return this and hopefully it is clear to the user
        else
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:); % Seems likely this would be something user might want
        end


        % Preallocate various things for the stats (as many will have jj as a dimension)
        % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
        for ff=1:numFnsToEvaluate
            if (fnlogbase(ff)==0 && (~has1 || logonlyBase(ff,1+rr))) || (fnlogbase(ff)>0 && ~any(whichcombos(ff,:,1+rr))) % as above
                continue
            end
            wsP=max(reshape(whichstatsArr(ff,:,1+rr,:),[ngroups,7]),[],1); % as above, for this restriction
            ws4=reshape(whichstatsArr(ff,:,1+rr,4),[1,ngroups]);
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



%% Create a different 'Values' for each of the variable to be evaluated
if N_z==0
    % StationaryDistVec=reshape(StationaryDist,[N_a,N_j]);
    %
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    % a_gridvals=CreateGridvals(n_a,a_grid,1);

    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'


        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            if ~any(whichcombos(ff,kk,:)) % no combination of this (function, age group) is wanted
                continue
            end
            Values=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end

            Values=reshape(Values,[N_a*(jend-j1+1),1]);
            if fnlogbase(ff)>0 % a log row (simoptions.logwhichcombos): the statistics are of the log of the function
                usedmass_kk=false(N_a*(jend-j1+1),1,'gpuArray'); % the points with positive mass in the populations used
                if whichcombos(ff,kk,1)==1
                    usedmass_kk=usedmass_kk | (StationaryDistVec_kk>0);
                end
                if useCondlRest==1
                    for rr=1:length(CondlRestnFnNames)
                        if whichcombos(ff,kk,1+rr)==1
                            usedmass_kk=usedmass_kk | reshape(StationaryDist(:,j1:jend).*RestrictionMask{rr}(:,j1:jend)>0,[N_a*(jend-j1+1),1]);
                        end
                    end
                end
                if any(Values(usedmass_kk)<=0)
                    error(['LifeCycleProfiles_FHorz_Case1: simoptions.logwhichcombos asks for the log of ',FnsToEvalNames{fnlogbase(ff)},' at the age group starting at age ',num2str(j1),', but it is not positive on some points with positive mass (use a conditional restriction, or a floor in the FnsToEvaluate)'])
                end
                Values(Values<=0)=1; % points with no mass in any population used (their weight is zero), log(1)=0 keeps them finite
                Values=log(Values);
            end

            if useCondlRest==1
                % The conditional restrictions only change the weights, so sort Values once and reuse the sort for the unrestricted stats and for every restriction
                % (only the points with positive mass are sorted; every restricted distribution is zero outside these)
                positivemass_kk=(StationaryDistVec_kk>0);
                [SortedValues_kk,SortedValues_index_kk]=sort(Values(positivemass_kk));
                SortedStationaryDistVec_kk=StationaryDistVec_kk(positivemass_kk);
                SortedStationaryDistVec_kk=SortedStationaryDistVec_kk(SortedValues_index_kk);
            end
            if whichcombos(ff,kk,1)==1 % the unrestricted stats of this (function, age group) are wanted
            ws=reshape(whichstatsArr(ff,kk,1,:),[1,7]);
            if useCondlRest==0
                tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws);
            else
                tempStats=StatsFromWeightedGrid(SortedValues_kk,SortedStationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws);
            end

            % Store them in AgeConditionalStats
            if ws(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if ws(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                if ws(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                end
            end
            if ws(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if ws(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                if ws(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                end
            end
            if ws(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if ws(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
            end
            if ws(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            end
            end % whichcombos(ff,kk,1)

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditional restrictions:
                % Only change is to use the restricted distribution (the mask times the agent distribution, normalised) as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if whichcombos(ff,kk,1+rr)==1 && sum(restrictedsamplemass(rr,j1:jend))>0 % this restriction is wanted for this (function, age group) and has mass
                        wsr=reshape(whichstatsArr(ff,kk,1+rr,:),[1,7]);
                        RestrictedStationaryDistVec_kk=StationaryDist(:,j1:jend).*RestrictionMask{rr}(:,j1:jend)/sum(restrictedsamplemass(rr,j1:jend)); % the restricted mass at each point of the age group, normalised to mass one over the group (so each age of the group is weighted by its restricted mass)
                        RestrictedStationaryDistVec_kk=RestrictedStationaryDistVec_kk(positivemass_kk);
                        tempStats=StatsFromWeightedGrid(SortedValues_kk,RestrictedStationaryDistVec_kk(SortedValues_index_kk),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,wsr);

                        % Store them in AgeConditionalStats
                        if wsr(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
                        end
                        if wsr(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                            if wsr(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if wsr(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
                        end
                        if wsr(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                            if wsr(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                            end
                        end
                        if wsr(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
                        end
                        if wsr(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
                        end
                        if wsr(7)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
                        end
                    end
                end
            end

        end
    end

else
    %% N_z
    % StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);
    %
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    % a_gridvals=CreateGridvals(n_a,a_grid,1);

    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*N_z*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'

        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            if ~any(whichcombos(ff,kk,:)) % no combination of this (function, age group) is wanted
                continue
            end
            Values=nan(N_a,N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            end

            Values=reshape(Values,[N_a*N_z*(jend-j1+1),1]);
            if fnlogbase(ff)>0 % a log row (simoptions.logwhichcombos): the statistics are of the log of the function
                usedmass_kk=false(N_a*N_z*(jend-j1+1),1,'gpuArray'); % the points with positive mass in the populations used
                if whichcombos(ff,kk,1)==1
                    usedmass_kk=usedmass_kk | (StationaryDistVec_kk>0);
                end
                if useCondlRest==1
                    for rr=1:length(CondlRestnFnNames)
                        if whichcombos(ff,kk,1+rr)==1
                            usedmass_kk=usedmass_kk | reshape(StationaryDist(:,j1:jend).*RestrictionMask{rr}(:,j1:jend)>0,[N_a*N_z*(jend-j1+1),1]);
                        end
                    end
                end
                if any(Values(usedmass_kk)<=0)
                    error(['LifeCycleProfiles_FHorz_Case1: simoptions.logwhichcombos asks for the log of ',FnsToEvalNames{fnlogbase(ff)},' at the age group starting at age ',num2str(j1),', but it is not positive on some points with positive mass (use a conditional restriction, or a floor in the FnsToEvaluate)'])
                end
                Values(Values<=0)=1; % points with no mass in any population used (their weight is zero), log(1)=0 keeps them finite
                Values=log(Values);
            end
            if useCondlRest==1
                % The conditional restrictions only change the weights, so sort Values once and reuse the sort for the unrestricted stats and for every restriction
                % (only the points with positive mass are sorted; every restricted distribution is zero outside these)
                positivemass_kk=(StationaryDistVec_kk>0);
                [SortedValues_kk,SortedValues_index_kk]=sort(Values(positivemass_kk));
                SortedStationaryDistVec_kk=StationaryDistVec_kk(positivemass_kk);
                SortedStationaryDistVec_kk=SortedStationaryDistVec_kk(SortedValues_index_kk);
            end
            if whichcombos(ff,kk,1)==1 % the unrestricted stats of this (function, age group) are wanted
            ws=reshape(whichstatsArr(ff,kk,1,:),[1,7]);
            if useCondlRest==0
                tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws);
            else
                tempStats=StatsFromWeightedGrid(SortedValues_kk,SortedStationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws);
            end

            % Store them in AgeConditionalStats
            if ws(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if ws(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
            end
            if ws(1)==1 && ws(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
            end
            if ws(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if ws(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                if ws(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                end
            end
            if ws(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if ws(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
            end
            if ws(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            end
            end % whichcombos(ff,kk,1)


            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditional restrictions:
                % Only change is to use the restricted distribution (the mask times the agent distribution, normalised) as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if whichcombos(ff,kk,1+rr)==1 && sum(restrictedsamplemass(rr,j1:jend))>0 % this restriction is wanted for this (function, age group) and has mass
                        wsr=reshape(whichstatsArr(ff,kk,1+rr,:),[1,7]);
                        RestrictedStationaryDistVec_kk=StationaryDist(:,j1:jend).*RestrictionMask{rr}(:,j1:jend)/sum(restrictedsamplemass(rr,j1:jend)); % the restricted mass at each point of the age group, normalised to mass one over the group (so each age of the group is weighted by its restricted mass)
                        RestrictedStationaryDistVec_kk=RestrictedStationaryDistVec_kk(positivemass_kk);
                        tempStats=StatsFromWeightedGrid(SortedValues_kk,RestrictedStationaryDistVec_kk(SortedValues_index_kk),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,wsr);

                        % Store them in AgeConditionalStats
                        if wsr(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
                        end
                        if wsr(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                            if wsr(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if wsr(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
                        end
                        if wsr(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                            if wsr(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                            end
                        end
                        if wsr(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
                        end
                        if wsr(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
                        end
                        if wsr(7)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
                        end
                    end
                end
            end
        end
    end
end


%% Multi-period sums/averages (simoptions.ntimeperiods with M>1): the statistics of W_j=x_j+...+x_{j+M-1} by start age j
% W_j depends on the path of the states from age j to j+M-1, so its distribution is that of a list of (state, partial sum, mass):
% at the start age the list is the agent distribution at j (one entry per state with positive mass, partial sum x_j, one mass
% column per population used: unrestricted, and each conditional restriction evaluated at j), and each age it is extended along
% every transition of the age (the policy step: a', and semiz' with gridinterplayer weights; times each z' with pi_z; times each
% e' with the next age's pi_e), adding x at the new state to the partial sum. Entries with the same state and the same partial
% sum have the same future, so they are merged (their masses added): the list is exact, there is no grid for the sums. After
% M-1 extensions the partial sums are W_j with their masses, and every statistic is that of these weighted values. The lists
% of all start ages and functions in flight are extended age by age, so each age's policy step is built once.
if nW>0
    if (isfield(simoptions,'experienceasset') && simoptions.experienceasset>=1) || (isfield(simoptions,'inheritanceasset') && simoptions.inheritanceasset==1) || (isfield(simoptions,'residualasset') && simoptions.residualasset==1)
        error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods with M>1 is not yet implemented with experience, inheritance or residual assets, ask on forum if you need this')
    end
    % The wanted combinations: level (whichcombosW) and log (wcLogW), by (function, start age, page, window length)
    wantW=(whichcombosW+wcLogW)>0;
    Mstop=zeros(numFnsBase,N_j); % the longest window wanted at each (function, start age)
    for lw=1:nW
        Mstop=max(Mstop,windowM(lw)*reshape(any(wantW(:,:,:,lw),3),[numFnsBase,N_j]));
    end
    pagesW=any(wantW,4); % [numFnsBase,N_j,nwhichpages]: the pages wanted at each (function, start age)

    % Preallocate the outputs of the windows: fields only for the (row, page) selected at some start age
    WNames=cell(numFnsBase,nW,2); % the output names: level and log
    for ff=1:numFnsBase
        for lw=1:nW
            WNames{ff,lw,1}=[FnsToEvalNames{ff},'_',num2str(windowM(lw)),windowstr];
            WNames{ff,lw,2}=[FnsToEvalNames{ff},'_',num2str(windowM(lw)),windowstr,'_log'];
            for kind=1:2
                for pp=1:nwhichpages
                    if kind==1
                        selW=reshape(whichcombosW(ff,:,pp,lw),[1,N_j]);
                    else
                        selW=reshape(wcLogW(ff,:,pp,lw),[1,N_j]);
                    end
                    if ~any(selW)
                        continue
                    end
                    wsP=max(reshape(whichstatsW(ff,selW==1,pp,:,lw),[sum(selW),7]),[],1); % a statistic is preallocated if a selected start age asks for it
                    ws4=reshape(whichstatsW(ff,selW==1,pp,4,lw),[1,sum(selW)]);
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
                    if pp==1
                        AgeConditionalStats.(WNames{ff,lw,kind})=SS;
                    else
                        AgeConditionalStats.(CondlRestnFnNames{pp-1}).(WNames{ff,lw,kind})=SS;
                    end
                end
            end
        end
    end

    % The shocks, separately: (semiz,z,e) [whichever are present], the states are ordered (a,semiz,z,e)
    N_zW=prod(n_zW);
    N_e=prod(simoptionsW.n_e);
    N_semiz=prod(simoptionsW.n_semiz);
    N_semizr=max(N_semiz,1);
    N_zr=max(N_zW,1);
    N_er=max(N_e,1);
    N_states=N_a*N_semizr*N_zr*N_er;
    if N_zW>0 || N_e>0
        if N_zW>0 && ~isfield(simoptionsW,'pi_z') && ~isfield(simoptionsW,'ExogShockFn')
            error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods with M>1 needs the transition probabilities of z, give them as simoptions.pi_z (the same as the pi_z input of StationaryDist_FHorz_Case1)')
        end
        if isfield(simoptionsW,'pi_z')
            pi_zW=simoptionsW.pi_z;
        else
            pi_zW=[];
        end
        [~,pi_z_J,simoptionsW]=ExogShockSetup_FHorz(n_zW,z_grid,pi_zW,N_j,Parameters,simoptionsW,2,0);
        if N_zW>0
            pi_z_J_gpu=gpuArray(pi_z_J);
        end
        if N_e>0
            pi_e_J_gpu=gpuArray(simoptionsW.pi_e_J);
        end
    end
    if N_semiz>0
        if ~isfield(simoptionsW,'l_dsemiz')
            simoptionsW.l_dsemiz=1; % by default, just one decision variable is used for the semi-exo state
        end
        simoptionsW=SemiExogShockSetup_FHorz(n_d,N_j,d_grid,Parameters,simoptionsW,2);
    end

    % The policy step: the indexes and weights of a' (and of the semi-exogenous transition) that each age's sparse matrix is built from
    if isempty(n_d) || prod(n_d)==0
        l_d=0;
    else
        l_d=length(n_d);
    end
    PolicyW=reshape(Policy,[size(Policy,1),N_a,N_semizr*N_zr*N_er,N_j]);
    if l_a==1
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:),1);
    elseif l_a==2
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1),1);
    elseif l_a==3
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyW(l_d+3,:,:,:)-1),1);
    elseif l_a==4
        Policy_aprime=shiftdim(PolicyW(l_d+1,:,:,:)+n_a(1)*(PolicyW(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyW(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyW(l_d+4,:,:,:)-1),1);
    else
        error('LifeCycleProfiles_FHorz_Case1: simoptions.ntimeperiods with M>1 cannot handle length(n_a)>4, contact me if you need this')
    end
    % Policy_aprime is [N_a,N_semizr*N_zr*N_er,N_j]
    if simoptions.gridinterplayer==1
        % two a' points per state: the lower grid point and the one above it, with the second-layer weights
        L2index=PolicyW(end-1,:,:,:); % L2 index (end-1 because end is L2flag)
        L2flag=PolicyW(end,:,:,:);
        L2index(L2flag==1)=1;                        % force all weight to lower grid point
        L2index(L2flag==3)=simoptions.ngridinterp+2; % force all weight to upper grid point
        probupper=shiftdim((L2index-1)/(simoptions.ngridinterp+1),1); % [N_a,N_semizr*N_zr*N_er,N_j]: probability of the upper grid point
    end
    if N_semiz==0
        % Add the z index of each (z,e) column (z varies first) to get the index into (a',z)
        zindex=N_a*repmat(gpuArray(0:1:N_zr-1),1,N_er);
        Policy_aprimez=Policy_aprime+zindex; % [N_a,N_zr*N_er,N_j], index into (a',z)
        if simoptions.gridinterplayer==0
            IIind=gpuArray((1:1:N_states)');
        elseif simoptions.gridinterplayer==1
            IIind=gpuArray(repmat((1:1:N_states)',2,1));
        end
    else
        % Semi-exogenous state (as StationaryDist_FHorz_Iteration_SemiExo_raw): from (a,semiz,z,e) to (a',semiz',z) with the
        % transition probabilities pi_semiz_J(semiz,semiz',dsemiz,jj) of the semi-exogenous decision dsemiz taken at the state
        % (only the N_semizshort largest entries of each row of pi_semiz_J are kept, which is all of the nonzeros)
        l_d1=l_d-simoptionsW.l_dsemiz; % the last l_dsemiz decision variables are the ones that influence semiz
        N_dsemiz=prod(n_d(l_d1+1:l_d));
        if simoptionsW.l_dsemiz==1
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:);
        elseif simoptionsW.l_dsemiz==2
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:)+n_d(l_d1+1)*(PolicyW(l_d1+2,:,:,:)-1);
        elseif simoptionsW.l_dsemiz==3
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:)+n_d(l_d1+1)*(PolicyW(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(PolicyW(l_d1+3,:,:,:)-1);
        elseif simoptionsW.l_dsemiz==4
            Policy_dsemiexo=PolicyW(l_d1+1,:,:,:)+n_d(l_d1+1)*(PolicyW(l_d1+2,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*(PolicyW(l_d1+3,:,:,:)-1)+n_d(l_d1+1)*n_d(l_d1+2)*n_d(l_d1+3)*(PolicyW(l_d1+4,:,:,:)-1);
        end
        Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_states,1,N_j]));
        N_semizshort=max(max(max(sum((simoptionsW.pi_semiz_J>0),2))));
        [pi_semiz_J_short,idx]=sort(simoptionsW.pi_semiz_J,2); % puts the zeros on the left
        pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:)); % [N_semiz,N_semizshort,N_dsemiz,N_j-1]
        idxshort=gather(idx(:,end-N_semizshort+1:end,:,:)); % the semiz' each kept entry belongs to
        semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_zr*N_er,1)+N_semiz*(0:1:N_semizshort-1); % [N_states,N_semizshort]: the semiz of each state, offset to each column of pi_semiz_J_short
        zprimeoffset=repmat(repelem(N_a*N_semiz*(0:1:N_zr-1)',N_a*N_semiz,1),N_er,1); % [N_states,1]: the z of each state, as an offset into (a',semiz',z)
        Policy_aprime_cpu=gather(reshape(Policy_aprime,[N_states,1,N_j]));
        if simoptions.gridinterplayer==0
            II2=repelem((1:1:N_states)',1,N_semizshort);
        elseif simoptions.gridinterplayer==1
            Policy_aprime2_cpu=cat(2,Policy_aprime_cpu,Policy_aprime_cpu+1); % [N_states,2,N_j]: the lower grid point and the one above it
            probupper_cpu=gather(reshape(probupper,[N_states,1,N_j]));
            PolicyProbs_cpu=cat(2,1-probupper_cpu,probupper_cpu); % [N_states,2,N_j]
            II2=repelem((1:1:N_states)',1,2*N_semizshort);
        end
    end

    % The values of each function with a window wanted, at every age
    ValuesW=cell(numFnsBase,1);
    for ff=1:numFnsBase
        if ~any(Mstop(ff,:))
            continue
        end
        ValuesW{ff}=nan(N_states,N_j,'gpuArray');
        for jj=1:N_j
            FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
            if N_z==0
                ValuesW{ff}(:,jj)=reshape(EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]),[N_states,1]);
            else
                ValuesW{ff}(:,jj)=reshape(EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj)),[N_states,1]);
            end
        end
    end

    % The lists in flight: function, start age, pages (with positive start mass), and the entries (state, partial sum, mass per page)
    Lff=zeros(1,0);
    Lj=zeros(1,0);
    Lpages=cell(1,0);
    Lst=cell(1,0);
    LS=cell(1,0);
    Lm=cell(1,0);
    for jj=1:N_j
        % (i) Start the lists of start age jj
        massjj=sum(StationaryDist(:,jj));
        for ff=1:numFnsBase
            if Mstop(ff,jj)==0 || massjj==0
                continue
            end
            pagesL=find(reshape(pagesW(ff,jj,:),[1,nwhichpages]));
            m0=zeros(N_states,length(pagesL),'gpuArray');
            for pp=1:length(pagesL)
                if pagesL(pp)==1
                    m0(:,pp)=StationaryDist(:,jj);
                else
                    m0(:,pp)=StationaryDist(:,jj).*RestrictionMask{pagesL(pp)-1}(:,jj); % the restriction is evaluated at the start age
                end
            end
            colmass=gather(sum(m0,1));
            keep=(colmass>0); % a page with no mass at the start age stays NaN
            if ~any(keep)
                continue
            end
            m0=m0(:,keep)./colmass(keep);
            pagesL=pagesL(keep);
            st=find(any(m0>0,2));
            Lff=[Lff,ff]; %#ok<AGROW>
            Lj=[Lj,jj]; %#ok<AGROW>
            Lpages=[Lpages,{pagesL}]; %#ok<AGROW>
            Lst=[Lst,{st}]; %#ok<AGROW>
            LS=[LS,{ValuesW{ff}(st,jj)}]; %#ok<AGROW>
            Lm=[Lm,{m0(st,:)}]; %#ok<AGROW>
        end

        % (ii) Read the windows that end at age jj: a list of start age j has the M-period sums for M=jj-j+1
        for ll=1:length(Lff)
            ff=Lff(ll);
            j0=Lj(ll);
            lw=find(windowM==jj-j0+1);
            if isempty(lw)
                continue
            end
            M=windowM(lw);
            for pp=1:length(Lpages{ll})
                pg=Lpages{ll}(pp);
                for kind=1:2
                    if (kind==1 && whichcombosW(ff,j0,pg,lw)==0) || (kind==2 && wcLogW(ff,j0,pg,lw)==0)
                        continue
                    end
                    w=Lm{ll}(:,pp);
                    pos=(w>0);
                    vals=LS{ll}(pos);
                    w=w(pos);
                    w=w/sum(w);
                    if strcmp(simoptions.ntimeperiodstype,'average')
                        vals=vals/M;
                    end
                    if kind==2
                        if any(vals<=0)
                            error(['LifeCycleProfiles_FHorz_Case1: simoptions.logwhichcombos asks for the log of the ',num2str(M),'-period sum/average of ',FnsToEvalNames{ff},' starting at age ',num2str(j0),', but it is not positive for some agents with positive mass (use a conditional restriction, or a floor in the FnsToEvaluate)'])
                        end
                        vals=log(vals);
                    end
                    ws=reshape(whichstatsW(ff,j0,pg,:,lw),[1,7]);
                    tempStats=StatsFromWeightedGrid(vals,w,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws);

                    % Store them in AgeConditionalStats
                    if pg==1
                        SS=AgeConditionalStats.(WNames{ff,lw,kind});
                    else
                        SS=AgeConditionalStats.(CondlRestnFnNames{pg-1}).(WNames{ff,lw,kind});
                    end
                    if ws(1)==1
                        SS.Mean(j0)=tempStats.Mean;
                    end
                    if ws(2)==1
                        SS.Median(j0)=tempStats.Median;
                        if ws(1)==1
                            SS.RatioMeanToMedian(j0)=tempStats.RatioMeanToMedian;
                        end
                    end
                    if ws(3)==1
                        SS.Variance(j0)=tempStats.Variance;
                        SS.StdDeviation(j0)=tempStats.StdDeviation;
                    end
                    if ws(4)>=1
                        SS.Gini(j0)=tempStats.Gini;
                        if ws(4)<3
                            SS.LorenzCurve(:,j0)=tempStats.LorenzCurve;
                        end
                    end
                    if ws(5)==1
                        SS.Minimum(j0)=tempStats.Minimum;
                        SS.Maximum(j0)=tempStats.Maximum;
                    end
                    if ws(6)>=1
                        SS.QuantileCutoffs(:,j0)=tempStats.QuantileCutoffs;
                        SS.QuantileMeans(:,j0)=tempStats.QuantileMeans;
                    end
                    if ws(7)==1
                        SS.MoreInequality.Top1share(j0)=tempStats.MoreInequality.Top1share;
                        SS.MoreInequality.Top5share(j0)=tempStats.MoreInequality.Top5share;
                        SS.MoreInequality.Top10share(j0)=tempStats.MoreInequality.Top10share;
                        SS.MoreInequality.Bottom50share(j0)=tempStats.MoreInequality.Bottom50share;
                        SS.MoreInequality.Percentile50th(j0)=tempStats.MoreInequality.Percentile50th;
                        SS.MoreInequality.Percentile90th(j0)=tempStats.MoreInequality.Percentile90th;
                        SS.MoreInequality.Percentile95th(j0)=tempStats.MoreInequality.Percentile95th;
                        SS.MoreInequality.Percentile99th(j0)=tempStats.MoreInequality.Percentile99th;
                    end
                    if pg==1
                        AgeConditionalStats.(WNames{ff,lw,kind})=SS;
                    else
                        AgeConditionalStats.(CondlRestnFnNames{pg-1}).(WNames{ff,lw,kind})=SS;
                    end
                end
            end
        end

        % (iii) Drop the lists whose longest window has been read
        done=false(1,length(Lff));
        for ll=1:length(Lff)
            if jj-Lj(ll)+1>=Mstop(Lff(ll),Lj(ll))
                done(ll)=true;
            end
        end
        Lff=Lff(~done);
        Lj=Lj(~done);
        Lpages=Lpages(~done);
        Lst=Lst(~done);
        LS=LS(~done);
        Lm=Lm(~done);
        if isempty(Lff)
            continue
        end

        % (iv) Extend every list in flight from age jj to jj+1
        % The policy step of age jj: maps a measure over the age-jj states (a,semiz,z,e) to a measure over (a',semiz',z) [z kept, e summed out]
        if N_semiz==0
            if simoptions.gridinterplayer==0
                Gammatranspose_jj=sparse(reshape(Policy_aprimez(:,:,jj),[],1),IIind,ones(N_states,1,'gpuArray'),N_a*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                Policy_aprimez_jj=reshape(Policy_aprimez(:,:,jj),[],1); % the lower grid point of each state (the upper one is the next index)
                probupper_jj=reshape(probupper(:,:,jj),[],1);
                Gammatranspose_jj=sparse([Policy_aprimez_jj;Policy_aprimez_jj+1],IIind,[1-probupper_jj;probupper_jj],N_a*N_zr,N_states); % sparse gpuArray
            end
        else
            semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % linear index into pi_semiz_J_short and idxshort
            if simoptions.gridinterplayer==0
                Policy_aprimesemizz_jj=repelem(Policy_aprime_cpu(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1)+zprimeoffset; % [N_states,N_semizshort]: index into (a',semiz',z)
                Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(pi_semiz_J_short(semizindex_short_jj)),N_a*N_semiz*N_zr,N_states); % sparse gpuArray
            elseif simoptions.gridinterplayer==1
                Policy_aprimesemizz_jj=repelem(Policy_aprime2_cpu(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,2)+zprimeoffset; % [N_states,2*N_semizshort]: the two grid points, each with every semiz'
                PolicyProbs_jj=repelem(PolicyProbs_cpu(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,2);
                Gammatranspose_jj=sparse(gpuArray(Policy_aprimesemizz_jj),gpuArray(II2),gpuArray(PolicyProbs_jj),N_a*N_semiz*N_zr,N_states); % sparse gpuArray; sparse() accumulates repeated indexes (the two grid points coincide only at the top of the grid)
            end
        end
        % Its nonzeros, grouped by the age-jj state (src): eG_ptr(s) is the number of nonzeros of the states before s, eG_cnt(s) those of s
        [eG_row,eG_src,eG_val]=find(Gammatranspose_jj); % the (a',semiz',z) index, the age-jj state, the probability
        [eG_src,ord]=sort(eG_src);
        eG_row=eG_row(ord);
        eG_val=eG_val(ord);
        eG_as=mod(eG_row-1,N_a*N_semizr)+1; % the (a',semiz') part
        eG_zr=floor((eG_row-1)/(N_a*N_semizr))+1; % the age-jj z (kept by the policy step)
        eG_cnt=accumarray(eG_src,1,[N_states,1]);
        eG_ptr=cumsum(eG_cnt)-eG_cnt;
        % The offsets of each (z',e') of age jj+1, and its probability
        destoffset=N_a*N_semizr*(0:1:N_zr-1)+reshape(N_a*N_semizr*N_zr*(0:1:N_er-1),[1,1,N_er]); % 1 x N_zr x N_er
        if N_e>0
            pe=reshape(pi_e_J_gpu(:,jj+1),[1,1,N_er]);
        else
            pe=1;
        end
        for ll=1:length(Lff)
            ff=Lff(ll);
            st=Lst{ll};
            % every entry times every nonzero of its state's policy step
            c=eG_cnt(st);
            s0=cumsum(c)-c+1; % the first row of each entry (every state has at least one nonzero)
            ntot=gather(sum(c));
            mark=zeros(ntot,1,'gpuArray');
            mark(s0)=1;
            ie=cumsum(mark); % the entry of each row
            g=eG_ptr(st(ie))+((1:1:ntot)'-s0(ie)+1); % the nonzero of each row
            if N_zW>0
                P=eG_val(g).*pi_z_J_gpu(eG_zr(g),:,jj).*pe; % ntot x N_zr x N_er: the probability of each (a',semiz',z',e')
            else
                P=eG_val(g).*pe;
            end
            D=eG_as(g)+destoffset; % ntot x N_zr x N_er: the age-jj+1 state
            P=P(:);
            D=D(:);
            ie=repmat(ie,N_zr*N_er,1);
            keep=(P>0);
            P=P(keep);
            D=D(keep);
            ie=ie(keep);
            newm=Lm{ll}(ie,:).*P;
            newS=LS{ll}(ie)+ValuesW{ff}(D,jj+1);
            if jj+1-Lj(ll)+1<Mstop(ff,Lj(ll))
                % merge the entries with the same state and partial sum (the list is extended again)
                [B,ord]=sortrows([D,newS]);
                first=[true;any(B(2:end,:)~=B(1:end-1,:),2)];
                grp=cumsum(first);
                newm=newm(ord,:);
                ngrp=gather(grp(end));
                mm=zeros(ngrp,size(newm,2),'gpuArray');
                for pp=1:size(newm,2)
                    mm(:,pp)=accumarray(grp,newm(:,pp),[ngrp,1]);
                end
                Lst{ll}=B(first,1);
                LS{ll}=B(first,2);
                Lm{ll}=mm;
            else
                % the next age is the last of this list: only the partial sums and masses are read
                Lst{ll}=D;
                LS{ll}=newS;
                Lm{ll}=newm;
            end
        end
    end
end






end


