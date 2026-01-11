function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
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
% The stationary distribution be a structure and will contain both the
% weights/distribution across the permenant types, as well as a pdf for the
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

computeForThesei=ones(N_i,1); % Used to omit the infinite horizon PTypes from computations (a message is printed to say they are being ignored)

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
    simoptions.lowmemory=0; % =1 is slow, but less memory demanding (only use if groupusingtdigest is still not enough to run codes)
        % lowmemory=0 has outerloop over ptype and inner loop of fnstoeval; lowmemory=1 has outerloop over fnstoeval, inner loop over ptype
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    defaultagegroupings=1;
    if isstruct(N_j)
        N_j_max=0;
        for ii=1:N_i
            if isfinite(N_j.(Names_i{ii}))
                simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
                N_j_max=max(N_j_max,N_j.(Names_i{ii}));
            else % Infinite horizon
                computeForThesei(ii)=0;
            end
        end
    else
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
        N_j_max=N_j;
    end
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
    simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
    simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    simoptions.groupusingtdigest=0; % if you are ptypestorecpu=1 and groupptypesforstats=1, you might also need to use groupusingtdigest=1 if you get out of memory errors
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.gridinterplayer=0;
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'lowmemory')
        simoptions.lowmemory=0; % =1 is slow, but less memory demanding (only use if groupusingtdigest=1 is still not enough to run codes)
            % lowmemory=0 has outerloop over ptype and inner loop of fnstoeval; lowmemory=1 has outerloop over fnstoeval, inner loop over ptype
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=100;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=100;
    end
    if isfield(simoptions,'agegroupings')==0
        defaultagegroupings=1;
        if isstruct(N_j)
            N_j_max=0;
            for ii=1:N_i
                if isfinite(N_j.(Names_i{ii}))
                    simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
                    N_j_max=max(N_j_max,N_j.(Names_i{ii}));
                else % Infinite horizon
                    computeForThesei(ii)=0;
                end
            end
        else
            simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
            N_j_max=N_j;
        end
    else
        defaultagegroupings=0;
        N_j_max=length(simoptions.agegroupings);
    end
    if isfield(simoptions,'nquantiles')==0
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'npoints')==0
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    elseif simoptions.npoints==0
        error('simoptions.npoints must be a positive (non-zero) integer')
    end
    if isfield(simoptions,'tolerance')==0    
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if isfield(simoptions,'agejshifter')==0
        simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
    end
    if ~isfield(simoptions,'whichstats')
        if ~isstruct(N_j)
            if any(simoptions.agegroupings(2:end)-simoptions.agegroupings(1:end-1)>4)
                % if some agegroupings are 'large', use the slower but lower memory versions
                simoptions.whichstats=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
            else
                simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
            end
        else
            for ii=1:N_i
                if isfinite(N_j.(Names_i{ii}))
                    temp=simoptions.agegroupings.(Names_i{ii});
                    if any(temp(2:end)-temp(1:end-1)>4)
                        % if some agegroupings are 'large', use the slower but lower memory versions
                        simoptions.whichstats.(Names_i{ii})=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
                    else
                        simoptions.whichstats.(Names_i{ii})=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
                    end
                else % Infinite horizon
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
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
end

%% Drop anything that is infinite horizon and print out a message to say so
if any(computeForThesei==0)
    N_i=sum(computeForThesei);
    Names_i2=Names_i;
    Names_i=cell(N_i,1);
    ii=0;
    for ii2=1:length(computeForThesei)
        if computeForThesei(ii2)==1
            ii=ii+1;
            Names_i{ii}=Names_i2{ii2};
        else % tell the user about it
            fprintf(['LifeCycleProfiles_FHorz_Case1_PType: Ignoring the ', num2str(ii2), '-th PType, ',Names_i2{ii2}, ' because it is infinite horizon \n']);
        end
    end   
    % Eliminate any no longer relevant functions from FnsToEvaluate (those which are only used for infinite horizon)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    FnsToEvaluate2=FnsToEvaluate;
    clear FnsToEvaluate
    for ff=1:length(fieldnames(FnsToEvaluate2))
        if isstruct(FnsToEvaluate2.(FnsToEvalNames{ff}))
            for ii=1:N_i
                if isfield(FnsToEvaluate2.(FnsToEvalNames{ff}),Names_i{ii})
                    FnsToEvaluate.(FnsToEvalNames{ff}).(Names_i{ii})=FnsToEvaluate2.(FnsToEvalNames{ff}).(Names_i{ii});
                end
            end
        else % Relevant to all the PTypes
            FnsToEvaluate.(FnsToEvalNames{ff})=FnsToEvaluate2.(FnsToEvalNames{ff});
        end
    end
    % Done. Because from here on we just use N_i and Names_i which now only
    % contain finite horizons. Note that it is anyway only possible to use
    % a mixture of infinite and finite horizon if you are explictly using
    % Names_i. So don't need to worry about vectors over N_i being the
    % wrong size.
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
maxngroups=max(ngroups(isfinite(ngroups)));
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
            if isfinite(N_j.(Names_i{ii}))
                N_j_max2=max(N_j_max2,simoptions.agejshifter(ii)+N_j.(Names_i{ii}));
            end
        end
    end
elseif isscalar(simoptions.agejshifter) % not using agejshifter
    simoptions.agejshifter=zeros(N_i,1);
    N_j_max2=N_j_max;
else % have inputed as a vector
    simoptions.agejshifter=simoptions.agejshifter-min(simoptions.agejshifter); % put them all relative to the minimum
    if isstruct(N_j)
        N_j_max2=0;
        for ii=1:N_i
            if isfinite(N_j.(Names_i{ii}))
                N_j_max2=max(N_j_max2,simoptions.agejshifter(ii)+N_j.(Names_i{ii}));
            end
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
AgeConditionalStats=struct();

% Preallocate
if simoptions.lowmemory==0
    if simoptions.groupusingtdigest==1 % Setupt things for t-Digest
        % Following few lines relate to the digest
        delta=10000;
        merge_nsofar=zeros(maxngroups,numFnsToEvaluate); % Keep count
        merge_nsofar2=zeros(maxngroups,numFnsToEvaluate); % Keep count

        AllCMerge=struct();
        Alldigestweightsmerge=struct();
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            for jj=1:maxngroups
                AllCMerge.(FnsToEvalNames{ff}).(jgroupstr{jj})=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
                Alldigestweightsmerge.(FnsToEvalNames{ff}).(jgroupstr{jj})=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
            end
        end
    else % Or we will just store the unique of grid and weights
        AllValues=struct();
        AllWeights=struct();
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            for jj=1:maxngroups
                AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj})=[];
                AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj})=[];
            end
        end
    end
% elseif simoptions.lowmemory=1, is done later (inside the outer loop over FnsToEvaluate)
end

FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');



%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(N_i,N_j,length(CondlRestnFnNames));
    % RestrictionStruct_ii=struct();

    if simoptions.groupusingtdigest==1 % Things are being stored on cpu but solved on gpu
        error('Have not implemented simoptions.groupusingtdigest==1 together with simoptions.conditionalrestrictions')
    else
        AllRestrictedWeights=struct(); % Only used if useCondlRest==1
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            for rr=1:length(CondlRestnFnNames)
                for jj=1:maxngroups
                    AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(jgroupstr{jj})=[];
                end
            end
        end
    end

    % Preallocate various things for the stats (as many will have jj as a dimension)
    % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
    for ff=1:numFnsToEvaluate
        for rr=1:length(CondlRestnFnNames)
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(1)==1 && simoptions.whichstats(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(3)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(4)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
                if simoptions.whichstats(4)<3
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
                end
            end
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(6)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(7)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
            end
        end
    end
end

AgeMasses=zeros(N_i,N_j_max,'gpuArray'); % Only ends up used if using simoptions.conditionalrestrictions
if useCondlRest==1 && isstruct(N_j)
    error('LifeCycleProfiles: Have not implemented combination of doing conditional restrictions with N_j being a structure (differing across agents)')
end

%% Do an outerloop over ptypes and an inner loop over FnsToEvaluate
if simoptions.lowmemory==0
    for ii=1:N_i

        % First set up simoptions
        simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted

        if simoptions_temp.verbose==1
            fprintf('Permanent type: %i of %i \n',ii, N_i)
        end
        if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
            PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii})); % Essentially just assuming vfoptions.ptypestorecpu=1 as well
            % StationaryDist_temp=gpuArray(StationaryDist.(Names_i{ii}));
        else
            PolicyIndexes_temp=Policy.(Names_i{ii});
            % StationaryDist_temp=StationaryDist.(Names_i{ii});
        end

        % Go through everything which might be dependent on permanent type (PType)
        % Notice that the way this is coded the grids (etc.) could be either
        % fixed, or a function (that depends on age, and possibly on permanent
        % type), or they could be a structure. Only in the case where they are
        % a structure is there a need to take just a specific part and send
        % only that to the 'non-PType' version of the command.

        if isstruct(n_d)
            n_d_temp=n_d.(Names_i{ii});
        else
            n_d_temp=n_d;
        end
        if isstruct(n_a)
            n_a_temp=n_a.(Names_i{ii});
        else
            n_a_temp=n_a;
        end
        if isstruct(n_z)
            n_z_temp=n_z.(Names_i{ii});
        else
            n_z_temp=n_z;
        end
        if isstruct(N_j)
            N_j_temp=N_j.(Names_i{ii});
        else
            N_j_temp=N_j;
        end
        if isstruct(d_grid)
            d_grid_temp=d_grid.(Names_i{ii});
        else
            d_grid_temp=d_grid;
        end
        if isstruct(a_grid)
            a_grid_temp=a_grid.(Names_i{ii});
        else
            a_grid_temp=a_grid;
        end
        if isstruct(z_grid)
            z_grid_temp=z_grid.(Names_i{ii});
        else
            z_grid_temp=z_grid;
        end

        % Parameters are allowed to be given as structure, or as vector/matrix
        % (in terms of their dependence on permanent type). So go through each of
        % these in term.
        Parameters_temp=Parameters;
        FullParamNames=fieldnames(Parameters);
        nFields=length(FullParamNames);
        for kField=1:nFields
            if isstruct(Parameters.(FullParamNames{kField})) % Check for permanent type in structure form
                names=fieldnames(Parameters.(FullParamNames{kField}));
                for jj=1:length(names)
                    if strcmp(names{jj},Names_i{ii})
                        Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{jj});
                    end
                end
            elseif any(size(Parameters.(FullParamNames{kField}))==N_i) % Check for permanent type in vector/matrix form.
                temp=Parameters.(FullParamNames{kField});
                [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
                if ptypedim==1
                    Parameters_temp.(FullParamNames{kField})=temp(ii,:);
                elseif ptypedim==2
                    Parameters_temp.(FullParamNames{kField})=temp(:,ii);
                end
            end
        end

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

        StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

        AgeMasses(ii,simoptions.agejshifter(ii)+(1:N_j_temp))=sum(StationaryDist_ii,1); % I think this is right, but haven't tested yet
        % AgeMasses(ii,:)=sum(StationaryDist_ii,1);

        %% Evaluate conditional restrictions for this PType (note: these use simoptions not simoptions_temp)
        if useCondlRest==1
            RestrictionStruct_ii=struct();

            l_daprime_temp=size(PolicyValues_temp,1);
            permuteindexes=[1+(1:1:(l_a_temp+l_z_temp)),1];
            PolicyValuesPermute_temp=permute(PolicyValues_temp,permuteindexes); %[n_a,n_z,l_d+l_a]

            % For each conditional restriction, create a 'restricted stationary distribution'
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

                RestrictedStationaryDistVec=StationaryDist_ii;
                RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction

                % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
                restrictedsamplemass(ii,:,rr)=sum(RestrictedStationaryDistVec,1);
                RestrictedStationaryDistVec=RestrictedStationaryDistVec./(restrictedsamplemass(ii,:,rr)+(restrictedsamplemass(ii,:,rr)==0)); % Normalize to mass of 1 [divides by 1 when the mass is zero, which will return zero (as is essentially doing zero/one)]
                % Store for later
                RestrictionStruct_ii(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec;

                AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.(Names_i{ii})=restrictedsamplemass(ii,:,rr); % Seems likely this would be something user might want

            end
        end
        
        
        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid

            if FnsAndPTypeIndicator_ii(ff)==1 % If this function is relevant to this ptype
                
                % Get parameter names for current FnsToEvaluate functions
                if isstruct(FnsToEvaluate.(FnsToEvalNames{ff}))
                    tempfn=FnsToEvaluate.(FnsToEvalNames{ff}).(Names_i{ii});
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
                % StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

                % Note, eliminating zero weights and unique() cannot be done yet as they need to be conditional on j
                % (otherwise lose the j dimension if I just apply them now)

                % Preallocate various things for the stats (as many will have jj as a dimension)
                % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).
                if simoptions_temp.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Mean=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions_temp.whichstats(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Median=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    if simoptions_temp.whichstats(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).RatioMeanToMedian=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    end
                end
                if simoptions_temp.whichstats(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Variance=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions_temp.whichstats(4)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Gini=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    if simoptions_temp.whichstats(4)<3
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve=nan(simoptions_temp.npoints,length(simoptions_temp.agegroupings),'gpuArray');
                    end
                end
                if simoptions_temp.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions_temp.whichstats(6)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,length(simoptions_temp.agegroupings),'gpuArray'); % Includes the min and max values
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans=nan(simoptions_temp.nquantiles,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions_temp.whichstats(7)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if useCondlRest==1
                    for rr=1:length(CondlRestnFnNames)
                        if simoptions_temp.whichstats(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Mean=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if simoptions_temp.whichstats(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Median=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            if simoptions_temp.whichstats(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).RatioMeanToMedian=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            end
                        end
                        if simoptions_temp.whichstats(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Variance=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if simoptions_temp.whichstats(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Gini=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            if simoptions.whichstats(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve=nan(simoptions_temp.npoints,length(simoptions_temp.agegroupings),'gpuArray');
                            end
                        end
                        if simoptions_temp.whichstats(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Minimum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Maximum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if simoptions_temp.whichstats(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,length(simoptions_temp.agegroupings),'gpuArray'); % Includes the min and max values
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans=nan(simoptions_temp.nquantiles,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                        if simoptions_temp.whichstats(7)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                        end
                    end
                end

                for jj=1:length(simoptions_temp.agegroupings)

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
                    Values_jj=reshape(ValuesOnGrid_ffii(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]);

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

                    SortedWeights_jj=SortedWeights_jj/sum(SortedWeights_jj(:)); % Normalize conditional on jj (is later renormalized ii weight before storing for groupstats)

                    %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and ff)
                    tempStats=StatsFromWeightedGrid(SortedValues_jj,SortedWeights_jj,simoptions_temp.npoints,simoptions_temp.nquantiles,simoptions_temp.tolerance,1,simoptions_temp.whichstats); % 1 is presorted

                    % Now store these based on jj
                    if simoptions_temp.whichstats(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Mean(jj)=tempStats.Mean;
                    end
                    if simoptions_temp.whichstats(2)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Median(jj)=tempStats.Median;
                        if simoptions_temp.whichstats(1)==1
                            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).RatioMeanToMedian(jj)=tempStats.RatioMeanToMedian;
                        end
                    end
                    if simoptions_temp.whichstats(3)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Variance(jj)=tempStats.Variance;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation(jj)=tempStats.StdDeviation;
                    end
                    if simoptions_temp.whichstats(4)>=1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Gini(jj)=tempStats.Gini;
                        if simoptions_temp.whichstats(4)<3
                            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve(:,jj)=tempStats.LorenzCurve;
                        end
                    end
                    if simoptions_temp.whichstats(5)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum(jj)=tempStats.Minimum;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum(jj)=tempStats.Maximum;
                    end
                    if simoptions_temp.whichstats(6)>=1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs(:,jj)=tempStats.QuantileCutoffs;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans(:,jj)=tempStats.QuantileMeans;
                    end
                    if simoptions_temp.whichstats(7)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share(jj)=tempStats.MoreInequality.Top1share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share(jj)=tempStats.MoreInequality.Top5share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share(jj)=tempStats.MoreInequality.Top10share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share(jj)=tempStats.MoreInequality.Bottom50share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th(jj)=tempStats.MoreInequality.Percentile50th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th(jj)=tempStats.MoreInequality.Percentile90th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th(jj)=tempStats.MoreInequality.Percentile95th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th(jj)=tempStats.MoreInequality.Percentile99th;
                    end

                    % For later, put the mean and std dev in a convenient place. These are instead done on jjageshifted (so they can be grouped across ptypes later)
                    if simoptions_temp.whichstats(1)==1
                        MeanVec(ff,ii,jjageshifted)=tempStats.Mean;
                    end
                    if simoptions_temp.whichstats(3)==1
                        StdDevVec(ff,ii,jjageshifted)=tempStats.StdDeviation;
                    end
                    % Do the same with the minimum and maximum
                    if simoptions_temp.whichstats(5)==1
                        minvaluevec(ff,ii,jjageshifted)=tempStats.Minimum;
                        maxvaluevec(ff,ii,jjageshifted)=tempStats.Maximum;
                    end

                    if simoptions.groupptypesforstats==1
                        % Store things to use later to use to calculate the grouped stats
                        if simoptions_temp.groupusingtdigest==1
                            Cmerge=AllCMerge.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted});
                            digestweightsmerge=Alldigestweightsmerge.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted});

                            %% Create digest (if unique() was not enough to make them small)
                            [C_jj,digestweights_jj,~]=createDigest(SortedValues_jj, SortedWeights_jj,delta,1); % 1 is presorted

                            merge_nsofar2(jjageshifted,ff)=merge_nsofar(jjageshifted,ff)+length(C_jj);
                            Cmerge(merge_nsofar(jjageshifted,ff)+1:merge_nsofar2(jjageshifted,ff))=C_jj;
                            digestweightsmerge(merge_nsofar(jjageshifted,ff)+1:merge_nsofar2(jjageshifted,ff))=digestweights_jj*StationaryDist.ptweights(ii);
                            merge_nsofar(jjageshifted,ff)=merge_nsofar2(jjageshifted,ff);

                            AllCMerge.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted})=Cmerge;
                            Alldigestweightsmerge.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted})=digestweightsmerge;
                        else
                            AllValues.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted})=[AllValues.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted}); SortedValues_jj];
                            AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted})=[AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jjageshifted}); SortedWeights_jj*StationaryDist.ptweights(ii)];
                        end
                    end


                    %% If using conditional restrictions, do those
                    if useCondlRest==1
                        for rr=1:length(CondlRestnFnNames)
                            if sum(restrictedsamplemass(ii,j1:jend,rr))~=0
                                % Do same to RestrictionStruct_ii(rr).RestrictedStationaryDistVec(:,jj) as was done to get SortedWeights_jj
                                RestrictedSortedWeights=RestrictionStruct_ii(rr).RestrictedStationaryDistVec(:,j1:jend);
                                RestrictedSortedWeights=RestrictedSortedWeights(~temp); % drop zeros masses (but ignoring the restrictions; this is just to match what was already done to SortedValues_jj)
                                RestrictedSortedWeights=accumarray(sortindex,RestrictedSortedWeights,[],@sum); % This has already been done to SortedValues, so have to do it to Restricted Agent Dist
                                RestrictedSortedWeights=RestrictedSortedWeights/sum(RestrictedSortedWeights(:)); % renormalize to 1
                                
                                tempStatsRestricted=StatsFromWeightedGrid(SortedValues_jj,RestrictedSortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,simoptions.whichstats);

                                % Now store these based on jj
                                if simoptions.whichstats(1)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Mean(jj)=tempStatsRestricted.Mean;
                                end
                                if simoptions.whichstats(2)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Median(jj)=tempStatsRestricted.Median;
                                    if simoptions.whichstats(1)==1
                                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).RatioMeanToMedian(jj)=tempStatsRestricted.RatioMeanToMedian;
                                    end
                                end
                                if simoptions.whichstats(3)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Variance(jj)=tempStatsRestricted.Variance;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation(jj)=tempStatsRestricted.StdDeviation;
                                end
                                if simoptions.whichstats(4)>=1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Gini(jj)=tempStatsRestricted.Gini;
                                    if simoptions.whichstats(4)<3
                                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve(:,jj)=tempStatsRestricted.LorenzCurve;
                                    end
                                end
                                if simoptions.whichstats(5)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Minimum(jj)=tempStatsRestricted.Minimum;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).Maximum(jj)=tempStatsRestricted.Maximum;
                                end
                                if simoptions.whichstats(6)>=1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs(:,jj)=tempStatsRestricted.QuantileCutoffs;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans(:,jj)=tempStatsRestricted.QuantileMeans;
                                end
                                if simoptions.whichstats(7)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share(jj)=tempStatsRestricted.MoreInequality.Top1share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share(jj)=tempStatsRestricted.MoreInequality.Top5share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share(jj)=tempStatsRestricted.MoreInequality.Top10share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share(jj)=tempStatsRestricted.MoreInequality.Bottom50share;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th(jj)=tempStatsRestricted.MoreInequality.Percentile50th;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th(jj)=tempStatsRestricted.MoreInequality.Percentile90th;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th(jj)=tempStatsRestricted.MoreInequality.Percentile95th;
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th(jj)=tempStatsRestricted.MoreInequality.Percentile99th;
                                end
                            else
                                RestrictedSortedWeights=zeros(size(SortedValues_jj),'gpuArray'); % Need this size for groupstats later
                            end

                            % For unrestricted stats, I do a more direct calculation of mean, std dev, min and max. But I don't bother with the conditional restriction stats.

                            % If doing grouped stats, store RestrictedSortedWeights
                            if simoptions_temp.groupusingtdigest==1
                                error('Code should never get here (should have thrown an error earlier')
                            else
                                if simoptions.ptypestorecpu==1
                                    AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(jgroupstr{jjageshifted})=[AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(jgroupstr{jjageshifted}); gather(RestrictedSortedWeights)*gather(sum(AgeMasses(ii,j1:jend).*restrictedsamplemass(ii,j1:jend,rr)))];
                                else
                                    AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(jgroupstr{jjageshifted})=[AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(jgroupstr{jjageshifted}); RestrictedSortedWeights*sum(AgeMasses(ii,j1:jend).*restrictedsamplemass(ii,j1:jend,rr))];
                                end
                                % Note: later normalize by sum(sum(restrictedsamplemass(:,j1:jend,rr),2))
                            end
                        end
                    end
                    
                end % end jj over agej groupings
            end
        end % end ff over FnsToEvalNames
    end % end ii over N_i


    
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
    for ff=1:numFnsToEvaluate
        if simoptions.whichstats(1)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,N_j_max2,'gpuArray'); % Note: N_j_max2=length(simoptions.agegroupings) in basic setup, will be different when N_j or agejshifter varies by PType
        end
        if simoptions.whichstats(2)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,N_j_max2,'gpuArray');
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,N_j_max2,'gpuArray');
            end
        end
        if simoptions.whichstats(3)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,N_j_max2,'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,N_j_max2,'gpuArray');
        end
        if simoptions.whichstats(4)>=1
            AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,N_j_max2,'gpuArray');
            if simoptions.whichstats(4)<3
                AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,N_j_max2,'gpuArray');
            end
        end
        if simoptions.whichstats(5)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,N_j_max2,'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,N_j_max2,'gpuArray');
        end
        if simoptions.whichstats(6)>=1
            AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,N_j_max2,'gpuArray'); % Includes the min and max values
            AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,N_j_max2,'gpuArray');
        end
        if simoptions.whichstats(7)==1
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
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean=nan(1,N_j_max2,'gpuArray');
                end
                if simoptions.whichstats(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,N_j_max2,'gpuArray');
                    if simoptions.whichstats(1)==1
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,N_j_max2,'gpuArray');
                    end
                end
                if simoptions.whichstats(3)==1
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance=nan(1,N_j_max2,'gpuArray');
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation=nan(1,N_j_max2,'gpuArray');
                end
                if simoptions.whichstats(4)>=1
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini=nan(1,N_j_max2,'gpuArray');
                    if simoptions.whichstats(4)<3
                        AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,N_j_max2,'gpuArray');
                    end
                end
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum=nan(1,N_j_max2,'gpuArray');
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum=nan(1,N_j_max2,'gpuArray');
                end
                if simoptions.whichstats(6)>=1
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,N_j_max2,'gpuArray'); % Includes the min and max values
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,N_j_max2,'gpuArray');
                end
                if simoptions.whichstats(7)==1
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

    %% Now compute the grouped stats
    if simoptions.groupptypesforstats==1
        for ff=1:numFnsToEvaluate
            for jj=1:1:maxngroups
                % We need to load up each ii, and put them together
                if simoptions.groupusingtdigest==1 % using t-Digests
                    Cmerge=AllCMerge.(FnsToEvalNames{ff}).(jgroupstr{jj});
                    digestweightsmerge=Alldigestweightsmerge.(FnsToEvalNames{ff}).(jgroupstr{jj});
                    % Clean off the zeros at the end of Cmerge (that exist because of how we preallocate 'too much' for Cmerge); same for digestweightsmerge.
                    Cmerge=Cmerge(1:merge_nsofar(jj,ff));
                    digestweightsmerge=digestweightsmerge(1:merge_nsofar(jj,ff));

                    % Merge the digests
                    [C_ff,digestweights_ff,~]=mergeDigest(Cmerge, digestweightsmerge, delta);

                    % digestweights_ff will sum to one, except if using different agejshifter across PTypes, so need to add a renormalization in case that is happening
                    if sum(digestweights_ff)>0
                        digestweights_ff=digestweights_ff/sum(digestweights_ff);
                    end
                    
                    tempStats=StatsFromWeightedGrid(C_ff,digestweights_ff,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
                elseif simoptions.ptypestorecpu==0 % just using unique() of the values and weights
                    [AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}),~,sortindex]=unique(AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}));
                    AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj})=accumarray(sortindex,AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}),[],@sum);

                    % AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}) will sum to one, except if using different agejshifter across PTypes, so need to add a renormalization in case that is happening
                    if sum(AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}))>0
                        AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj})=AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj})/sum(AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}));
                    end

                    tempStats=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}),AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
                end
                % Store them in AgeConditionalStats
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=tempStats.Mean;
                end
                if simoptions.whichstats(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Median(jj)=tempStats.Median;
                    if simoptions.whichstats(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(jj)=tempStats.RatioMeanToMedian;
                    end
                end
                if simoptions.whichstats(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=tempStats.Variance;
                    AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=tempStats.StdDeviation;
                end
                if simoptions.whichstats(4)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Gini(jj)=tempStats.Gini;
                    if simoptions.whichstats(4)<3
                        AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,jj)=tempStats.LorenzCurve;
                    end
                end
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=tempStats.Minimum;
                    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=tempStats.Maximum;
                end
                if simoptions.whichstats(6)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,jj)=tempStats.QuantileCutoffs;
                    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,jj)=tempStats.QuantileMeans;
                end
                if simoptions.whichstats(7)==1
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
                SigmaNxi=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights)'); % The sum of the masses of the relevant types

                % Mean
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*MeanVec(ff,:,jj))/SigmaNxi;
                end

                % Standard Deviation
                if simoptions.whichstats(3)==1
                    if N_i==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=StdDevVec(ff,:,jj);
                    else
                        temp2=zeros(N_i,1);
                        for ii=2:N_i
                            if FnsAndPTypeIndicator(ff,ii)==1
                                temp=MeanVec(ff,1:(ii-1),jj)-MeanVec(ff,ii,jj); % This bit with temp is just to handle numerical rounding errors where temp evalaulated to negative with order -15
                                if any(temp<0) && all(temp>10^(-12))
                                    temp=max(temp,0);
                                end
                                temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator(ff,1:(ii-1)).*(StationaryDist.ptweights(1:(ii-1))').*(temp.^2));
                            end
                        end
                        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=sqrt(sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*StdDevVec(ff,:,jj))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
                    end
                    AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=(AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj))^2;
                end

                % Similarly, directly calculate the minimum and maximum as this is cleaner (and overwrite these)
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=max(maxvaluevec(ff,:,jj));
                    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=min(minvaluevec(ff,:,jj));
                end

                %% Deal with conditional restrictions
                if useCondlRest==1
                    j1=simoptions_temp.agegroupings(jj);
                    if jj<length(simoptions_temp.agegroupings)
                        jend=simoptions_temp.agegroupings(jj+1)-1;
                    else
                        jend=N_j_temp;
                    end

                    for rr=1:length(CondlRestnFnNames)

                        if sum(sum(restrictedsamplemass(:,j1:jend,rr)))>0
                            % We need to load up each ii, and put them together
                            if simoptions.groupusingtdigest==1 % using t-Digests
                                error('You should not be able to get here in the code')
                            elseif simoptions.ptypestorecpu==0 % just using unique() of the values and weights
                                % [AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}),~,sortindex]=unique(AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}));
                                AllRestrictedWeights_rrffjj=accumarray(sortindex,AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(jgroupstr{jj})/sum(sum(restrictedsamplemass(:,j1:jend,rr),2)),[],@sum);
                                AllRestrictedWeights_rrffjj=AllRestrictedWeights_rrffjj/sum(AllRestrictedWeights_rrffjj(:));
                                
                                % AllRestrictedWeights_rrffjj will sum to one, except if using different agejshifter across PTypes, so need to add a renormalization in case that is happening
                                if sum(AllRestrictedWeights_rrffjj)>0
                                    AllRestrictedWeights_rrffjj=AllRestrictedWeights_rrffjj/sum(AllRestrictedWeights_rrffjj);
                                end
                                
                                tempStats2=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}),AllRestrictedWeights_rrffjj,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
                            end
                            % Store them in AgeConditionalStats
                            if simoptions.whichstats(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(jj)=tempStats2.Mean;
                            end
                            if simoptions.whichstats(2)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(jj)=tempStats2.Median;
                                if simoptions.whichstats(1)==1
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(jj)=tempStats2.RatioMeanToMedian;
                                end
                            end
                            if simoptions.whichstats(3)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(jj)=tempStats2.Variance;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(jj)=tempStats2.StdDeviation;
                            end
                            if simoptions.whichstats(4)>=1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(jj)=tempStats2.Gini;
                                if simoptions.whichstats(4)<3
                                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,jj)=tempStats2.LorenzCurve;
                                end
                            end
                            if simoptions.whichstats(5)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(jj)=tempStats2.Minimum;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(jj)=tempStats2.Maximum;
                            end
                            if simoptions.whichstats(6)>=1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,jj)=tempStats2.QuantileCutoffs;
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,jj)=tempStats2.QuantileMeans;
                            end
                            if simoptions.whichstats(7)==1
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
                    end
                end
            end
        end
    end

    if useCondlRest==1 % Store the restricted masses
        for rr=1:length(CondlRestnFnNames)
            if sum(sum(restrictedsamplemass(:,:,rr)))==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            end
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.ByAge=sum(restrictedsamplemass(:,:,rr).*StationaryDist.ptweights,1); % Conditional on age, what fraction satisfy restriction
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.ByPType=sum(restrictedsamplemass(:,:,rr).*AgeMasses,2); % Conditional on ptype, what fraction satisfy restriction
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.Total=sum(sum(restrictedsamplemass(:,:,rr).*AgeMasses.*StationaryDist.ptweights,1),2); % What fraction satisfy restriction

        end
    end
    

    %%
elseif simoptions.lowmemory==1
    warning('You are using simoptions.lowmemory=1 (in LifeCycleProfiles_FHorz_Case1_PType). Only do this if simoptions.groupusingtdigest=1 was not enough reduction in memory use to run (as lowmemory=1 will be much slower)')

    if isfield(simoptions,'conditionalrestrictions')
        error('simoptions.conditionalrestrictions cannot be used with simoptions.lowmemory=1 in LifeCycleProfiles_FHorz_Case1_PType (I am too lazy to implement it just now, contact me if you need this)')
    end
    %% Outer-loop over FnsToEvaluate and inner loop over ptype

    for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid

        % Setup things for grouping across ptypes
        if simoptions.groupusingtdigest==1 % Setupt things for t-Digest
            % Following few lines relate to the digest
            delta=10000;
            merge_nsofar=zeros(maxngroups,1); % Keep count
            merge_nsofar2=zeros(maxngroups,1); % Keep count

            AllCMerge=struct();
            Alldigestweightsmerge=struct();
            for jj=1:maxngroups
                AllCMerge.(FnsToEvalNames{ff}).(jgroupstr{jj})=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
                Alldigestweightsmerge.(FnsToEvalNames{ff}).(jgroupstr{jj})=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
            end

        else % Or we will just store the unique of grid and weights
            AllValues=struct();
            AllWeights=struct();
            for jj=1:maxngroups
                AllValues.(jgroupstr{jj})=[];
                AllWeights.(jgroupstr{jj})=[];
            end
        end


        %% Inner loop
        for ii=1:N_i

            % First set up simoptions
            simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted

            if simoptions_temp.verbose==1
                fprintf('Permanent type: %i of %i \n',ii, N_i)
            end
            if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
                PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii})); % Essentially just assuming vfoptions.ptypestorecpu=1 as well
                StationaryDist_ii=gpuArray(StationaryDist.(Names_i{ii}));
            else
                PolicyIndexes_temp=Policy.(Names_i{ii});
                StationaryDist_ii=StationaryDist.(Names_i{ii});
            end

            % Go through everything which might be dependent on permanent type (PType)
            % Notice that the way this is coded the grids (etc.) could be either
            % fixed, or a function (that depends on age, and possibly on permanent
            % type), or they could be a structure. Only in the case where they are
            % a structure is there a need to take just a specific part and send
            % only that to the 'non-PType' version of the command.

            if isstruct(n_d)
                n_d_temp=n_d.(Names_i{ii});
            else
                n_d_temp=n_d;
            end
            if isstruct(n_a)
                n_a_temp=n_a.(Names_i{ii});
            else
                n_a_temp=n_a;
            end
            if isstruct(n_z)
                n_z_temp=n_z.(Names_i{ii});
            else
                n_z_temp=n_z;
            end
            if isstruct(N_j)
                N_j_temp=N_j.(Names_i{ii});
            else
                N_j_temp=N_j;
            end
            if isstruct(d_grid)
                d_grid_temp=d_grid.(Names_i{ii});
            else
                d_grid_temp=d_grid;
            end
            if isstruct(a_grid)
                a_grid_temp=a_grid.(Names_i{ii});
            else
                a_grid_temp=a_grid;
            end
            if isstruct(z_grid)
                z_grid_temp=z_grid.(Names_i{ii});
            else
                z_grid_temp=z_grid;
            end

            % Parameters are allowed to be given as structure, or as vector/matrix
            % (in terms of their dependence on permanent type). So go through each of
            % these in term.
            Parameters_temp=Parameters;
            FullParamNames=fieldnames(Parameters);
            nFields=length(FullParamNames);
            for kField=1:nFields
                if isstruct(Parameters.(FullParamNames{kField})) % Check for permanent type in structure form
                    names=fieldnames(Parameters.(FullParamNames{kField}));
                    for jj=1:length(names)
                        if strcmp(names{jj},Names_i{ii})
                            Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{jj});
                        end
                    end
                elseif any(size(Parameters.(FullParamNames{kField}))==N_i) % Check for permanent type in vector/matrix form.
                    temp=Parameters.(FullParamNames{kField});
                    [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
                    if ptypedim==1
                        Parameters_temp.(FullParamNames{kField})=temp(ii,:);
                    elseif ptypedim==2
                        Parameters_temp.(FullParamNames{kField})=temp(:,ii);
                    end
                end
            end

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
            [n_z_temp,z_gridvals_J_temp,N_z_temp,l_z_temp,simoptions_temp]=CreateGridvals_FnsToEvaluate_FHorz(n_z_temp,z_grid_temp,N_j_temp,simoptions_temp);

            l_daprime_temp=size(PolicyIndexes_temp,1);
            % Switch to PolicyVals
            PolicyValues_temp=PolicyInd2Val_FHorz(PolicyIndexes_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,simoptions_temp,1);
            clear PolicyIndexes_temp % trying to reduce memory usage
            if l_z_temp==0
                PolicyValuesPermute_temp=permute(PolicyValues_temp,[2,3,1]); % (N_a,N_j,l_daprime)
            else
                PolicyValuesPermute_temp=permute(PolicyValues_temp,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
            end
            clear PolicyValues_temp % trying to reduce memory usage

            [~,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
            FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

            StationaryDist_ii=reshape(StationaryDist_ii,[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

            if FnsAndPTypeIndicator_ii(ff)==1 % If this function is relevant to this ptype

                % Get parameter names for current FnsToEvaluate functions
                tempnames=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
                if length(tempnames)>(l_d_temp+l_a_temp+l_a_temp+l_z_temp)
                    FnsToEvaluateParamNames={tempnames{l_d_temp+l_a_temp+l_a_temp+l_z_temp+1:end}}; % the first inputs will always be (d,aprime,a,z)
                else
                    FnsToEvaluateParamNames={};
                end
                if l_z_temp==0
                    CellOverAgeOfParamValues.(FnsToEvalNames{ff})=CreateCellOverAgeFromParams(Parameters_temp,FnsToEvaluateParamNames,N_j_temp,2);
                else
                    CellOverAgeOfParamValues.(FnsToEvalNames{ff})=CreateCellOverAgeFromParams(Parameters_temp,FnsToEvaluateParamNames,N_j_temp,3);
                end
                
                %% We have set up the current PType, now do some calculations for it.
                simoptions_temp.keepoutputasmatrix=2;
                ValuesOnGrid_ffii=EvalFnOnAgentDist_Grid_J(FnsToEvaluate.(FnsToEvalNames{ff}),CellOverAgeOfParamValues.(FnsToEvalNames{ff}),PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,n_z_temp,a_gridvals_temp,z_gridvals_J_temp);
                ValuesOnGrid_ffii=reshape(ValuesOnGrid_ffii,[N_a_temp*N_z_temp,N_j_temp]);

                % StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

                % Note, eliminating zero weights and unique() cannot be done yet as they need to be conditional on j
                % (otherwise lose the j dimension if I just apply them now)


                % Preallocate various things for the stats (as many will have jj as a dimension)
                % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Mean=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions.whichstats(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Median=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    if simoptions.whichstats(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).RatioMeanToMedian=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    end
                end
                if simoptions.whichstats(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Variance=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions.whichstats(4)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Gini=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    if simoptions.whichstats(4)<3
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve=nan(simoptions_temp.npoints,length(simoptions_temp.agegroupings),'gpuArray');
                    end
                end
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions.whichstats(6)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,length(simoptions_temp.agegroupings),'gpuArray'); % Includes the min and max values
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans=nan(simoptions_temp.nquantiles,length(simoptions_temp.agegroupings),'gpuArray');
                end
                if simoptions.whichstats(7)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
                end


                for jj=1:length(simoptions_temp.agegroupings)

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
                    Values_jj=reshape(ValuesOnGrid_ffii(:,j1:jend),[N_a_temp*N_z_temp*(jend-j1+1),1]);

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

                    SortedWeights_jj=SortedWeights_jj/sum(SortedWeights_jj(:)); % Normalize conditional on jj (is later renormalized ii weight before storing for groupstats)

                    %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and ff)
                    tempStats=StatsFromWeightedGrid(SortedValues_jj,SortedWeights_jj,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats); % 1 is presorted

                    % Now store these based on jjageshifted
                    if simoptions.whichstats(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Mean(jjageshifted)=tempStats.Mean;
                    end
                    if simoptions.whichstats(2)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Median(jjageshifted)=tempStats.Median;
                        if simoptions.whichstats(1)==1
                            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).RatioMeanToMedian(jjageshifted)=tempStats.RatioMeanToMedian;
                        end
                    end
                    if simoptions.whichstats(3)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Variance(jjageshifted)=tempStats.Variance;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation(jjageshifted)=tempStats.StdDeviation;
                    end
                    if simoptions.whichstats(4)>=1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Gini(jjageshifted)=tempStats.Gini;
                        if simoptions.whichstats(4)<3
                            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve(:,jjageshifted)=tempStats.LorenzCurve;
                        end
                    end
                    if simoptions.whichstats(5)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum(jjageshifted)=tempStats.Minimum;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum(jjageshifted)=tempStats.Maximum;
                    end
                    if simoptions.whichstats(6)>=1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs(:,jjageshifted)=tempStats.QuantileCutoffs;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans(:,jjageshifted)=tempStats.QuantileMeans;
                    end
                    if simoptions.whichstats(7)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share(jjageshifted)=tempStats.MoreInequality.Top1share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share(jjageshifted)=tempStats.MoreInequality.Top5share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share(jjageshifted)=tempStats.MoreInequality.Top10share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share(jjageshifted)=tempStats.MoreInequality.Bottom50share;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th(jjageshifted)=tempStats.MoreInequality.Percentile50th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th(jjageshifted)=tempStats.MoreInequality.Percentile90th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th(jjageshifted)=tempStats.MoreInequality.Percentile95th;
                        AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th(jjageshifted)=tempStats.MoreInequality.Percentile99th;
                    end

                    % For later, put the mean and std dev in a convenient place
                    if simoptions.whichstats(1)==1
                        MeanVec(ff,ii,jjageshifted)=tempStats.Mean;
                    end
                    if simoptions.whichstats(3)==1
                        StdDevVec(ff,ii,jjageshifted)=tempStats.StdDeviation;
                    end
                    % Do the same with the minimum and maximum
                    if simoptions.whichstats(5)==1
                        minvaluevec(ff,ii,jjageshifted)=tempStats.Minimum;
                        maxvaluevec(ff,ii,jjageshifted)=tempStats.Maximum;
                    end

                    if simoptions.groupptypesforstats==1 % Note: when using lowmemory=1 we just keep the current ff (FnsToEvaluate) [this is what makes it lower memory usage]
                        % Store things to use later to use to calculate the grouped stats
                        if simoptions_temp.groupusingtdigest==1
                            Cmerge=AllCMerge.(jgroupstr{jjageshifted});
                            digestweightsmerge=Alldigestweightsmerge.(jgroupstr{jjageshifted});

                            %% Create digest (if unique() was not enough to make them small)
                            [C_jj,digestweights_jj,~]=createDigest(SortedValues_jj, SortedWeights_jj,delta,1); % 1 is presorted

                            merge_nsofar2(jjageshifted)=merge_nsofar(jjageshifted)+length(C_jj);
                            Cmerge(merge_nsofar(jjageshifted)+1:merge_nsofar2(jjageshifted))=C_jj;
                            digestweightsmerge(merge_nsofar(jjageshifted)+1:merge_nsofar2(jjageshifted))=digestweights_jj*StationaryDist.ptweights(ii);
                            merge_nsofar(jjageshifted)=merge_nsofar2(jjageshifted);

                            AllCMerge.(jgroupstr{jjageshifted})=Cmerge;
                            Alldigestweightsmerge.(jgroupstr{jjageshifted})=digestweightsmerge;
                        else
                            AllValues.(jgroupstr{jjageshifted})=[AllValues.(jgroupstr{jjageshifted}); SortedValues_jj];
                            AllWeights.(jgroupstr{jjageshifted})=[AllWeights.(jgroupstr{jjageshifted}); SortedWeights_jj*StationaryDist.ptweights(ii)];
                        end
                    end
                end % end jj over agej groupings
            end
        end % end ii over N_i


        %% Now we compute the grouped stats
        % Preallocate various things for the stats (as many will have jj as a dimension)
        % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
        if simoptions.whichstats(1)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
        end
        if simoptions.whichstats(2)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
        end
        if simoptions.whichstats(3)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
        end
        if simoptions.whichstats(4)>=1
            AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
            if simoptions.whichstats(4)<3
                AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
            end
        end
        if simoptions.whichstats(5)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
        end
        if simoptions.whichstats(6)>=1
            AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
            AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
        end
        if simoptions.whichstats(7)==1
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
        end


        if simoptions.groupptypesforstats==1
            for jj=1:1:maxngroups
                % We need to load up each ii, and put them together
                if simoptions.groupusingtdigest==1 % using t-Digests
                    Cmerge=AllCMerge.(jgroupstr{jj});
                    digestweightsmerge=Alldigestweightsmerge.(jgroupstr{jj});
                    % Clean off the zeros at the end of Cmerge (that exist because of how we preallocate 'too much' for Cmerge); same for digestweightsmerge.
                    Cmerge=Cmerge(1:merge_nsofar(jj));
                    digestweightsmerge=digestweightsmerge(1:merge_nsofar(jj));

                    % Merge the digests
                    [C_ff,digestweights_ff,~]=mergeDigest(Cmerge, digestweightsmerge, delta);

                    tempStats=StatsFromWeightedGrid(C_ff,digestweights_ff,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
                elseif simoptions.ptypestorecpu==0 % just using unique() of the values and weights
                    [AllValues.(jgroupstr{jj}),~,sortindex]=unique(AllValues.(jgroupstr{jj}));
                    AllWeights.(jgroupstr{jj})=accumarray(sortindex,AllWeights.(jgroupstr{jj}),[],@sum);

                    tempStats=StatsFromWeightedGrid(AllValues.(jgroupstr{jj}),AllWeights.(jgroupstr{jj}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
                end
                % Store them in AgeConditionalStats
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=tempStats.Mean;
                end
                if simoptions.whichstats(2)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Median(jj)=tempStats.Median;
                    if simoptions.whichstats(1)==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(jj)=tempStats.RatioMeanToMedian;
                    end
                end
                if simoptions.whichstats(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=tempStats.Variance;
                    AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=tempStats.StdDeviation;
                end
                if simoptions.whichstats(4)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Gini(jj)=tempStats.Gini;
                    if simoptions.whichstats(4)<3
                        AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,jj)=tempStats.LorenzCurve;
                    end
                end
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=tempStats.Minimum;
                    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=tempStats.Maximum;
                end
                if simoptions.whichstats(6)>=1
                    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,jj)=tempStats.QuantileCutoffs;
                    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,jj)=tempStats.QuantileMeans;
                end
                if simoptions.whichstats(7)==1
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
                SigmaNxi=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights)'); % The sum of the masses of the relevant types

                % Mean
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*MeanVec(ff,:,jj))/SigmaNxi;
                end

                % Standard Deviation
                if simoptions.whichstats(3)==1
                    if N_i==1
                        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=StdDevVec(ff,:,jj);
                    else
                        temp2=zeros(N_i,1);
                        for ii=2:N_i
                            if FnsAndPTypeIndicator(ff,ii)==1
                                temp=MeanVec(ff,1:(ii-1),jj)-MeanVec(ff,ii,jj); % This bit with temp is just to handle numerical rounding errors where temp evalaulated to negative with order -15
                                if any(temp<0) && all(temp>10^(-12))
                                    temp=max(temp,0);
                                end
                                temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator(ff,1:(ii-1)).*(StationaryDist.ptweights(1:(ii-1))').*(temp.^2));
                            end
                        end
                        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=sqrt(sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*StdDevVec(ff,:,jj))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
                    end
                    AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=(AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj))^2;
                end

                % Similarly, directly calculate the minimum and maximum as this is cleaner (and overwrite these)
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=max(maxvaluevec(ff,:,jj));
                    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=min(minvaluevec(ff,:,jj));
                end
            end
        end
    
    end % end loop over ff (FnsToEvaluate)

end





end
