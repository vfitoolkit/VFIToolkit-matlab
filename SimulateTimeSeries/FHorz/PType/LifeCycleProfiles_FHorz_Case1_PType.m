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
    simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    simoptions.groupusingtdigest=0; % if you are ptypestorecpu=1 and groupptypesforstats=1, you might also need to use groupusingtdigest=1 if you get out of memory errors
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    defaultagegroupings=1;
    if isstruct(N_j)
        for ii=1:N_i
            if isfinite(N_j.(Names_i{ii}))
                simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
            else % Infinite horizon
                computeForThesei(ii)=0;
            end
        end
    else
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.agejshifter=0; % Use when different PTypes have different initial ages (will be a structure when actually used)
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
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
    if isfield(simoptions,'agegroupings')==0
        defaultagegroupings=1;
        if isstruct(N_j)
            for ii=1:N_i
                if isfinite(N_j.(Names_i{ii}))
                    simoptions.agegroupings.(Names_i{ii})=1:1:N_j.(Names_i{ii});
                else % Infinite horizon
                    computeForThesei(ii)=0;
                end
            end
        else
            simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
        end
    else
        defaultagegroupings=0;
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
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
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
elseif length(simoptions.agejshifter)==1 % not using agejshifter
    simoptions.agejshifter=zeros(N_i,1);
else % have inputed as a vector
    simoptions.agejshifter=simoptions.agejshifter-min(simoptions.agejshifter); % put them all relative to the minimum
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

FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');


%% Do an outerloop over ptypes and an inner loop over FnsToEvaluate
for ii=1:N_i

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end    
    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii})); % Essentially just assuming vfoptions.ptypestorecpu=1 as well
        StationaryDist_temp=gpuArray(StationaryDist.(Names_i{ii}));
    else
        PolicyIndexes_temp=Policy.(Names_i{ii});
        StationaryDist_temp=StationaryDist.(Names_i{ii});
    end
    
    % Go through everything which might be dependent on permanent type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on permanent
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    
    if isa(n_d,'struct')
        n_d_temp=n_d.(Names_i{ii});
    else
        n_d_temp=n_d;
    end
    if isa(n_a,'struct')
        n_a_temp=n_a.(Names_i{ii});
    else
        n_a_temp=n_a;
    end
    if isa(n_z,'struct')
        n_z_temp=n_z.(Names_i{ii});
    else
        n_z_temp=n_z;
    end
    if isa(N_j,'struct')
        N_j_temp=N_j.(Names_i{ii});
    else
        N_j_temp=N_j;
    end
    if isa(d_grid,'struct')
        d_grid_temp=d_grid.(Names_i{ii});
    else
        d_grid_temp=d_grid;
    end
    if isa(a_grid,'struct')
        a_grid_temp=a_grid.(Names_i{ii});
    else
        a_grid_temp=a_grid;
    end
    if isa(z_grid,'struct')
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
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
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
    
    % Switch to PolicyVals
    PolicyValues_temp=PolicyInd2Val_FHorz(PolicyIndexes_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,simoptions_temp,1);
    
    % A few other things we can do in outer loop
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);
    
    N_a_temp=prod(n_a_temp);
    if isfield(simoptions_temp,'n_e')
        n_ze_temp=[n_z_temp,simoptions_temp.n_e];
    else
        n_ze_temp=n_z_temp;
    end
    if isfield(simoptions_temp,'n_semiz')
        n_ze_temp=[simoptions_temp.n_semiz,n_ze_temp];
    end
    N_ze_temp=max(prod(n_ze_temp),1); % if zero, overwrite with one

    [~,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

    StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_ze_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

    for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid

        if FnsAndPTypeIndicator_ii(ff)==1 % If this function is relevant to this ptype

            clear FnsToEvaluate_iiff
            FnsToEvaluate_iiff.(FnsToEvalNames{ff})=FnsToEvaluate.(FnsToEvalNames{ff});

            %% We have set up the current PType, now do some calculations for it.
            simoptions_temp.keepoutputasmatrix=2;
            ValuesOnGrid_ffii=EvalFnOnAgentDist_ValuesOnGrid_FHorz_subfn(PolicyValues_temp, FnsToEvaluate_iiff, Parameters_temp, [], n_d_temp, n_a_temp, n_z_temp, N_j_temp, a_grid_temp, z_grid_temp, simoptions_temp);

            % ValuesOnGrid_ffii=reshape(ValuesOnGrid_ffii,[N_a_temp*N_ze_temp,N_j_temp]); Already has this shape
            % StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_ze_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

            % Note, eliminating zero weights and unique() cannot be done yet as they need to be conditional on j 
            % (otherwise lose the j dimension if I just apply them now)


            % Preallocate various things for the stats (as many will have jj as a dimension)
            % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Mean=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Median=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Variance=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Gini=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top1share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top5share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Top10share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Bottom50share=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile50th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile90th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile95th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).MoreInequality.Percentile99th=nan(1,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve=nan(simoptions_temp.npoints,length(simoptions_temp.agegroupings),'gpuArray');
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileCutoffs=nan(simoptions_temp.nquantiles+1,length(simoptions_temp.agegroupings),'gpuArray'); % Includes the min and max values
            AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).QuantileMeans=nan(simoptions_temp.nquantiles,length(simoptions_temp.agegroupings),'gpuArray');

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
                StationaryDistVec_jj=reshape(StationaryDist_ii(:,j1:jend),[N_a_temp*N_ze_temp*(jend-j1+1),1]);
                Values_jj=reshape(ValuesOnGrid_ffii(:,j1:jend),[N_a_temp*N_ze_temp*(jend-j1+1),1]);

                % Eliminate all the zero-weighted points (this doesn't really save runtime for the exact calculation and often can increase it, but
                % for the createDigest it slashes the runtime. So since we want it then we may as well do it now.)
                temp=logical(StationaryDistVec_jj~=0); % NOTE: This and the next line could in principle be done outside all of these loops (just looping over j)
                StationaryDistVec_jj=StationaryDistVec_jj(temp);
                Values_jj=Values_jj(temp);

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
                end
                if simoptions.whichstats(3)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Variance(jjageshifted)=tempStats.Variance;
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation(jjageshifted)=tempStats.StdDeviation;
                end
                if simoptions.whichstats(4)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).LorenzCurve(:,jjageshifted)=tempStats.LorenzCurve;
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Gini(jjageshifted)=tempStats.Gini;
                end
                if simoptions.whichstats(5)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum(jjageshifted)=tempStats.Minimum;
                    AgeConditionalStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum(jjageshifted)=tempStats.Maximum;
                end
                if simoptions.whichstats(6)==1
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
            end % end jj
        end
    end % end ff
end % end ii



%% Now we compute the grouped stats
% Preallocate various things for the stats (as many will have jj as a dimension)
% Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
for ff=1:numFnsToEvaluate
    AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
    AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
end


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

                tempStats=StatsFromWeightedGrid(C_ff,digestweights_ff,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
            elseif simoptions.ptypestorecpu==0 % just using unique() of the values and weights
                [AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}),~,sortindex]=unique(AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}));
                AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj})=accumarray(sortindex,AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}),[],@sum);
                tempStats=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}).(jgroupstr{jj}),AllWeights.(FnsToEvalNames{ff}).(jgroupstr{jj}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
            end
            % Store them in AgeConditionalStats
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(jj)=tempStats.Mean;
            end
            if simoptions.whichstats(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(jj)=tempStats.Median;
            end
            if simoptions.whichstats(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(jj)=tempStats.StdDeviation;
            end
            if simoptions.whichstats(4)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(jj)=tempStats.Gini;
                AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,jj)=tempStats.LorenzCurve;
            end
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=tempStats.Maximum;
            end
            if simoptions.whichstats(6)==1
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
                    AgeConditionalStats.(FnsToEvalNames{ff}).StdDev(jj)=StdDevVec(ff,:,jj);
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
                    AgeConditionalStats.(FnsToEvalNames{ff}).StdDev(jj)=sqrt(sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*StdDevVec(ff,:,jj))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
                end
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(jj)=(AgeConditionalStats.(FnsToEvalNames{ff}).StdDev(jj))^2;
            end

            % Similarly, directly calculate the minimum and maximum as this is cleaner (and overwrite these)
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(jj)=max(maxvaluevec(ff,:,jj));
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(jj)=min(minvaluevec(ff,:,jj));
            end
        end
    end    
end







end
