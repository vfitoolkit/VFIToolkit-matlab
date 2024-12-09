function AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
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


%% NOTE GROUPING ONLY WORKS IF THE GRIDS ARE THE SAME SIZES FOR EACH AGENT (for whom a given FnsToEvaluate is being calculated)
% (mainly because otherwise would have to deal with simoptions.agegroupings being different for each agent and this requires more complex code)
% Will throw an error if this is not the case

% If grouping, we have ValuesOnDist and StationaryDist that contain
% everything we will need. Now we just have to compute them.
% Note that I do not currently allow the following simoptions to differ by PType


for ii=1:N_i

    tic;

    % First set up simoptions
    simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted

    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end

    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii}));
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

    % Start with those that determine whether the current permanent type is finite or
    % infinite horizon, and whether it is Case 1 or Case 2
    % Figure out which case is relevant to the current PType. This is done
    % using N_j which for the current type will evaluate to 'Inf' if it is
    % infinite horizon and a finite number for any other finite horizon.
    % First, check if it is a structure, and otherwise just get the
    % relevant value.

    % Horizon is determined via N_j
    if isstruct(N_j)
        N_j_temp=N_j.(Names_i{ii});
    elseif isscalar(N_j)
        N_j_temp=N_j;
    else % is a vector
        N_j_temp=N_j(ii);
    end

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
    % ie. Parameters.alpha=[0;1]; or Parameters.alpha.ptype1=0; Parameters.alpha.ptype2=1;
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters); % all the different parameters
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check the current parameter for permanent type in structure form
            % Check if this parameter is used for the current permanent type (it may or may not be, some parameters are only used be a subset of permanent types)
            if isfield(Parameters.(FullParamNames{kField}),Names_i{ii})
                Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(Names_i{ii});
            end
        elseif sum(size(Parameters.(FullParamNames{kField}))==N_i)>=1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
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
    [n_z_temp,z_gridvals_J_temp,N_z_temp,l_z_temp,simoptions_temp]=CreateGridvals_FnsToEvaluate_FHorz(n_z_temp,z_grid_temp,N_j_temp,simoptions_temp);
    if N_z_temp==0
        N_z_temp=1; % Just makes things easier below
    end

    l_daprime_temp=size(PolicyIndexes_temp,1);
    % Switch to PolicyVals
    PolicyValues_temp=PolicyInd2Val_FHorz(PolicyIndexes_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,simoptions_temp,1);
    if l_z_temp==0
        PolicyValuesPermute_temp=permute(PolicyValues_temp,[2,3,1]); % (N_a,N_j,l_daprime)
    else
        PolicyValuesPermute_temp=permute(PolicyValues_temp,[2,3,4,1]); % (N_a,N_z,N_j,l_daprime)
    end
    
    [~,~,~,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

    %% Some things that don't need to go in the loop over FnsToEvalaute
    StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp*N_j_temp,1]); % Note: does not impose *StationaryDist.ptweights(ii)
    % Eliminate all the zero-weighted points (this doesn't really save runtime for the exact calculation and often can increase it, but
    % for the createDigest it slashes the runtime. So since we want it then we may as well do it now.)    
    temp=logical(StationaryDist_ii~=0);
    % StationaryDist_ii=StationaryDist_ii(temp); % This has to happen after the conditional restriction dist is calculated
    
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
            RestrictedStationaryDistVec=RestrictedStationaryDistVec(temp); % This has already been done to StationaryDist_ii, so have to do it to Restricted Agent Dist

            % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
            restrictedsamplemass(ii,rr)=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec./restrictedsamplemass(ii,rr); % Normalize to mass of 1
            % Store for later
            RestrictionStruct_ii(rr).RestrictedStationaryDistVec=RestrictedStationaryDistVec;
            
            AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.(Names_i{ii})=restrictedsamplemass(ii,rr); % Seems likely this would be something user might want
            if restrictedsamplemass(ii,rr)==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            end
        end
    end
    
    %%
    StationaryDist_ii=StationaryDist_ii(temp);
    
    %%
    for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
        if FnsAndPTypeIndicator_ii(ff)==1 % If this function is relevant to this ptype

            % Get parameter names for current FnsToEvaluate functions
            tempnames=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
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
            ValuesOnGrid_ii=EvalFnOnAgentDist_Grid_J(FnsToEvaluate.(FnsToEvalNames{ff}),CellOverAgeOfParamValues,PolicyValuesPermute_temp,l_daprime_temp,n_a_temp,n_z_temp,a_gridvals_temp,z_gridvals_J_temp);
            ValuesOnGrid_ii=reshape(ValuesOnGrid_ii,[N_a_temp*N_z_temp*N_j_temp,1]);

            % StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp*N_j_temp,1]); % Note: does not impose *StationaryDist.ptweights(ii)

            % Eliminate all the zero-weighted points (this doesn't really save runtime for the exact calculation and often can increase it, but
            % for the createDigest it slashes the runtime. So since we want it then we may as well do it now.)
            % temp=logical(StationaryDist_ii~=0);
            % StationaryDist_ii=StationaryDist_ii(temp);
            ValuesOnGrid_ii=ValuesOnGrid_ii(temp);

            % I want to use unique to make it easier to put the different agent
            % ptypes together (as all the matrices are typically smaller).
            % May as well do it before doing the StatsFromWeightedGrid
            [SortedValues,~,sortindex]=unique(ValuesOnGrid_ii);
            SortedWeights=accumarray(sortindex,StationaryDist_ii,[],@sum);
            
            %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and kk)
            AllStats.(FnsToEvalNames{ff}).(Names_i{ii})=StatsFromWeightedGrid(SortedValues,SortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats); % 1 is presorted

            %% If using conditional restrictions, do those
            if useCondlRest==1
                for rr=1:length(CondlRestnFnNames)
                    RestrictedSortedWeights=accumarray(sortindex,RestrictionStruct_ii(rr).RestrictedStationaryDistVec,[],@sum); % This has already been done to SortedValues, so have to do it to Restricted Agent Dist
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(Names_i{ii})=StatsFromWeightedGrid(SortedValues,RestrictedSortedWeights,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats); % 1 is presorted
                    
                    % If doing grouped stats, store RestrictedSortedWeights
                    if simoptions_temp.groupusingtdigest==1
                        error('Code should never get here (should have thrown an error earlier')
                    else
                        if simoptions.ptypestorecpu==1
                            AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=[AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}); gather(RestrictedSortedWeights*restrictedsamplemass(ii,rr))]; %*gather(StationaryDist.ptweights(ii))];
                        else
                            AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=[AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}); RestrictedSortedWeights*restrictedsamplemass(ii,rr)]; %*StationaryDist.ptweights(ii)];
                        end
                        % Note: later once we have all the ii do AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}) renormalized by sum(restrictedsamplemass(:,rr))
                    end
                end
            end
            
            %% For later, put the mean and std dev in a convenient place
            if simoptions.whichstats(1)==1
                MeanVec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(Names_i{ii}).Mean;
            end
            if simoptions.whichstats(3)==1
                StdDevVec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(Names_i{ii}).StdDeviation;
            end
            % Do the same with the minimum and maximum
            if simoptions.whichstats(5)==1
                minvaluevec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(Names_i{ii}).Minimum;
                maxvaluevec(ff,ii)=AllStats.(FnsToEvalNames{ff}).(Names_i{ii}).Maximum;
            end
            
            if simoptions_temp.groupusingtdigest==1
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
for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid    

    if simoptions_temp.groupusingtdigest==1
        Cmerge=AllCMerge.(FnsToEvalNames{ff});
        digestweightsmerge=Alldigestweightsmerge.(FnsToEvalNames{ff});
        % Clean off the zeros at the end of Cmerge (that exist because of how we preallocate 'too much' for Cmerge); same for digestweightsmerge.
        Cmerge=Cmerge(1:merge_nsofar(ff));
        digestweightsmerge=digestweightsmerge(1:merge_nsofar(ff));

        % Merge the digests
        [C_kk,digestweights_kk,~]=mergeDigest(Cmerge, digestweightsmerge, delta);

        tempStats=StatsFromWeightedGrid(C_kk,digestweights_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);

        allstatnames=fieldnames(tempStats);
    else
        % Do unique() before we calculate stats
        [AllValues.(FnsToEvalNames{ff}),~,sortindex]=unique(AllValues.(FnsToEvalNames{ff}));
        AllWeights.(FnsToEvalNames{ff})=accumarray(sortindex,AllWeights.(FnsToEvalNames{ff}),[],@sum);

        tempStats=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}),AllWeights.(FnsToEvalNames{ff}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);

        allstatnames=fieldnames(tempStats);
        if useCondlRest==1
            for rr=1:length(CondlRestnFnNames)
                AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})=accumarray(sortindex,AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff})/sum(restrictedsamplemass(:,rr)),[],@sum);
                tempStatsRestricted=StatsFromWeightedGrid(AllValues.(FnsToEvalNames{ff}),AllRestrictedWeights.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,simoptions.whichstats);
                % Following is necessary as just AllStats=StatsFromWeightedGrid() overwrote the existing subfields
                rallstatnames=fieldnames(tempStatsRestricted);
                for aa=1:length(rallstatnames)
                    AllStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).(rallstatnames{aa})=tempStatsRestricted.(rallstatnames{aa});
                end
                if ff==1
                    AllStats.(CondlRestnFnNames{rr}).RestrictedSampleMass.TotalAllPTypes=sum(restrictedsamplemass(:,rr));
                end
            end
        end
    end
    % Following is necessary as just AllStats=StatsFromWeightedGrid() overwrote the existing subfields
    % allstatnames=fieldnames(tempStats);
    for aa=1:length(allstatnames)
        AllStats.(FnsToEvalNames{ff}).(allstatnames{aa})=tempStats.(allstatnames{aa});
    end
    

    % Grouped mean and standard deviation are overwritten on a more direct calculation that does not involve the digests
    SigmaNxi=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights)'); % The sum of the masses of the relevant types
    
    % Mean
    if simoptions.whichstats(1)==1
        AllStats.(FnsToEvalNames{ff}).Mean=sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*MeanVec(ff,:))/SigmaNxi;
    end

    % Standard Deviation
    if simoptions.whichstats(3)==1
        if N_i==1
            AllStats.(FnsToEvalNames{ff}).StdDev=StdDevVec(ff,:);
        else
            temp2=zeros(N_i,1);
            for ii=2:N_i
                if FnsAndPTypeIndicator(ff,ii)==1
                    temp=MeanVec(ff,1:(ii-1))-MeanVec(ff,ii); % This bit with temp is just to handle numerical rounding errors where temp evalaulated to negative with order -15
                    if any(temp<0) && all(temp>10^(-12))
                        temp=max(temp,0);
                    end
                    temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator(ff,1:(ii-1)).*(StationaryDist.ptweights(1:(ii-1))').*(temp.^2));
                end
            end
            AllStats.(FnsToEvalNames{ff}).StdDev=sqrt(sum(FnsAndPTypeIndicator(ff,:).*(StationaryDist.ptweights').*StdDevVec(ff,:))/SigmaNxi + sum(temp2)/(SigmaNxi^2));
        end
        AllStats.(FnsToEvalNames{ff}).Variance=(AllStats.(FnsToEvalNames{ff}).StdDev)^2;
    end

    % Similarly, directly calculate the minimum and maximum as this is cleaner (and overwrite these)
    if simoptions.whichstats(5)==1
        AllStats.(FnsToEvalNames{ff}).Maximum=max(maxvaluevec(ff,:));
        AllStats.(FnsToEvalNames{ff}).Minimum=min(minvaluevec(ff,:));
    end
end


end
