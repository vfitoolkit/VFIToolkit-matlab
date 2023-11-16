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
    Names_i={'ptype001'};
    for ii=2:N_i
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
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
else
    if ~isfield(simoptions,'groupptypesforstats')
        simoptions.groupptypesforstats=1;
    end
    if ~isfield(simoptions,'ptypestorecpu')
        if simoptions.groupptypesforstats==1
            simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
        elseif simoptions.groupptypesforstats==0
            simoptions.ptypestorecpu=0;
        end
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
minvaluevec=nan(N_i,1);
maxvaluevec=nan(N_i,1);


%% NOTE GROUPING ONLY WORKS IF THE GRIDS ARE THE SAME SIZES FOR EACH AGENT (for whom a given FnsToEvaluate is being calculated)
% (mainly because otherwise would have to deal with simoptions.agegroupings being different for each agent and this requires more complex code)
% Will throw an error if this is not the case

% If grouping, we have ValuesOnDist and StationaryDist that contain
% everything we will need. Now we just have to compute them.
% Note that I do not currently allow the following simoptions to differ by PType

for kk=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
    clear FnsToEvaluate_kk
    FnsToEvaluate_kk.(FnsToEvalNames{kk})=FnsToEvaluate.(FnsToEvalNames{kk}); % Structure containing just this funcion
    FnsAndPTypeIndicator_kk=zeros(1,N_i,'gpuArray');

    % Following few lines relate to the digest
    delta=10000;
    Cmerge=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
    digestweightsmerge=zeros(5000*N_i,1); % This is intended to be an upper limit on number of points that might be use
    merge_nsofar=0; % Keep count

    MeanVec=zeros(1,N_i);
    StdDevVec=zeros(1,N_i);

    
    for ii=1:N_i
        % First set up simoptions
        simoptions_temp=PType_Options(simoptions,Names_i,ii); % Note: already check for existence of simoptions and created it if it was not inputted

        if simoptions_temp.verbose==1
            fprintf('Permanent type: %i of %i \n',ii, N_i)
        end

        if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
            PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii}));
            StationaryDist_temp=gpuArray(StationaryDist.(Names_i{ii}));
        else
            PolicyIndexes_temp=Policy.(Names_i{ii});
            StationaryDist_temp=StationaryDist.(Names_i{ii});
        end
        % Parallel is determined by StationaryDist, unless it is specified
        if isa(StationaryDist_temp, 'gpuArray')
            Parallel_temp=2;
        else
            Parallel_temp=1;
        end
        if isfield(simoptions_temp,'parallel')
            Parallel_temp=simoptions.parallel;
            if Parallel_temp~=2
                PolicyIndexes_temp=gather(PolicyIndexes_temp);
                StationaryDist_temp=gather(StationaryDist_temp);
            end
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

        n_d_temp=n_d;
        if isa(n_d,'struct')
            n_d_temp=n_d.(Names_i{ii});
        else
            temp=size(n_d);
            if temp(1)>1 % n_d depends on fixed type
                n_d_temp=n_d(ii,:);
            elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
                sprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
            end
        end
        n_a_temp=n_a;
        if isa(n_a,'struct')
            n_a_temp=n_a.(Names_i{ii});
        else
            temp=size(n_a);
            if temp(1)>1 % n_a depends on fixed type
                n_a_temp=n_a(ii,:);
            elseif temp(2)==N_i % If there is one row, but number of elements in n_a happens to coincide with number of permanent types, then just let user know
                sprintf('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
                dbstack
            end
        end
        n_z_temp=n_z;
        if isa(n_z,'struct')
            n_z_temp=n_z.(Names_i{ii});
        else
            temp=size(n_z);
            if temp(1)>1 % n_z depends on fixed type
                n_z_temp=n_z(ii,:);
            elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
                sprintf('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
                dbstack
            end
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

        % Figure out which functions are actually relevant to the present PType. Only the relevant ones need to be evaluated.
        % The dependence of FnsToEvaluate and FnsToEvaluateFnParamNames are necessarily the same.
        % Allows for FnsToEvaluate as structure.
        if n_d_temp(1)==0
            l_d_temp=0;
        else
            l_d_temp=1;
        end
        l_a_temp=length(n_a_temp);
        l_z_temp=length(n_z_temp);
        % Note: next line uses FnsToEvaluate_kk
        [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate_kk,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
        FnsAndPTypeIndicator_kk(ii)=FnsAndPTypeIndicator_ii;

        %% We have set up the current PType, now do some calculations for it.
        simoptions_temp.outputasstructure=0;
        ValuesOnGrid_ii=gather(EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, simoptions_temp));
        N_a_temp=prod(n_a_temp);
        if isfield(simoptions_temp,'n_e')
            n_z_temp=[n_z_temp,simoptions_temp.n_e];
        end
        N_z_temp=prod(n_z_temp);
        
        ValuesOnGrid_ii=reshape(ValuesOnGrid_ii,[N_a_temp*N_z_temp*N_j_temp,1]);
        
        StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp*N_j_temp,1]); % Note: does not impose *StationaryDist.ptweights(ii)

        % Eliminate all the zero-weighted points (this doesn't really same
        % runtime for the exact calculation and often can increase it, but 
        % for the createDigest it slashes the runtime. So since we want it 
        % then we may as well do it now.)
        temp=logical(StationaryDist_ii~=0);
        StationaryDist_ii=StationaryDist_ii(temp);
        ValuesOnGrid_ii=ValuesOnGrid_ii(temp);
        
        %% Use the full ValuesOnGrid_ii and StationaryDist_ii to calculate various statistics for the current PType-FnsToEvaluate (current ii and kk)
        AllStats.(FnsToEvalNames{kk}).(Names_i{ii})=StatsFromWeightedGrid(ValuesOnGrid_ii,StationaryDist_ii,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance);

        % For later, put the mean and std dev in a convenient place
        MeanVec(ii)=AllStats.(FnsToEvalNames{kk}).(Names_i{ii}).Mean;
        StdDevVec(ii)=AllStats.(FnsToEvalNames{kk}).(Names_i{ii}).StdDeviation;
        % Do the same with the minimum and maximum
        minvaluevec(ii)=AllStats.(FnsToEvalNames{kk}).(Names_i{ii}).Minimum;
        maxvaluevec(ii)=AllStats.(FnsToEvalNames{kk}).(Names_i{ii}).Maximum;

        %% Create digest
        [SortedValues,SortedValues_index]=sort(ValuesOnGrid_ii);
        SortedWeights=StationaryDist_ii(SortedValues_index);
        
        [C_ii,digestweights_ii,~]=createDigest(SortedValues, SortedWeights,delta,1); % 1 is presorted

        merge_nsofar2=merge_nsofar+length(C_ii);
        Cmerge(merge_nsofar+1:merge_nsofar2)=C_ii;
        digestweightsmerge(merge_nsofar+1:merge_nsofar2)=digestweights_ii*StationaryDist.ptweights(ii);
        merge_nsofar=merge_nsofar2;
    end
    % Clean off the zeros at the end of Cmerge (that exist because of how
    % we preallocate 'too much' for Cmerge); same for digestweightsmerge.
    Cmerge=Cmerge(1:merge_nsofar);
    digestweightsmerge=digestweightsmerge(1:merge_nsofar);

    %% Now for the grouped stats, most of which are calculated from digests
    
    % Merge the digests
    [C_kk,digestweights_kk,qlimitvec_kk]=mergeDigest(Cmerge, digestweightsmerge, delta);
    
    AllStats.(FnsToEvalNames{kk})=StatsFromWeightedGrid(C_kk,digestweights_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance);    
    
    % Grouped mean and standard deviation are overwritten on a more direct calculation that does not involve the digests
    SigmaNxi=sum(FnsAndPTypeIndicator_kk.*(StationaryDist.ptweights)'); % The sum of the masses of the relevant types
    
    % Mean
    AllStats.(FnsToEvalNames{kk}).Mean=sum(FnsAndPTypeIndicator_kk.*(StationaryDist.ptweights').*MeanVec)/SigmaNxi;

    % Standard Deviation
    if N_i==1
        AllStats.(FnsToEvalNames{kk}).StdDev=StdDevVec;
    else
        temp2=zeros(N_i,1);
        for ii=2:N_i
            if FnsAndPTypeIndicator_kk(ii)==1
                temp2(ii)=StationaryDist.ptweights(ii)*sum(FnsAndPTypeIndicator_kk(1:(ii-1)).*(StationaryDist.ptweights(1:(ii-1))').*((MeanVec(1:(ii-1))-MeanVec(ii)).^2));
            end
        end
        AllStats.(FnsToEvalNames{kk}).StdDev=sqrt(sum(FnsAndPTypeIndicator_kk.*(StationaryDist.ptweights').*StdDevVec)/SigmaNxi + sum(temp2)/(SigmaNxi^2));
    end
    AllStats.(FnsToEvalNames{kk}).Variance=(AllStats.(FnsToEvalNames{kk}).StdDev)^2;
    
    % Similarly, directly calculate the minimum and maximum as this is cleaner (and overwrite these)
    AllStats.(FnsToEvalNames{kk}).Maximum=max(maxvaluevec);
    AllStats.(FnsToEvalNames{kk}).Minimum=min(minvaluevec);

end




end
