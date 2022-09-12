function LorenzCurve=EvalFnOnAgentDist_LorenzCurve_FHorz_Case1_PType(StationaryDist, Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
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

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.npoints=100;
    simoptions.groupptypesforstats=1;
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
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
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100;
    end
end

if isstruct(FnsToEvaluate)
    numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
else
    numFnsToEvaluate=length(FnsToEvaluate);
end
FnsAndPTypeIndicator=zeros(numFnsToEvaluate,N_i,'gpuArray');

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.groupptypesforstats=1;
else
    if ~isfield(simoptions,'groupptypesforstats')
       simoptions.groupptypesforstats=1;
    end
end

FnNames=fieldnames(FnsToEvaluate);

%% First do groupptypesforstats==0, then groupptypesforstats==1
% The later I want to loop over the functions to evaluate and within that over permanent type.
if simoptions.groupptypesforstats==0
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
        if isa(StationaryDist_temp, 'gpuArray')
            Parallel_temp=2;
        else
            Parallel_temp=1;
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
                fprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
                dbstack
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
                fprintf('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
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
                fprintf('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
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
                [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType, it should be the row dimension, if it is not then give a warning.
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
        [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,FnsAndPTypeIndicator_ii]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
        FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

        LorenzCurve.(Names_i{ii})=EvalFnOnAgentDist_LorenzCurve_FHorz_Case1(StationaryDist_temp,PolicyIndexes_temp, FnsToEvaluate_temp,Parameters_temp,FnsToEvaluateParamNames_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,z_grid_temp,Parallel_temp,simoptions_temp);
    
    end
    
else %simoptions.groupptypesforstats==1
    %% NOTE GROUPING ONLY WORKS IF THE GRIDS ARE THE SAME SIZES FOR EACH AGENT (for whom a given FnsToEvaluate is being calculated)
    % (mainly because otherwise would have to deal with simoptions.agegroupings being different for each agent and this requires more complex code)
    % Will throw an error if this is not the case

    % If grouping, we have ValuesOnDist and StationaryDist that contain
    % everything we will need. Now we just have to compute them.
    % Note that I do not currently allow the following simoptions to differ by PType

    if isstruct(FnsToEvaluate)
        FnsToEvalNames=fieldnames(FnsToEvaluate);
        numFnsToEvaluate=length(FnsToEvalNames);
    else
        error('You can only use PType when FnsToEvaluate is a structure')
    end

    for kk=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
        clear FnsToEvaluate_kk
        FnsToEvaluate_kk.(FnsToEvalNames{kk})=FnsToEvaluate.(FnsToEvalNames{kk}); % Structure containing just this funcion

        % Following few lines relate to the digest
        delta=10000;
        Cmerge=[];
        digestweightsmerge=[];
        
        for ii=1:N_i
            % First set up simoptions
            if exist('simoptions','var')
                simoptions_temp=PType_Options(simoptions,Names_i,ii);
                if ~isfield(simoptions_temp,'verbose')
                    simoptions_temp.verbose=0;
                end
                if ~isfield(simoptions_temp,'verboseparams')
                    simoptions_temp.verboseparams=0;
                end
                if ~isfield(simoptions_temp,'ptypestorecpu')
                    simoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
                end
            else
                simoptions_temp.verbose=0;
                simoptions_temp.verboseparams=0;
                simoptions_temp.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
            end

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
            FnsAndPTypeIndicator(:,ii)=FnsAndPTypeIndicator_ii;

            simoptions_temp.keepoutputasmatrix=2; %2: is a matrix, but of a different form to 1
            ValuesOnDist_ii=gather(EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp, simoptions_temp));
            N_a_temp=prod(n_a_temp);
            if isfield(simoptions_temp,'n_e')
                n_z_temp=[n_z_temp,simoptions_temp.n_e];
            end
            N_z_temp=prod(n_z_temp);
            % Note: the shape of ValuesOnGrid_ii is because of simoptions_temp.keepoutputasmatrix=2
            %             ValuesOnDist.(Names_i{ii})=reshape(ValuesOnGrid_ii,[N_a_temp*N_z_temp,N_j_temp]);

            StationaryDist_ii=reshape(StationaryDist.(Names_i{ii}),[N_a_temp*N_z_temp,N_j_temp]); % Note: does not impose *StationaryDist.ptweights(ii)

            % Create digest
            [C_ii,digestweights_ii,~]=createDigest(ValuesOnDist_ii, StationaryDist_ii,delta);

            % Store the digest so that we can merge them all across agent type
            Cmerge=[Cmerge;C_ii];
            digestweightsmerge=[digestweightsmerge;digestweights_ii*StationaryDist.ptweights(ii)];

        end

        % Merge the digests
        [C_kk,digestweights_kk,qlimitvec_kk]=mergeDigest(Cmerge, digestweightsmerge, delta);

        % Calculate the quantiles
        LorenzCurve.(FnNames{kk})=LorenzCurve_subfunction_PreSorted(C_kk.*digestweights_kk,qlimitvec_kk,simoptions_temp.npoints,1);

    end
end

% If using FnsToEvaluate as structure need to get in appropriate form for output
if isstruct(FnsToEvaluate) && simoptions.groupptypesforstats==0
    FnNames=fieldnames(FnsToEvaluate);
    % Change the output into a structure
    LorenzCurve2=LorenzCurve;
    LorenzCurve=struct();
    for ff=1:length(FnNames)
        for ii=1:N_i
            LorenzCurve.(FnNames{ff}).(Names_i{ii})=LorenzCurve2.(Names_i{ii}).(FnNames{ff});
        end
    end
end
%Note: nothing needs to be dome if simoptions.groupptypesforstats==1


end
