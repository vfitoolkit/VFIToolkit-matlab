function ValuesOnGrid=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1_PType(Policy, FnsToEvaluate, Parameters,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, simoptions)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
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

if isstruct(FnsToEvaluate)
    numFnsToEvaluate=length(fieldnames(FnsToEvaluate));
else
    numFnsToEvaluate=length(FnsToEvaluate);
end

% RIGHT NOW THIS ValuesOnGrid ONLY WORKS WHEN ALL AGENTS ARE ON THE SAME GRID
N_a=prod(n_a);
N_z=prod(n_z);
if ~isstruct(FnsToEvaluate)
    ValuesOnDist_Kron=nan(numFnsToEvaluate,N_a,N_z,N_j,'gpuArray');
end
ValuesOnGrid=struct();

% Set default of grouping all the PTypes together when reporting statistics
if ~exist('simoptions','var')
    simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    simoptions.verbose=0;
    simoptions.verboseparams=0;
else
    if ~isfield(simoptions,'ptypestorecpu')
        simoptions.ptypestorecpu=1; % GPU memory is limited, so switch solutions to the cpu
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=1;
    end
    if ~isfield(simoptions,'verboseparams')
        simoptions.verboseparams=0;
    end
end
% Note: pass to subcommand EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(), so no need to handle alreadygridvals and the like as those can be done there.


%%
for ii=1:N_i % First set up simoptions
    simoptions_temp=PType_Options(simoptions,Names_i,ii);  % Note: already check for existence of simoptions and created it if it was not inputted
    
    if simoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
    
    if simoptions_temp.ptypestorecpu==1 % Things are being stored on cpu but solved on gpu
        PolicyIndexes_temp=gpuArray(Policy.(Names_i{ii}));
    else
        PolicyIndexes_temp=Policy.(Names_i{ii});
    end
    if isfield(simoptions_temp,'parallel')
        if simoptions.parallel~=2
            PolicyIndexes_temp=gather(PolicyIndexes_temp);
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


    %% Exogenous shocks
    if isstruct(z_grid)
        z_grid_temp=z_grid.(Names_i{ii});
    else
        nn=size(z_grid,ndims(z_grid));
        if nn==N_i
            otherdims = repmat({':'},1,ndims(z_grid)-1);
            z_grid_temp=z_grid(otherdims{:},ii);
        else
            z_grid_temp=z_grid;
        end
    end

    % e
    if isfield(simoptions_temp,'n_e')
        % If simoptions_temp.e_grid is a structure that was already dealt with by PType_Options() command
        if ~isstruct(simoptions.e_grid)
            % So just need to check if last dimension is of length N_i
            nn=size(simoptions_temp.e_grid,ndims(simoptions_temp.e_grid));
            if nn==N_i
                otherdims = repmat({':'},1,ndims(simoptions_temp.e_grid)-1);
                simoptions_temp.e_grid=simoptions_temp.e_grid(otherdims{:},ii);
            end
        end
    end

    % semiz
    if isfield(simoptions_temp,'n_semiz')
        % If simoptions_temp.semiz_grid is a structure that was already dealt with by PType_Options() command
        if ~isstruct(simoptions.semiz_grid)
            % So just need to check if last dimension is of length N_i
            nn=size(simoptions_temp.semiz_grid,ndims(simoptions_temp.semiz_grid));
            if nn==N_i
                otherdims = repmat({':'},1,ndims(simoptions_temp.semiz_grid)-1);
                simoptions_temp.semiz_grid=simoptions_temp.semiz_grid(otherdims{:},ii);
            end
        end
    end

    %% Parameters
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

    
    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are necessarily the same.
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=1;
    end
    l_a_temp=length(n_a_temp);
    l_z_temp=length(n_z_temp);  
    [FnsToEvaluate_temp,FnsToEvaluateParamNames_temp, WhichFnsForCurrentPType,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    
    ValuesOnGrid_ii=EvalFnOnAgentDist_ValuesOnGrid_FHorz_Case1(PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, simoptions_temp);
    
    n_ze_temp=[];
    if isfield(simoptions_temp,'n_semiz') && prod(simoptions_temp.n_semiz)>0
        n_ze_temp=[n_ze_temp,simoptions_temp.n_semiz];
    end
    if prod(n_z_temp)>0
        n_ze_temp=[n_ze_temp,n_z_temp];
    end
    if isfield(simoptions_temp,'n_e') && prod(simoptions_temp.n_e)>0
        n_ze_temp=[n_ze_temp,simoptions_temp.n_e];
    end
    
    if isempty(n_ze_temp) % no exogenous states
        if isstruct(FnsToEvaluate)
            FnNames=fieldnames(FnsToEvaluate);
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    if simoptions.ptypestorecpu==0
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,N_j_temp]);
                    else
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=gather(reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,N_j_temp]));
                    end
                end
            end

        else % Note: this only works when all agents use same grid
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    ValuesOnDist_Kron(kk,:,:,:)=ValuesOnGrid_ii(jj,:,:,:);
                end
            end
            if simoptions.ptypestorecpu==0
                ValuesOnGrid.(Names_i{ii})=reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,N_j_temp]);
            else
                ValuesOnGrid.(Names_i{ii})=gather(reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,N_j_temp]));
            end
        end
    else
        if isstruct(FnsToEvaluate)
            FnNames=fieldnames(FnsToEvaluate);
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    if simoptions.ptypestorecpu==0
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,n_ze_temp,N_j_temp]);
                    else
                        ValuesOnGrid.(FnNames{kk}).(Names_i{ii})=gather(reshape(ValuesOnGrid_ii.(FnNames{kk}),[n_a_temp,n_ze_temp,N_j_temp]));
                    end
                end
            end

        else % Note: this only works when all agents use same grid
            for kk=1:numFnsToEvaluate
                jj=WhichFnsForCurrentPType(kk);
                if jj>0
                    ValuesOnDist_Kron(kk,:,:,:)=ValuesOnGrid_ii(jj,:,:,:);
                end
            end
            if simoptions.ptypestorecpu==0
                ValuesOnGrid.(Names_i{ii})=reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,n_ze_temp,N_j_temp]);
            else
                ValuesOnGrid.(Names_i{ii})=gather(reshape(ValuesOnDist_Kron,[numFnsToEvaluate,n_a_temp,n_ze_temp,N_j_temp]));
            end
        end
    end    
end

end
