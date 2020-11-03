function AggVars=EvalFnOnAgentDist_AggVars_PType(StationaryDist, Policy,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid,Phi_aprime, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, options, AgeDependentGridParamNames)
% Allows for different permanent (fixed) types of agent.
% See ValueFnIter_PType for general idea.
%
% Rest of this description describes how those inputs not already used for
% ValueFnIter_PType or StationaryDist_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
% PTypeDistNames can either be a cell which tells you the parameter name relating to the weights, or just a vector of the weights
% themselves
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
    N_i=Names_i;
    Names_i={'pt1'};
    for ii=2:N_i
        Names_i{ii}=['pt',num2str(ii)];
    end
end

numFnsToEvaluate=length(FnsToEvaluate);

if isa(StationaryDist.(Names_i{1}), 'gpuArray')
    AggVars=zeros(numFnsToEvaluate,1,'gpuArray');
else
    AggVars=zeros(numFnsToEvaluate,1);
end

for ii=1:N_i
    
    if exist('options','var') % options.verbose (allowed to depend on permanent type)
        if ~isempty(options)
            options_temp=options; % some options will differ by permanent type, will clean these up as we go before they are passed
            if length(options.verbose)==1
                if options.verbose==1
                    sprintf('Permanent type: %i of %i',ii, N_i)
                end
            else
                if options.verbose(ii)==1
                    sprintf('Permanent type: %i of %i',ii, N_i)
                    options_temp.verbose=options.verbose(ii);
                end
            end
        else % isempty(options)
            options_temp.verbose=0;
        end
    else
        options_temp.verbose=0;
    end
    
    PolicyIndexes_temp=Policy.(Names_i{ii});
    StationaryDist_temp=StationaryDist.(Names_i{ii});
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
    finitehorz=0;
    if isstruct(N_j)
        if isfield(N_j, Names_i{ii})
            if isfinite(N_j.(Names_i{ii}))
                finitehorz=1;
                N_j_temp=N_j.(Names_i{ii});
                % else
                % % do nothing: finitehorz=0
            end
            % else
            % % do nothing: finitehorz=0
        end
    elseif ~isempty(N_j)
        if isfinite(N_j(ii))
            finitehorz=1;
            N_j_temp=N_j(ii);
            %         else
            %             % do nothing: finitehorz=0
        end
    % else % in situtation of isempty(N_j)
        % do nothing: finitehorz=0

    end
    
    % Case 1 or Case 2 is determined via Phi_aprime
    if exist('Phi_aprime','var') % If all the Permanent Types are 'Case 1' then there will be no Phi_aprime
        if isa(Phi_aprime,'struct')
            if isfield(Phi_aprime,Names_i{ii})==1 % Check if it exists for the current permanent type
                Case1orCase2=2;
            else
                Case1orCase2=1;
            end
        else
            % if Phi_aprime is not a structure then it must be relevant for all permanent types
            Case1orCase2=2;
        end
    elseif isempty(Phi_aprime)
        Case1orCase2=1;
    else
        Case1orCase2=1;
    end
    
    % Now that we have finitehorz and Case1orCase2, do everything else for the current permanent type.
    
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
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType, it should be the row dimension, if it is not then give a warning.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                sprintf('Possible Warning: some parameters appear to have been imputted with dependence on permanent type indexed by column rather than row \n')
                sprintf(['Specifically, parameter: ', FullParamNames{kField}, ' \n'])
                sprintf('(it is possible this is just a coincidence of number of columns) \n')
                dbstack
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    if options_temp.verbose==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    
    
    % Check for some options that may depend on permanent type (already
    % dealt with verbose and agedependentgrids)
    if exist('options','var')
        if isfield(options,'dynasty')
            if isa(options.dynasty,'struct')
                if isfield(options.dynasty, Names_i{ii})
                    options_temp.dynasty=options.dynasty.(Names_i{ii});
                else
                    options_temp.dynasty=0; % the default value
                end
            elseif prod(size(options.dynasty))~=1
                options_temp.dynasty=options.dynasty(ii);
            end
        end
        if isfield(options,'parallel')
            if isa(options.parallel, 'struct')
                if isfield(options.parallel, Names_i{ii})
                    options_temp.parallel=options.parallel.(Names_i{ii});
                else
                    options_temp.parallel=2; % the default value
                end
            elseif prod(size(options.parallel))~=1
                options_temp.parallel=options.parallel(ii);
            end
        end
        if isfield(options,'agedependentgrids')
            if isa(options.agedependentgrids, 'struct')
                if isfield(options.agedependentgrids, Names_i{ii})
                    options_temp.agedependentgrids=options.agedependentgrids.(Names_i{ii});
                else
                    % The current permanent type does not use age dependent grids.
                    options_temp=rmfield(options_temp,'agedependentgrids');
                end
            else
                temp=size(options.agedependentgrids);
                if temp(1)>1 % So different permanent types use different settings for age dependent grids
                    if prod(temp(ii,:))>0
                        options_temp.agedependentgrids=options.agedependentgrids(ii,:);
                    else
                        options_temp=rmfield(options_temp,'agedependentgrids');
                    end
                    %                 else
                    %                     % do nothing
                end
            end
        end
        % Now that we have figured out if we are using agedependentgrids
        % and stored this in options_temp we can use this to figure out if
        % we need AgeDependentGridParamNames_temp
        if isfield(options_temp,'agedependentgrids')
            if isa(AgeDependentGridParamNames.d_grid,'struct')
                AgeDependentGridParamNames_temp.d_grid=AgeDependentGridParamNames.d_grid.(Names_i{ii}); % Different grids by permanent type
            else
                AgeDependentGridParamNames_temp.d_grid=AgeDependentGridParamNames.d_grid;
            end
            if isa(AgeDependentGridParamNames.a_grid,'struct')
                AgeDependentGridParamNames_temp.a_grid=AgeDependentGridParamNames.a_grid.(Names_i{ii}); % Different grids by permanent type
            else
                AgeDependentGridParamNames_temp.a_grid=AgeDependentGridParamNames.a_grid;
            end
            if isa(AgeDependentGridParamNames.z_grid,'struct')
                AgeDependentGridParamNames_temp.z_grid=AgeDependentGridParamNames.z_grid.(Names_i{ii}); % Different grids by permanent type
            else
                AgeDependentGridParamNames_temp.z_grid=AgeDependentGridParamNames.z_grid;
            end
        end
    end
    
    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
    % necessarily the same.
    FnsToEvaluate_temp={};
    FnsToEvaluateParamNames_temp=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten
%     numFnsToEvaluate=length(FnsToEvaluateFn); % Now done outside the for-ii-loop
    WhichFnsForCurrentPType=zeros(numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:numFnsToEvaluate
        if isa(FnsToEvaluate{kk},'struct')
            if isfield(FnsToEvaluate{kk}, Names_i{ii})
                FnsToEvaluate_temp{jj}=FnsToEvaluate{kk}.(Names_i{ii});
                if isa(FnsToEvaluateParamNames(kk).Names,'struct')
                    FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names.(Names_i{ii});
                else
                    FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names;
                end
                WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            FnsToEvaluate_temp{jj}=FnsToEvaluate{kk};
            FnsToEvaluateParamNames_temp(jj).Names=FnsToEvaluateParamNames(kk).Names;
            WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
        end
    end
    
    if finitehorz==0  % Infinite horizon
        if Case1orCase2==1
            StatsFromDist_AggVars_ii=EvalFnOnAgentDist_AggVars_Case1(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp);
        elseif Case1orCase2==2
            StatsFromDist_AggVars_ii=EvalFnOnAgentDist_AggVars_Case2(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp);
        end
    elseif finitehorz==1 % Finite horizon
        if Case1orCase2==1
            StatsFromDist_AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp);
        elseif Case1orCase2==2
            if exist('options','var')
                StatsFromDist_AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case2(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, options_temp, AgeDependentGridParamNames_temp);
            else
                StatsFromDist_AggVars_ii=EvalFnOnAgentDist_AggVars_FHorz_Case2(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluate_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp);
            end
        end
    end
        
%     if isa(PTypeDistNames, 'array')
%         PTypeWeight_ii=PTypeDistNames(ii);
%     else
%         PTypeWeight_ii=Parameters.(PTypeDistNames{1}).(Names_i{ii});
%     end
    PTypeWeight_ii=StationaryDist.ptweights(ii);
    
    for kk=1:numFnsToEvaluate
        jj=WhichFnsForCurrentPType(kk);
        if jj>0
            AggVars(kk,:)=AggVars(kk,:)+PTypeWeight_ii*StatsFromDist_AggVars_ii(jj,:);
        end
    end
    
end


end
