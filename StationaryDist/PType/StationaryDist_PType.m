function StationaryDist=StationaryDist_PType(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid,pi_z,Phi_aprime,Case2_Type,Parameters,PhiaprimeParamNames,PTypeDistNames,simoptions)
% Allows for different permanent (fixed) types of agent. 
% See ValueFnIter_PType for general idea.
%
% Rest of this description describes how those inputs not used for
% ValueFnIter_PType should be set up.
%
% jequaloneDist can either be same for all permanent types, or must be passed as a structure.
% AgeWeightParamNames is either same for all permanent types, or must be passed as a structure.
%
%
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
    Names_i={'ft1'};
    for ii=2:N_i
       Names_i{ii}=['ft',num2str(ii)];
    end
end

for ii=1:N_i

    if exist('simoptions','var') % simoptions.verbose (allowed to depend on permanent type)
        simoptions_temp=simoptions; % some simoptions will differ by permanent type, will clean these up as we go before they are passed
        if isfield(simoptions,'verbose')==1
            if length(simoptions.verbose)==1
                if simoptions.verbose==1
                    sprintf('Permanent type: %i of %i',ii, N_i)
                end
            else
                if simoptions.verbose(ii)==1
                    sprintf('Permanent type: %i of %i',ii, N_i)
                    simoptions_temp.verbose=simoptions.verbose(ii);
                end
            end
        else
            simoptions_temp.verbose=0;
        end
    end
           
    
    Policy_temp=Policy.(Names_i{ii});
    
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
                %         names=fieldnames(Phi_aprime);
                Case1orCase2=2;
                Case2_Type_temp=Case2_Type.(Names_i{ii});
                Phi_aprime_temp=Phi_aprime.(Names_i{ii});
            else
                Case1orCase2=1;
                Case2_Type_temp=Case2_Type;
                Phi_aprime_temp=Phi_aprime;
            end
        elseif isempty(Phi_aprime)
            Case1orCase2=1;
        else
            % if Phi_aprime is not a structure then it must be relevant for all permanent types
            Case1orCase2=2;
            Case2_Type_temp=Case2_Type;
            Phi_aprime_temp=Phi_aprime;
        end
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
    
    if finitehorz==1 && Case1orCase2==2 % Only needed for simulation in this case
        d_grid_temp=d_grid;
        if isa(d_grid,'struct')
            d_grid_temp=d_grid.(Names_i{ii});
        end
        a_grid_temp=a_grid;
        if isa(a_grid,'struct')
            a_grid_temp=a_grid.(Names_i{ii});
        end
        z_grid_temp=z_grid;
        if isa(z_grid,'struct')
            z_grid_temp=z_grid.(Names_i{ii});
        end
    end
    
    pi_z_temp=pi_z;
    % If using 'agedependentgrids' then pi_z will actually be the AgeDependentGridParamNames, which is a structure. 
    % Following gets complicated as pi_z being a structure could be because
    % it depends just on age, or on permanent type, or on both.
    if exist('simoptions','var')
        if isfield(simoptions,'agedependentgrids')
            if isa(simoptions.agedependentgrids, 'struct')
                if isfield(simoptions.agedependentgrids, Names_i{ii})
                    simoptions_temp.agedependentgrids=simoptions.agedependentgrids.(Names_i{ii});
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    pi_z_temp=pi_z.(Names_i{ii});
                else
                    % The current permanent type does not use age dependent grids.
                    simoptions_temp=rmfield(simoptions_temp,'agedependentgrids');
                    % Different grids by permanent type (some of them must be using agedependentgrids even though not the current permanent type), but not depending on age.
                    pi_z_temp=pi_z.(Names_i{ii});
                end
            else
                temp=size(simoptions.agedependentgrids);
                if temp(1)>1 % So different permanent types use different settings for age dependent grids
                    if prod(temp(ii,:))>0
                        simoptions_temp.agedependentgrids=simoptions.agedependentgrids(ii,:);
                    else
                        simoptions_temp=rmfield(simoptions_temp,'agedependentgrids');
                    end
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    pi_z_temp=pi_z.(Names_i{ii});
                else % Grids depend on age, but not on permanent type (at least the function does not, you could set it up so that this is handled by the same function but a parameter whose value differs by permanent type
                    pi_z_temp=pi_z;
                end
            end
        elseif isa(pi_z,'struct')
            pi_z_temp=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age.
        end
    elseif isa(pi_z,'struct')
        pi_z_temp=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occour with or without the existence of simoptions, as long as there is no simoptions.agedependentgrids)
    end
    
    if finitehorz==1 % Only needed for simulation in this case
        
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
        
        if simoptions_temp.verbose==1
            sprintf('Parameter values for the current permanent type')
            Parameters_temp
        end
        
        jequaloneDist_temp=jequaloneDist;
        if isa(jequaloneDist,'struct')
            if isfield(jequaloneDist,Names_i{ii})
                jequaloneDist_temp=jequaloneDist.(Names_i{ii});
            else
                if isfinite(N_j_temp)
                    sprintf(['ERROR: You must input jequaloneDist for permanent type ', Names_i{ii}, ' \n'])
                    dbstack
                end
            end
        end
        AgeWeightParamNames_temp=AgeWeightParamNames;
        if isa(AgeWeightParamNames,'struct')
            if isfield(AgeWeightParamNames,Names_i{ii})
                AgeWeightParamNames_temp=AgeWeightParamNames.(Names_i{ii});
            else
                if isfinite(N_j_temp)
                    sprintf(['ERROR: You must input AgeWeightParamNames for permanent type ', Names_i{ii}, ' \n'])
                    dbstack
                end
            end
        end
    end

    % The parameter names can be made to depend on the permanent-type
    if Case1orCase2==2
        PhiaprimeParamNames_temp=PhiaprimeParamNames;
        if isa(PhiaprimeParamNames,'struct')
            PhiaprimeParamNames_temp=PhiaprimeParamNames.(Names_i{ii});
        end
    end
    
    % Check for some simoptions that may depend on permanent type (already
    % dealt with verbose and agedependentgrids)
    if exist('simoptions','var')
        if isfield(simoptions,'dynasty')
            if isa(simoptions.dynasty,'struct')
                if isfield(simoptions.dynasty, Names_i{ii})
                    simoptions_temp.dynasty=simoptions.dynasty.(Names_i{ii});
                else
                    simoptions_temp.dynasty=0; % the default value
                end
            elseif prod(size(simoptions.dynasty))~=1
                simoptions_temp.dynasty=simoptions.dynasty(ii);
            end
        end
        if isfield(simoptions,'lowmemory')
            if isa(simoptions.lowmemory, 'struct')
                if isfield(simoptions.lowmemory, Names_i{ii})
                    simoptions_temp.lowmemory=simoptions.lowmemory.(Names_i{ii});
                else
                    simoptions_temp.lowmemory=0; % the default value
                end
            elseif prod(size(simoptions.lowmemory))~=1
                simoptions_temp.lowmemory=simoptions.lowmemory(ii);
            end
        end
        if isfield(simoptions,'parallel')
            if isa(simoptions.parallel, 'struct')
                if isfield(simoptions.parallel, Names_i{ii})
                    simoptions_temp.parallel=simoptions.parallel.(Names_i{ii});
                else
                    simoptions_temp.parallel=3; % the default value
                end
            elseif prod(size(simoptions.parallel))~=1
                simoptions_temp.parallel=simoptions.parallel(ii);
            end
        end
        if isfield(simoptions,'nsims')
            if isa(simoptions.nsims, 'struct')
                if isfield(simoptions.nsims, Names_i{ii})
                    simoptions_temp.nsims=simoptions.nsims.(Names_i{ii});
                else
                    simoptions_temp.nsims=10^4; % the default value
                end
            elseif prod(size(simoptions.nsims))~=1
                simoptions_temp.nsims=simoptions.nsims(ii);
            end
        end
        if isfield(simoptions,'ncores')
            if isa(simoptions.ncores, 'struct')
                if isfield(simoptions.ncores, Names_i{ii})
                    simoptions_temp.ncores=simoptions.ncores.(Names_i{ii});
                else
                    simoptions_temp.ncores=1; % the default value
                end
            elseif prod(size(simoptions.ncores))~=1
                simoptions_temp.nsims=simoptions.ncores(ii);
            end
        end
        if isfield(simoptions,'iterate')
            if isa(simoptions.iterate, 'struct')
                if isfield(simoptions.iterate, Names_i{ii})
                    simoptions_temp.iterate=simoptions.iterate.(Names_i{ii});
                else
                    simoptions_temp.iterate=1; % the default value
                end
            elseif prod(size(simoptions.iterate))~=1
                simoptions_temp.nsims=simoptions.iterate(ii);
            end
        end
        if isfield(simoptions,'tolerance')
            if isa(simoptions.tolerance, 'struct')
                if isfield(simoptions.tolerance, Names_i{ii})
                    simoptions_temp.tolerance=simoptions.tolerance.(Names_i{ii});
                else
                    simoptions_temp.tolerance=1; % the default value
                end
            elseif prod(size(simoptions.tolerance))~=1
                simoptions_temp.nsims=simoptions.tolerance(ii);
            end
        end
    end
	
    if finitehorz==0  % Infinite horizon
        if Case1orCase2==1
            if exist('simoptions','var')
                StationaryDist_ii=StationaryDist_Case1(Policy_temp,n_d_temp,n_a_temp,n_z_temp,pi_z_temp,simoptions_temp);
            else
                StationaryDist_ii=StationaryDist_Case1(Policy_temp,n_d_temp,n_a_temp,n_z_temp,pi_z_temp);
            end
        elseif Case1orCase2==2
            if exist('simoptions','var')
                StationaryDist_ii=StationaryDist_Case2(Policy_temp,Phi_aprime_temp,Case2_Type_temp,n_d_temp,n_a_temp,n_z_temp,pi_z_temp,simoptions_temp);
            else
                StationaryDist_ii=StationaryDist_Case2(Policy_temp,Phi_aprime_temp,Case2_Type_temp,n_d_temp,n_a_temp,n_z_temp,pi_z_temp);
            end
        end
    elseif finitehorz==1 % Finite horizon
        % Check for some relevant simoptions that may depend on permanent type
        % dynasty, agedependentgrids, lowmemory, (parallel??)
        if Case1orCase2==1
            if exist('simoptions','var')
                StationaryDist_ii=StationaryDist_FHorz_Case1(jequaloneDist_temp,AgeWeightParamNames_temp,Policy_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,pi_z_temp,Parameters_temp,simoptions_temp);
            else
                StationaryDist_ii=StationaryDist_FHorz_Case1(jequaloneDist_temp,AgeWeightParamNames_temp,Policy_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,pi_z_temp,Parameters_temp);
            end
        elseif Case1orCase2==2
            if exist('simoptions','var')
                sum(sum(sum(sum(jequaloneDist_temp))))
                StationaryDist_ii=StationaryDist_FHorz_Case2(jequaloneDist_temp,AgeWeightParamNames_temp,Policy_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp,pi_z_temp,Phi_aprime_temp,Case2_Type_temp,Parameters_temp,PhiaprimeParamNames_temp,simoptions_temp);
            else
                StationaryDist_ii=StationaryDist_FHorz_Case2(jequaloneDist_temp,AgeWeightParamNames_temp,Policy_temp,n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp,pi_z_temp,Phi_aprime_temp,Case2_Type_temp,Parameters_temp,PhiaprimeParamNames_temp);
            end
        end
    end
	
    StationaryDist.(Names_i{ii})=StationaryDist_ii;

end

StationaryDist.ftweights=Parameters.(PTypeDistNames{:});

end
