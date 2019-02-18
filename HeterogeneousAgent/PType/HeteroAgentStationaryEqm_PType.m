function [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_PType(n_d, n_a, n_z, N_j, Names_i, n_p, pi_z, d_grid, a_grid, z_grid,jequaloneDist, Phi_aprime, Case2_Type, ReturnFn, FnsToEvaluateFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, AgeWeightParamNames, PTypeDistNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions, simoptions, vfoptions)
% Inputting vfoptions and simoptions is optional (they are not required inputs)
%
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
%
% Names_i can either be a cell containing the 'names' of the different
% permanent types, or if there are no structures used (just parameters that
% depend on permanent type and inputted as vectors or matrices as appropriate) 
% then Names_i can just be the number of permanent types (but does not have to be, can still be names).



%%
N_p=prod(n_p);
l_p=length(n_p);

if exist('heteroagentoptions','var')==0
    heteroagentoptions.multiGEcritereon=1;
    heteroagentoptions.verbose=0;
    heteroagentoptions.parallel=2;
    heteroagentoptions.fminalgo=1; % use fminsearch
else
    if isfield(heteroagentoptions,'multiGEcriterion')==0
        heteroagentoptions.multiGEcriterion=1;
    end
    if N_p~=0
        if isfield(heteroagentoptions,'p_grid')==0
            disp('ERROR: you have set n_p to a non-zero value, but not declared heteroagentoptions.pgrid')
            dbstack
        end
    end
    if isfield(heteroagentoptions,'verbose')==0
        heteroagentoptions.verbose=0;
    end
    if isfield(heteroagentoptions,'fminalgo')==0
        heteroagentoptions.fminalgo=1; % use fminsearch
    end
    if isfield(heteroagentoptions,'parallel')==0
        heteroagentoptions.parallel=2;
    end
end

%%
% PTypeStructure contains everything for the different permanent types.
% There are two fields, PTypeStructure.Names_i, and PTypeStructure.N_i.
% Then everything else is stored in, eg., PTypeStructure.ptype004, for the
% 4th permanent type.
%
% Code implicitly assumes that simoptions.agedependentgrids contains the same as
% vfoptions.agedependentgrids. Seems likely that you would always want this
% to be the case anyway.

% PTypeStructure.Names_i never really gets used. Just makes things easier
% to read when you are looking at PTypeStructure (which only ever exists
% internally to the VFI Toolkit)
if iscell(Names_i)
    PTypeStructure.Names_i=Names_i;
    PTypeStructure.N_i=length(Names_i);
else
    PTypeStructure.N_i=Names_i;
    PTypeStructure.Names_i={'ptype001'};
    for ii=2:PTypeStructure.N_i
        if ii<10
            PTypeStructure.Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            PTypeStructure.Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            PTypeStructure.Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

for ii=1:PTypeStructure.N_i

    % Create all the things specific for each Permanent type and store them all in PTypeStructure.
    if ii<10 % one digit
        iistr=['ptype00',num2str(ii)];
    elseif ii<100 % two digit
        iistr=['ptype0',num2str(ii)];
    elseif ii<1000 % three digit
        iistr=['ptype',num2str(ii)];
    end
    PTypeStructure.iistr{ii}=iistr;
    
    if exist('vfoptions','var') % vfoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).vfoptions=vfoptions; % some vfoptions will differ by permanent type, will clean these up as we go before they are passed
        if length(vfoptions.verbose)==1
            if vfoptions.verbose==1
                sprintf('Permanent type: %i of %i',ii, N_i)
            end
        else
            if vfoptions.verbose(ii)==1
                sprintf('Permanent type: %i of %i',ii, N_i)
                PTypeStructure.(iistr).vfoptions.verbose=vfoptions.verbose(ii);
            end
        end
    else
        PTypeStructure.(iistr).vfoptions.verbose=0;
    end
    if exist('simoptions','var') % simoptions.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).simoptions=simoptions; % some simoptions will differ by permanent type, will clean these up as we go before they are passed
        if length(simoptions.verbose)==1
            if simoptions.verbose==1
                sprintf('Permanent type: %i of %i',ii, N_i)
            end
        else
            if simoptions.verbose(ii)==1
                sprintf('Permanent type: %i of %i',ii, N_i)
                PTypeStructure.(iistr).simoptions.verbose=simoptions.verbose(ii);
            end
        end
    else
        PTypeStructure.(iistr).simoptions.verbose=0;
    end
    if exist('options','var') % options.verbose (allowed to depend on permanent type)
        PTypeStructure.(iistr).vfoptions=options; % some options will differ by permanent type, will clean these up as we go before they are passed
        if length(options.verbose)==1
            if options.verbose==1
                sprintf('Permanent type: %i of %i',ii, N_i)
            end
        else
            if options.verbose(ii)==1
                sprintf('Permanent type: %i of %i',ii, N_i)
                PTypeStructure.(iistr).vfoptions.verbose=options.verbose(ii);
            end
        end
    else
        PTypeStructure.(iistr).vfoptions.verbose=0;
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
    PTypeStructure.(iistr).finitehorz=0;
    if isstruct(N_j)
        if isfield(N_j, Names_i{ii})
            if isfinite(N_j.(Names_i{ii}))
                PTypeStructure.(iistr).finitehorz=1;
                PTypeStructure.(iistr).N_j=N_j.(Names_i{ii});
                % else
                % % do nothing: PTypeStructure.(iistr).finitehorz=0
            end
            % else
                % % do nothing: PTypeStructure.(iistr).finitehorz=0
        end
    else
        if isfinite(N_j(ii))
            PTypeStructure.(iistr).finitehorz=1;
            PTypeStructure.(iistr).N_j=N_j(ii);
%         else
%             % do nothing: PTypeStructure.(iistr).finitehorz=0
        end
    end
    
    % Case 1 or Case 2 is determined via Phi_aprime
    if exist('Phi_aprime','var') % If all the Permanent Types are 'Case 1' then there will be no Phi_aprime
         if isa(Phi_aprime,'struct')
            if isfield(Phi_aprime,Names_i{ii})==1 % Check if it exists for the current permanent type
                %         names=fieldnames(Phi_aprime);
                PTypeStructure.(iistr).Case1orCase2=2;
                PTypeStructure.(iistr).Case2_Type=Case2_Type.(Names_i{ii});
                PTypeStructure.(iistr).Phi_aprime=Phi_aprime.(Names_i{ii});
            else
                PTypeStructure.(iistr).Case1orCase2=1;
                PTypeStructure.(iistr).Case2_Type=Case2_Type;
                PTypeStructure.(iistr).Phi_aprime=Phi_aprime;
            end
        else
            % if Phi_aprime is not a structure then it must be relevant for all permanent types
            PTypeStructure.(iistr).Case1orCase2=2;
            PTypeStructure.(iistr).Case2_Type=Case2_Type;
            PTypeStructure.(iistr).Phi_aprime=Phi_aprime;
        end
    else
        PTypeStructure.(iistr).Case1orCase2=1;
    end
    
    % Now that we have PTypeStructure.(iistr).finitehorz and PTypeStructure.(iistr).Case1orCase2, do everything else for the current permanent type.

    PTypeStructure.(iistr).n_d=n_d;
    if isa(n_d,'struct')
        PTypeStructure.(iistr).n_d=n_d.(Names_i{ii});
    else
        temp=size(n_d);
        if temp(1)>1 % n_d depends on fixed type
            PTypeStructure.(iistr).n_d=n_d(ii,:);
        elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            sprintf('Possible Warning: Number of columns of n_d is the same as the number of permanent types. \n This may just be coincidence as number of d variables is equal to number of permanent types. \n If they are intended to be permanent types then n_d should have them as different rows (not columns). \n')
        end
    end
    PTypeStructure.(iistr).n_a=n_a;
    if isa(n_a,'struct')
        PTypeStructure.(iistr).n_a=n_a.(Names_i{ii});
    else
        temp=size(n_a);
        if temp(1)>1 % n_a depends on fixed type
            PTypeStructure.(iistr).n_a=n_a(ii,:);
        elseif temp(2)==N_i % If there is one row, but number of elements in n_a happens to coincide with number of permanent types, then just let user know
            sprintf('Possible Warning: Number of columns of n_a is the same as the number of permanent types. \n This may just be coincidence as number of a variables is equal to number of permanent types. \n If they are intended to be permanent types then n_a should have them as different rows (not columns). \n')
            dbstack
        end
    end
    PTypeStructure.(iistr).n_z=n_z;
    if isa(n_z,'struct')
        PTypeStructure.(iistr).n_z=n_z.(Names_i{ii});
    else
        temp=size(n_z);
        if temp(1)>1 % n_z depends on fixed type
            PTypeStructure.(iistr).n_z=n_z(ii,:);
        elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
            sprintf('Possible Warning: Number of columns of n_z is the same as the number of permanent types. \n This may just be coincidence as number of z variables is equal to number of permanent types. \n If they are intended to be permanent types then n_z should have them as different rows (not columns). \n')
            dbstack
        end
    end
    
    PTypeStructure.(iistr).d_grid=d_grid;
    if isa(d_grid,'struct')
        PTypeStructure.(iistr).d_grid=d_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).a_grid=a_grid;
    if isa(a_grid,'struct')
        PTypeStructure.(iistr).a_grid=a_grid.(Names_i{ii});
    end
    PTypeStructure.(iistr).z_grid=z_grid;
    if isa(z_grid,'struct')
        PTypeStructure.(iistr).z_grid=z_grid.(Names_i{ii});
    end
    
    PTypeStructure.(iistr).pi_z=pi_z;
    % If using 'agedependentgrids' then pi_z will actually be the AgeDependentGridParamNames, which is a structure. 
    % Following gets complicated as pi_z being a structure could be because
    % it depends just on age, or on permanent type, or on both.
    if exist('vfoptions','var')
        if isfield(vfoptions,'agedependentgrids')
            if isa(vfoptions.agedependentgrids, 'struct')
                if isfield(vfoptions.agedependentgrids, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.agedependentgrids=vfoptions.agedependentgrids.(Names_i{ii});
                    PTypeStructure.(iistr).simoptions.agedependentgrids=simoptions.agedependentgrids.(Names_i{ii});
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                else
                    % The current permanent type does not use age dependent grids.
                    PTypeStructure.(iistr).vfoptions=rmfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids');
                    PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'agedependentgrids');
                    % Different grids by permanent type (some of them must be using agedependentgrids even though not the current permanent type), but not depending on age.
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                end
            else
                temp=size(vfoptions.agedependentgrids);
                if temp(1)>1 % So different permanent types use different settings for age dependent grids
                    if prod(temp(ii,:))>0
                        PTypeStructure.(iistr).vfoptions.agedependentgrids=vfoptions.agedependentgrids(ii,:);
                        PTypeStructure.(iistr).simoptions.agedependentgrids=simoptions.agedependentgrids(ii,:);
                    else
                        PTypeStructure.(iistr).vfoptions=rmfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids');
                        PTypeStructure.(iistr).simoptions=rmfield(PTypeStructure.(iistr).simoptions,'agedependentgrids');
                    end
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii});
                else % Grids depend on age, but not on permanent type (at least the function does not, you could set it up so that this is handled by the same function but a parameter whose value differs by permanent type
                    PTypeStructure.(iistr).pi_z=pi_z;
                end
            end
        elseif isa(pi_z,'struct')
            PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age.
        end
    elseif isa(pi_z,'struct')
        PTypeStructure.(iistr).pi_z=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occour with or without the existence of vfoptions, as long as there is no vfoptions.agedependentgrids)
    end
    
    PTypeStructure.(iistr).ReturnFn=ReturnFn;
    if isa(ReturnFn,'struct')
        PTypeStructure.(iistr).ReturnFn=ReturnFn.(Names_i{ii});
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on permanent type). So go through each of
    % these in term.
    % ie. Parameters.alpha=[0;1]; or Parameters.alpha.ptype1=0; Parameters.alpha.ptype2=1;
    PTypeStructure.(iistr).Parameters=Parameters;
    FullParamNames=fieldnames(Parameters); % all the different parameters
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check the current parameter for permanent type in structure form
            % Check if this parameter is used for the current permanent type (it may or may not be, some parameters are only used be a subset of permanent types)
            if isfield(Parameters.(FullParamNames{kField}),Names_i{ii})
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(Names_i{ii});
            end
        elseif sum(size(Parameters.(FullParamNames{kField}))==PTypeStructure.N_i)>=1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==PTypeStructure.N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType, it should be the row dimension, if it is not then give a warning.
            if ptypedim==1
                PTypeStructure.(iistr).Parameters.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                sprintf('Possible Warning: some parameters appear to have been imputted with dependence on permanent type indexed by column rather than row \n')
                sprintf(['Specifically, parameter: ', FullParamNames{kField}, ' \n'])
                sprintf('(it is possible this is just a coincidence of number of columns) \n')
                dbstack
            end
        end
    end
    % THIS TREATMENT OF PARAMETERS COULD BE IMPROVED TO BETTER DETECT INPUT SHAPE ERRORS.
    
    if PTypeStructure.(iistr).finitehorz==1
        PTypeStructure.(iistr).jequaloneDist=jequaloneDist;
        if isa(jequaloneDist,'struct')
            if isfield(jequaloneDist,Names_i{ii})
                PTypeStructure.(iistr).jequaloneDist=jequaloneDist.(Names_i{ii});
            else
                if isfinite(PTypeStructure.(iistr).N_j)
                    sprintf(['ERROR: You must input jequaloneDist for permanent type ', Names_i{ii}, ' \n'])
                    dbstack
                end
            end
        end
    end
    
    % The parameter names can be made to depend on the permanent-type
    PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        PTypeStructure.(iistr).DiscountFactorParamNames=DiscountFactorParamNames.(Names_i{ii});
    end
    PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames;
    if isa(ReturnFnParamNames,'struct')
        PTypeStructure.(iistr).ReturnFnParamNames=ReturnFnParamNames.(Names_i{ii});
    end
    if PTypeStructure.(iistr).Case1orCase2==2
        PTypeStructure.(iistr).PhiaprimeParamNames=PhiaprimeParamNames;
        if isa(PhiaprimeParamNames,'struct')
            PTypeStructure.(iistr).PhiaprimeParamNames=PhiaprimeParamNames.(Names_i{ii});
        end
    end
    PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightParamNames;
    if isa(AgeWeightParamNames,'struct')
        if isfield(AgeWeightParamNames,Names_i{ii})
            PTypeStructure.(iistr).AgeWeightParamNames=AgeWeightParamNames.(Names_i{ii});
        else
            if isfinite(PTypeStructure.(iistr).N_j)
                sprintf(['ERROR: You must input AgeWeightParamNames for permanent type ', Names_i{ii}, ' \n'])
                dbstack
            end
        end
    end
    
    
    % Check for some vfoptions that may depend on permanent type (already
    % dealt with verbose and agedependentgrids)
    if exist('vfoptions','var')
        if isfield(vfoptions,'dynasty')
            if isa(vfoptions.dynasty,'struct')
                if isfield(vfoptions.dynasty, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.dynasty=vfoptions.dynasty.(Names_i{ii});
                else
                    PTypeStructure.(iistr).vfoptions.dynasty=0; % the default value
                end
            elseif prod(size(vfoptions.dynasty))~=1
                PTypeStructure.(iistr).vfoptions.dynasty=vfoptions.dynasty(ii);
            end
        end
        if isfield(vfoptions,'lowmemory')
            if isa(vfoptions.lowmemory, 'struct')
                if isfield(vfoptions.lowmemory, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.lowmemory=vfoptions.lowmemory.(Names_i{ii});
                else
                    PTypeStructure.(iistr).vfoptions.lowmemory=0; % the default value
                end
            elseif prod(size(vfoptions.lowmemory))~=1
                PTypeStructure.(iistr).vfoptions.lowmemory=vfoptions.lowmemory(ii);
            end
        end
        if isfield(vfoptions,'parallel')
            if isa(vfoptions.parallel, 'struct')
                if isfield(vfoptions.parallel, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.parallel=vfoptions.parallel.(Names_i{ii});
                else
                    PTypeStructure.(iistr).vfoptions.parallel=2; % the default value
                end
            elseif prod(size(vfoptions.parallel))~=1
                PTypeStructure.(iistr).vfoptions.parallel=vfoptions.parallel(ii);
            end
        end
        if isfield(vfoptions,'tolerance')
            if isa(vfoptions.tolerance, 'struct')
                if isfield(vfoptions.tolerance, Names_i{ii})
                    PTypeStructure.(iistr).vfoptions.tolerance=vfoptions.tolerance.(Names_i{ii});
                else
                    PTypeStructure.(iistr).vfoptions.tolerance=1; % the default value
                end
            elseif prod(size(vfoptions.tolerance))~=1
                PTypeStructure.(iistr).vfoptions.nsims=vfoptions.tolerance(ii);
            end
        end
    end
    
    % Check for some simoptions that may depend on permanent type (already
    % dealt with verbose and agedependentgrids)
    if exist('simoptions','var')
        if isfield(simoptions,'dynasty')
            if isa(simoptions.dynasty,'struct')
                if isfield(simoptions.dynasty, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.dynasty=simoptions.dynasty.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.dynasty=0; % the default value
                end
            elseif prod(size(simoptions.dynasty))~=1
                PTypeStructure.(iistr).simoptions.dynasty=simoptions.dynasty(ii);
            end
        end
        if isfield(simoptions,'lowmemory')
            if isa(simoptions.lowmemory, 'struct')
                if isfield(simoptions.lowmemory, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.lowmemory=simoptions.lowmemory.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.lowmemory=0; % the default value
                end
            elseif prod(size(simoptions.lowmemory))~=1
                PTypeStructure.(iistr).simoptions.lowmemory=simoptions.lowmemory(ii);
            end
        end
        if isfield(simoptions,'parallel')
            if isa(simoptions.parallel, 'struct')
                if isfield(simoptions.parallel, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.parallel=simoptions.parallel.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.parallel=3; % the default value
                end
            elseif prod(size(simoptions.parallel))~=1
                PTypeStructure.(iistr).simoptions.parallel=simoptions.parallel(ii);
            end
        end
        if isfield(simoptions,'nsims')
            if isa(simoptions.nsims, 'struct')
                if isfield(simoptions.nsims, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.nsims=simoptions.nsims.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.nsims=10^4; % the default value
                end
            elseif prod(size(simoptions.nsims))~=1
                PTypeStructure.(iistr).simoptions.nsims=simoptions.nsims(ii);
            end
        end
        if isfield(simoptions,'ncores')
            if isa(simoptions.ncores, 'struct')
                if isfield(simoptions.ncores, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.ncores=simoptions.ncores.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.ncores=1; % the default value
                end
            elseif prod(size(simoptions.ncores))~=1
                PTypeStructure.(iistr).simoptions.nsims=simoptions.ncores(ii);
            end
        end
        if isfield(simoptions,'iterate')
            if isa(simoptions.iterate, 'struct')
                if isfield(simoptions.iterate, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.iterate=simoptions.iterate.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.iterate=1; % the default value
                end
            elseif prod(size(simoptions.iterate))~=1
                PTypeStructure.(iistr).simoptions.nsims=simoptions.iterate(ii);
            end
        end
        if isfield(simoptions,'tolerance')
            if isa(simoptions.tolerance, 'struct')
                if isfield(simoptions.tolerance, Names_i{ii})
                    PTypeStructure.(iistr).simoptions.tolerance=simoptions.tolerance.(Names_i{ii});
                else
                    PTypeStructure.(iistr).simoptions.tolerance=1; % the default value
                end
            elseif prod(size(simoptions.tolerance))~=1
                PTypeStructure.(iistr).simoptions.nsims=simoptions.tolerance(ii);
            end
        end
    end
    
    % AgeDependentGridParamNames will be inputted in place of pi_z, so deal
    % with this when dealing with pi_z
%     % Now that we have figured out if we are using agedependentgrids
%     % and stored this in PTypeStructure.(iistr).vfoptions we can use this to figure out if
%     % we need PTypeStructure.(iistr).AgeDependentGridParamNames
%     if isfield(PTypeStructure.(iistr).vfoptions,'agedependentgrids')
%         if isa(AgeDependentGridParamNames.d_grid,'struct')
%             PTypeStructure.(iistr).AgeDependentGridParamNames.d_grid=AgeDependentGridParamNames.d_grid.(Names_i{ii}); % Different grids by permanent type
%         else
%             PTypeStructure.(iistr).AgeDependentGridParamNames.d_grid=AgeDependentGridParamNames.d_grid;
%         end
%         if isa(AgeDependentGridParamNames.a_grid,'struct')
%             PTypeStructure.(iistr).AgeDependentGridParamNames.a_grid=AgeDependentGridParamNames.a_grid.(Names_i{ii}); % Different grids by permanent type
%         else
%             PTypeStructure.(iistr).AgeDependentGridParamNames.a_grid=AgeDependentGridParamNames.a_grid;
%         end
%         if isa(AgeDependentGridParamNames.z_grid,'struct')
%             PTypeStructure.(iistr).AgeDependentGridParamNames.z_grid=AgeDependentGridParamNames.z_grid.(Names_i{ii}); % Different grids by permanent type
%         else
%             PTypeStructure.(iistr).AgeDependentGridParamNames.z_grid=AgeDependentGridParamNames.z_grid;
%         end
%     end
    
    % Figure out which functions are actually relevant to the present
    % PType. Only the relevant ones need to be evaluated.
    % The dependence of FnsToEvaluateFn and FnsToEvaluateFnParamNames are
    % necessarily the same.
    PTypeStructure.(iistr).FnsToEvaluateFn={};
    PTypeStructure.(iistr).FnsToEvaluateParamNames=struct(); %(1).Names={}; % This is just an initialization value and will be overwritten
    PTypeStructure.(iistr).numFnsToEvaluate=length(FnsToEvaluateFn);
    PTypeStructure.(iistr).WhichFnsForCurrentPType=zeros(PTypeStructure.(iistr).numFnsToEvaluate,1);
    jj=1; % jj indexes the FnsToEvaluate that are relevant to the current PType
    for kk=1:PTypeStructure.(iistr).numFnsToEvaluate
        if isa(FnsToEvaluateFn{kk},'struct')
            if isfield(FnsToEvaluateFn{kk}, Names_i{ii})
                PTypeStructure.(iistr).FnsToEvaluateFn{jj}=FnsToEvaluateFn{kk}.(Names_i{ii});
                if isa(FnsToEvaluateParamNames(kk).Names,'struct')
                    PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names.(Names_i{ii});
                else
                    PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names;
                end
                PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
                % else
                %  % do nothing as this FnToEvaluate is not relevant for the current PType
                % % Implicitly, PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=0
            end
        else
            % If the Fn is not a structure (if it is a function) it is assumed to be relevant to all PTypes.
            PTypeStructure.(iistr).FnsToEvaluateFn{jj}=FnsToEvaluateFn{kk};
            PTypeStructure.(iistr).FnsToEvaluateParamNames(jj).Names=FnsToEvaluateParamNames(kk).Names;
            PTypeStructure.(iistr).WhichFnsForCurrentPType(kk)=jj; jj=jj+1;
        end
    end
    
    
%     PTypeStructure.(iistr).PTypeWeight=nan;
    if isa(PTypeDistNames, 'array')
        PTypeStructure.(iistr).PTypeWeight=PTypeDistNames(ii);
    else
%         PTypeStructure.(iistr).PTypeWeight;
%         PTypeStructure.(iistr).Parameters
%         iistr
%         PTypeDistNames{1}
%         PTypeStructure.(iistr).Parameters.(PTypeDistNames{1}); % {1} as I simply assume there is only a single parameter (name) that contains the distribution (weights) of the PTypes.
%         PTypeStructure.(iistr).Parameters.(PTypeDistNames{1})
%         PTypeStructure.(iistr).Parameters.(PTypeDistNames{1}).(Names_i{ii});
        PTypeStructure.(iistr).PTypeWeight=PTypeStructure.(iistr).Parameters.(PTypeDistNames{1}); % Don't need '.(Names_i{ii}' as this was already done when putting it into PTypeStrucutre, and here I take it straing from PTypeStructure.(iistr).Parameters rather than from Parameters itself.
    end
end

%%
% Have now finished creating PTypeStructure. Time to do the actual finding the HeteroAgentStationaryEqm:
if heteroagentoptions.verbose==1
    for ii=1:PTypeStructure.N_i
        % Create all the things specific for each Permanent type and store them all in PTypeStructure.
        if ii<10 % one digit
            iistr=['ptype00',num2str(ii)];
        elseif ii<100 % two digit
            iistr=['ptype0',num2str(ii)];
        elseif ii<1000 % three digit
            iistr=['ptype',num2str(ii)];
        end
        PTypeStructure.(iistr)
    end
end


%%
if N_p~=0
    [p_eqm,p_eqm_index,GeneralEqmConditions]=HeteroAgentStationaryEqm_PType_pgrid(n_p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames, heteroagentoptions);
    return
end

%%  Otherwise, use fminsearch to find the general equilibrium

GeneralEqmConditionsFn=@(p) HeteroAgentStationaryEqm_PType_subfn(p, PTypeStructure, Parameters, GeneralEqmEqns, GeneralEqmEqnParamNames, GEPriceParamNames,heteroagentoptions)

p0=nan(length(GEPriceParamNames),1);
for ii=1:length(GEPriceParamNames)
    p0(ii)=Parameters.(GEPriceParamNames{ii});
end

if heteroagentoptions.fminalgo==0 % fzero doesn't appear to be a good choice in practice, at least not with it's default settings.
    heteroagentoptions.multimarketcriterion=0;
    [p_eqm,GeneralEqmConditions]=fzero(GeneralEqmConditionsFn,p0);    
elseif heteroagentoptions.fminalgo==1
    [p_eqm,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFn,p0);
else
    [p_eqm,GeneralEqmConditions]=fminsearch(GeneralEqmConditionsFn,p0);
end

p_eqm_index=nan; % If not using p_grid then this is irrelevant/useless


    
%     PolicyIndexes_temp=Policy.(Names_i{ii});
%     StationaryDist_temp=StationaryDist.(Names_i{ii});
%     if isa(StationaryDist_temp, 'gpuArray')
%         Parallel_temp=2;
%     else
%         Parallel_temp=1;
%     end
%     
%     if finitehorz==0  % Infinite horizon
%         % Infinite Horizon requires an initial guess of value function. For
%         % the present I simply don't let this feature be used when using
%         % permanent types. WOULD BE GOOD TO CHANGE THIS IN FUTURE SOMEHOW.
%         V_ii=zeros(prod(n_a_temp),prod(n_z_temp)); % The initial guess (note that its value is 'irrelevant' in the sense that global uniform convergence is anyway known to occour for VFI).
%         if Case1orCase2==1
%             if exist('vfoptions','var')
%                 [V_ii, Policy_ii]=ValueFnIter_Case1(V_ii,n_d_temp,n_a_temp,n_z_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, vfPTypeStructure.(iistr).vfoptions);
%             else
%                 [V_ii, Policy_ii]=ValueFnIter_Case1(V_ii,n_d_temp,n_a_temp,n_z_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp);
%             end
%         elseif Case1orCase2==2
%             if exist('vfoptions','var')
%                 [V_ii, Policy_ii]=ValueFnIter_Case2(V_ii,n_d_temp,n_a_temp,n_z_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, PhiaprimeParamNames_temp, vfPTypeStructure.(iistr).vfoptions);
%             else
%                 [V_ii, Policy_ii]=ValueFnIter_Case2(V_ii,n_d_temp,n_a_temp,n_z_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, PhiaprimeParamNames_temp);
%             end
%         end
%     elseif finitehorz==1 % Finite horizon
%         % Check for some relevant vfoptions that may depend on permanent type
%         % dynasty, agedependentgrids, lowmemory, (parallel??)
%         if Case1orCase2==1
%             if exist('vfoptions','var')
%                 [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, vfPTypeStructure.(iistr).vfoptions);
%             else
%                 [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp);
%             end
%         elseif Case1orCase2==2
%             if exist('vfoptions','var')
%                 [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, PhiaprimeParamNames_temp, vfPTypeStructure.(iistr).vfoptions);
%             else
%                 [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, PhiaprimeParamNames_temp);
%             end
%         end
%     end
%         
%     V.(Names_i{ii})=V_ii;
%     Policy.(Names_i{ii})=Policy_ii;    
    
%     if PTypeStructure.(iistr).finitehorz==0  % Infinite horizon
%         if PTypeStructure.(iistr).Case1orCase2==1
%             if exist('simoptions','var')
%                 StationaryDist_ii=StationaryDist_Case1(PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions);
%             else
%                 StationaryDist_ii=StationaryDist_Case1(PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z);
%             end
%         elseif PTypeStructure.(iistr).Case1orCase2==2
%             if exist('simoptions','var')
%                 StationaryDist_ii=StationaryDist_Case2(PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).Phi_aprime_temp,PTypeStructure.(iistr).Case2_Type_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).simoptions);
%             else
%                 StationaryDist_ii=StationaryDist_Case2(PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).Phi_aprime_temp,PTypeStructure.(iistr).Case2_Type_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).pi_z);
%             end
%         end
%     elseif PTypeStructure.(iistr).finitehorz==1 % Finite horizon
%         % Check for some relevant simoptions that may depend on permanent type
%         % dynasty, agedependentgrids, lowmemory, (parallel??)
%         if PTypeStructure.(iistr).Case1orCase2==1
%             if exist('simoptions','var')
%                 StationaryDist=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j_temp,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).simoptions);
%             else
%                 StationaryDist=StationaryDist_FHorz_Case1(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j_temp,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Parameters);
%             end
%         elseif PTypeStructure.(iistr).Case1orCase2==2
%             if exist('simoptions','var')
%                 StationaryDist_ii=StationaryDist_FHorz_Case2(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j_temp,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Phi_aprime_temp,PTypeStructure.(iistr).Case2_Type_temp,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).PhiaprimeParamNames,PTypeStructure.(iistr).simoptions);
%             else
%                 StationaryDist_ii=StationaryDist_FHorz_Case2(PTypeStructure.(iistr).jequaloneDist,PTypeStructure.(iistr).AgeWeightParamNames,PTypeStructure.(iistr).Policy_temp,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).N_j_temp,PTypeStructure.(iistr).d_grid, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_grid,PTypeStructure.(iistr).pi_z,PTypeStructure.(iistr).Phi_aprime_temp,PTypeStructure.(iistr).Case2_Type_temp,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).PhiaprimeParamNames);
%             end
%         end
%     end
% 	
%     StationaryDist.(Names_i{ii})=StationaryDist_ii;
% 
%     if finitehorz==0  % Infinite horizon
%         if Case1orCase2==1
%             StatsFromDist_AggVars_ii=SSvalues_AggVars_Case1(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluateFn_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp);
%         elseif Case1orCase2==2
%             StatsFromDist_AggVars_ii=SSvalues_AggVars_Case2(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluateFn_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp);
%         end
%     elseif finitehorz==1 % Finite horizon
%         if Case1orCase2==1
%             StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case1(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluateFn_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, Parallel_temp);
%         elseif Case1orCase2==2
%             if exist('options','var')
%                 StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case2(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluateFn_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp, options_temp, AgeDependentGridParamNames_temp);
%             else
%                 StatsFromDist_AggVars_ii=SSvalues_AggVars_FHorz_Case2(StationaryDist_temp, PolicyIndexes_temp, FnsToEvaluateFn_temp, Parameters_temp, FnsToEvaluateParamNames_temp, n_d_temp, n_a_temp, n_z_temp, N_j_temp, d_grid_temp, a_grid_temp, z_grid_temp);
%             end
%         end
%     end
%     
%     if isa(PTypeDistNames, 'array')
%         PTypeWeight_ii=PTypeDistNames(ii);
%     else
%         PTypeWeight_ii=Parameters.(PTypeDistNames{1}).(Names_i{ii});
%     end
%     
%     StatsFromDist_AggVars=zeros(PTypeStructure.(iistr).numFnsToEvaluate,1,'gpuArray');
%     for kk=1:PTypeStructure.(iistr).numFnsToEvaluate
%         jj=PTypeStructure.(iistr).WhichFnsForCurrentPType(kk);
%         if jj>0
%             StatsFromDist_AggVars(kk,:)=StatsFromDist_AggVars(kk,:)+PTypeWeight_ii*StatsFromDist_AggVars_ii(jj,:);
%         end
%     end


end
