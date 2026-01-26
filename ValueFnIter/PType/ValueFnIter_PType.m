function [V, Policy]=ValueFnIter_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)
% Allows for different permanent (fixed) types of agent. The different
% agent types can involve different time horizons and may be 'Case1 or
% Case2'. The vfoptions (such as 'dynasty' and 'AgeDependentGrids') can
% also differ by permanent type.
% How exactly to handle these differences between permanent (fixed) types
% is to some extent left to the user. You can, for example, input
% parameters that differ by permanent type as a vector with different rows f
% for each type, or as a structure with different fields for each type.
%
% The main 'extra' input required (compared to other ValueFnIter codes) is 
% Names_i. But many other inputs may be in different forms wherever they depend on the
% permanent type. When structures are used for the different permanent
% types Names_i should be a cell of the 'names' for each permanent type. When
% the names are not necessary Names_i can just be the number of permanent types.
%
% Any input that does not depend on the permanent type is just passed in
% exactly the same form as normal.
%
% For grids (d,a,z), the transition matrix on z, the return function, and
% Phiaprime function this can either be done by creating structure with each 
% permanent type being a different field in the structure (see the grids in 
% replication of Hubbard, Skinner & Zeldes (1994WP) for an example of this
% https://github.com/vfitoolkit/vfitoolkit-matlab-replication/tree/master/HubbardSkinnerZeldes1994
% Alternatively they can be inputted as a function, where one of the
% parameters depends on the permanent-type. (Note that if the grids are
% functions you will need to be using vfoptions.agedependentgrids, even if
% in principle they do not actually depend on age; I am too lazy to write
% extra code.)
%
% For parameters dependence on permanent-type can either be done by using a
% structure, again see the example of Hubbard, Skinner & Zeldes (1994WP).
% Or the dependence can be done by creating the parameters as matrices,
% where each row represents a different permanent type. When parameters
% depend on by permanent type and age they would be a matrix where row
% indicates permanent type and column indicates age.
%
% For 'parameter names' there are two options to allow them to depend on
% the permanent type. The first is just to ignore this, in the sense of
% anyway passing them all (the union across the parameters for each of the
% permanent types) to all of the functions, and then just having the
% functions only use them internally for some but not for other permanent types.
% The other option is to set them up as structures in the same way as is
% done elsewhere.
%

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

vfoptions_temp.verbose=0; % May change below

for ii=1:N_i
    % First set up vfoptions
    if exist('vfoptions','var')
        vfoptions_temp=PType_Options(vfoptions,Names_i,ii);
        if vfoptions_temp.verbose==1
            fprintf('Permanent type: %i of %i',ii, N_i)
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
         if isstruct(Phi_aprime)
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
    pt_temp=struct("n_d", n_d, "n_a", n_a, "n_z", n_z, "d_grid", d_grid, "a_grid", a_grid, "z_grid", z_grid);
    for n_x=["n_d" "n_a" "n_z"]
        if isstruct(pt_temp.(n_x))
            pt_temp.(n_x)=pt_temp.(n_x).(Names_i{ii});
        else
            temp=size(pt_temp.(n_x));
            if temp(1)>1 % n_d depends on fixed type
                pt_temp.(n_x)=pt_temp.(n_x)(ii,:);
            elseif temp(2)==N_i % If there is one row, but number of elements in n_d happens to coincide with number of permanent types, then just let user know
                sprintf('Possible Warning: Number of columns of %s is the same as the number of permanent types. \n This may just be coincidence as number of %s variables is equal to number of permanent types. \n If they are intended to be permanent types then %s should have them as different rows (not columns). \n',...
                    n_x, n_x(end), n_x)
            end
        end
    end

    for x_grid=["d_grid" "a_grid" "z_grid"]
        if isstruct(pt_temp.(x_grid))
            pt_temp.(x_grid)=pt_temp.(x_grid).(Names_i{ii});
        end
    end
    
    pi_z_temp=pi_z;
    % If using 'agedependentgrids' then pi_z will actually be the AgeDependentGridParamNames, which is a structure. 
    % Following gets complicated as pi_z being a structure could be because
    % it depends just on age, or on permanent type, or on both.
    if exist('vfoptions','var')
        if isfield(vfoptions,'agedependentgrids')
            if isa(vfoptions.agedependentgrids, 'struct')
                if isfield(vfoptions.agedependentgrids, Names_i{ii})
                    vfoptions_temp.agedependentgrids=vfoptions.agedependentgrids.(Names_i{ii});
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    pi_z_temp=pi_z.(Names_i{ii});
                else
                    % The current permanent type does not use age dependent grids.
                    vfoptions_temp=rmfield(vfoptions_temp,'agedependentgrids');
                    % Different grids by permanent type (some of them must be using agedependentgrids even though not the current permanent type), but not depending on age.
                    pi_z_temp=pi_z.(Names_i{ii});
                end
            else
                temp=size(vfoptions.agedependentgrids);
                if temp(1)>1 % So different permanent types use different settings for age dependent grids
                    if prod(temp(ii,:))>0
                        vfoptions_temp.agedependentgrids=vfoptions.agedependentgrids(ii,:);
                    else
                        vfoptions_temp=rmfield(vfoptions_temp,'agedependentgrids');
                    end
                    % In this case AgeDependentGridParamNames must be set up as, e.g., AgeDependentGridParamNames.ptype1.d_grid
                    pi_z_temp=pi_z.(Names_i{ii});
                else % Grids depend on age, but not on permanent type (at least the function does not, you could set it up so that this is handled by the same function but a parameter whose value differs by permanent type
                    pi_z_temp=pi_z;
                end
            end
        elseif isstruct(pi_z)
            pi_z_temp=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age.
        end
    elseif isstruct(pi_z)
        pi_z_temp=pi_z.(Names_i{ii}); % Different grids by permanent type, but not depending on age. (same as the case just above; this case can occour with or without the existence of vfoptions, as long as there is no vfoptions.agedependentgrids)
    end
    
    ReturnFn_temp=ReturnFn;
    if isstruct(ReturnFn)
        ReturnFn_temp=ReturnFn.(Names_i{ii});
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on permanent type). So go through each of
    % these in term.
    % ie. Parameters.alpha=[0;1]; or Parameters.alpha.ptype1=0; Parameters.alpha.ptype2=1;
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters); % all the different parameters
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isstruct(Parameters.(FullParamNames{kField})) % Check the current parameter for permanent type in structure form
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
    
    if vfoptions_temp.verbose==1
        sprintf('Parameter values for the current permanent type')
        Parameters_temp
    end
    
    % The parameter names can be made to depend on the permanent-type
    DiscountFactorParamNames_temp=DiscountFactorParamNames;
    if isstruct(DiscountFactorParamNames)
        DiscountFactorParamNames_temp=DiscountFactorParamNames.(Names_i{ii});
    end
    ReturnFnParamNames_temp=ReturnFnParamNames;
    if isstruct(ReturnFnParamNames)
        ReturnFnParamNames_temp=ReturnFnParamNames.(Names_i{ii});
    end
    if Case1orCase2==2
        PhiaprimeParamNames_temp=PhiaprimeParamNames;
        if isstruct(PhiaprimeParamNames)
            PhiaprimeParamNames_temp=PhiaprimeParamNames.(Names_i{ii});
        end
    end
    	
    if finitehorz==0  % Infinite horizon
        % Infinite Horizon requires an initial guess of value function. For
        % the present I simply don't let this feature be used when using
        % permanent types. WOULD BE GOOD TO CHANGE THIS IN FUTURE SOMEHOW.
%         V_ii=zeros(prod(pt_temp.n_a),prod(pt_temp.n_z)); % The initial guess (note that its value is 'irrelevant' in the sense that global uniform convergence is anyway known to occour for VFI).
        pt_ns = {pt_temp.n_d,pt_temp.n_a,pt_temp.n_z};
        pt_grids = {pt_temp.d_grid, pt_temp.a_grid, pt_temp.z_grid, pi_z_temp};
        pt_return_params = {ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp};
        if Case1orCase2==1
            if exist('vfoptions','var')
                [V_ii, Policy_ii]=ValueFnIter_Case1(pt_ns{:},pt_grids{:}, pt_return_params{:}, vfoptions_temp);
            else
                [V_ii, Policy_ii]=ValueFnIter_Case1(pt_ns{:},pt_grids{:}, pt_return_params{:});
            end
        elseif Case1orCase2==2
            if exist('vfoptions','var')
                [V_ii, Policy_ii]=ValueFnIter_Case2(V_ii,pt_ns{:},pt_grids{:}, Phi_aprime_temp, Case2_Type_temp, pt_return_params{:}, PhiaprimeParamNames_temp, vfoptions_temp);
            else
                [V_ii, Policy_ii]=ValueFnIter_Case2(V_ii,pt_ns{:},pt_grids{:}, Phi_aprime_temp, Case2_Type_temp, pt_return_params{:}, PhiaprimeParamNames_temp);
            end
        end
    elseif finitehorz==1 % Finite horizon
        % Check for some relevant vfoptions that may depend on permanent type
        % dynasty, agedependentgrids, lowmemory, (parallel??)
        if Case1orCase2==1
            if exist('vfoptions','var')
                [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(pt_ns{:},N_j_temp,pt_grids{:}, pt_return_params{:}, vfoptions_temp);
            else
                [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(pt_ns{:},N_j_temp,pt_grids{:}, pt_return_params{:}, ReturnFnParamNames_temp);
            end
        elseif Case1orCase2==2
            if exist('vfoptions','var')
                [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(pt_ns{:},N_j_temp,pt_grids{:}, Phi_aprime_temp, Case2_Type_temp, pt_return_params{:}, PhiaprimeParamNames_temp, vfoptions_temp);
            else
                [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(pt_ns{:},N_j_temp,pt_grids{:}, Phi_aprime_temp, Case2_Type_temp, pt_return_params{:}, PhiaprimeParamNames_temp);
            end
        end
    end
        
    V.(Names_i{ii})=V_ii;
    Policy.(Names_i{ii})=Policy_ii;    

end

end