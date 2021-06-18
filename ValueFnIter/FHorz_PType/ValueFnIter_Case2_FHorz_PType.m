function [V, Policy]=ValueFnIter_Case2_FHorz_PType(n_d,n_a,n_z,N_j,N_i,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)
%
% Allows for different permanent (fixed) types of agent. The only 'extra'
% input required (compared to ValueFnIter_Case2_FHorz, is N_i. But many
% other inputs may be in different forms wherever they depend on the
% permanent-type. 
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

for ii=1:N_i
    if exist('vfoptions','var')
        if vfoptions.verbose==1
            sprintf('Fixed type: %i of %i',ii, N_i)
        end
    end
    
    % Go through everything which might be dependent on fixed type (PType)
    % Notice that the way this is coded the grids (etc.) could be either
    % fixed, or a function (that depends on age, and possibly on fixed
    % type), or they could be a structure. Only in the case where they are
    % a structure is there a need to take just a specific part and send
    % only that to the 'non-PType' version of the command.
    % [THIS could be better coded]
    d_grid_temp=d_grid;
    if isa(d_grid,'struct')
        names=fieldnames(d_grid);
        d_grid_temp=d_grid.(names{ii});
    end
    a_grid_temp=a_grid;
    if isa(a_grid,'struct')
        names=fieldnames(a_grid);
        a_grid_temp=a_grid.(names{ii});        
    end
    z_grid_temp=z_grid;
    if isa(z_grid,'struct')
        names=fieldnames(z_grid);
        z_grid_temp=z_grid.(names{ii});
    end
    pi_z_temp=pi_z;
    % If using 'agedependentgrids' then pi_z will actually be the  AgeDependentGridParamNames, which is a structure. 
    % But in this case it is necessary to pass the entire structure.    
    if exist('vfoptions','var')
        if isfield(vfoptions,'agedependentgrids')
            % Do nothing.
        elseif isa(pi_z,'struct')
            names=fieldnames(pi_z);
            pi_z_temp=pi_z.(names{ii});
        end
    else isa(pi_z,'struct')
        names=fieldnames(pi_z);
        pi_z_temp=pi_z.(names{ii});
    end
    ReturnFn_temp=ReturnFn;
    if isa(ReturnFn,'struct')
        names=fieldnames(ReturnFn);
        ReturnFn_temp=ReturnFn.(names{ii});
    end
    Phi_aprime_temp=Phi_aprime;
    if isa(Phi_aprime,'struct')
        names=fieldnames(Phi_aprime);
        Phi_aprime_temp=Phi_aprime.(names{ii});
    end
    
    % Parameters are allowed to be given as structure, or as vector/matrix
    % (in terms of their dependence on fixed type). So go through each of
    % these in term.
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct') % Check for permanent type in structure form
            names=fieldnames(Parameters.(FullParamNames{kField}));
            Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{ii});
        elseif sum(size(Parameters.(FullParamNames{kField}))==N_i)==1 % Check for permanent type in vector/matrix form.
            temp=Parameters.(FullParamNames{kField});
            [~,ptypedim]=max(size(Parameters.(FullParamNames{kField}))==N_i); % Parameters as vector/matrix can be at most two dimensional, figure out which relates to PType.
            if ptypedim==1
                Parameters_temp.(FullParamNames{kField})=temp(ii,:);
            elseif ptypedim==2
                Parameters_temp.(FullParamNames{kField})=temp(:,ii);
            end
        end
    end
    
    % The parameter names can be made to depend on the permanent-type
    DiscountFactorParamNames_temp=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        names=fieldnames(DiscountFactorParamNames);
        DiscountFactorParamNames_temp=DiscountFactorParamNames.(names{ii});
    end
    ReturnFnParamNames_temp=ReturnFnParamNames;
    if isa(ReturnFnParamNames,'struct')
        names=fieldnames(ReturnFnParamNames);
        ReturnFnParamNames_temp=ReturnFnParamNames.(names{ii});
    end
    PhiaprimeParamNames_temp=PhiaprimeParamNames;
    if isa(PhiaprimeParamNames,'struct')
        names=fieldnames(PhiaprimeParamNames);
        PhiaprimeParamNames_temp=PhiaprimeParamNames.(names{ii});
    end
    
    if exist('vfoptions','var')
        [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, PhiaprimeParamNames_temp, vfoptions);
    else
        [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(n_d,n_a,n_z,N_j,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, PhiaprimeParamNames_temp);
    end
    
    ftstring=['ft',num2str(ii)];
    
    V.(ftstring)=V_ii;
    Policy.(ftstring)=Policy_ii;    

end

end