function [V, Policy]=ValueFnIter_Case2_FHorz_PType(n_d,n_a,n_z,N_j,Names_i,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)
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

for ii=1:N_i
        
    % First set up vfoptions
    if exist('vfoptions','var')
        vfoptions_temp=PType_Options(vfoptions,Names_i,ii);
        if ~isfield(vfoptions_temp,'verbose')
            vfoptions_temp.verbose=0;
        end
        if ~isfield(vfoptions_temp,'verboseparams')
            vfoptions_temp.verboseparams=0;
        end
        if ~isfield(vfoptions_temp,'ptypestorecpu')
            vfoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
        end
    else
        vfoptions_temp.verbose=0;
        vfoptions_temp.verboseparams=0;
        vfoptions_temp.ptypestorecpu=0; % GPU memory is limited, so switch solutions to the cpu. Off by default.
    end 
    
    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
    end
    
    % Go through everything which might be dependent on fixed type (PType)
    % [THIS could be better coded, 'names' are same for all these and just need to be found once outside of ii loop]
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
    if isa(pi_z,'struct')
        pi_z_temp=pi_z.(Names_i{ii});
    else
        pi_z_temp=pi_z;
    end
    if isa(ReturnFn,'struct')
        ReturnFn_temp=ReturnFn.(Names_i{ii});
    else
        ReturnFn_temp=ReturnFn;
    end
    if isa(ReturnFn,'struct')
        Phi_aprime_temp=Phi_aprime.(Names_i{ii});
    else
        Phi_aprime_temp=Phi_aprime;
    end
    if isa(Case2_Type,'struct')
        Case2_Type_temp=Case2_Type.(Names_i{ii});
    else
        Case2_Type_temp=Case2_Type;
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
    DiscountFactorParamNames_temp=DiscountFactorParamNames;
    if isa(DiscountFactorParamNames,'struct')
        names=fieldnames(DiscountFactorParamNames);
        for jj=1:length(names)
            if strcmp(names{jj},Names_i{ii})
                DiscountFactorParamNames_temp=DiscountFactorParamNames.(names{jj});
            end
        end
    end
    
    [V_ii, Policy_ii]=ValueFnIter_Case2_FHorz(n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, Phi_aprime_temp, Case2_Type_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, [], [], vfoptions_temp);
    
    if vfoptions_temp.ptypestorecpu==1
        V.(Names_i{ii})=gather(V_ii);
        Policy.(Names_i{ii})=gather(Policy_ii);
    else
        V.(Names_i{ii})=V_ii;
        Policy.(Names_i{ii})=Policy_ii;
    end 

end

end