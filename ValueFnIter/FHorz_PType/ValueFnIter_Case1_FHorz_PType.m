function [V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z, N_j,N_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
%[V, PolicyIndexes]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,n_i,N_j, pi_z, beta_j, FmatrixFn_ij)

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
% N_i=prod(n_i);

for ii=1:N_i
    if exist('vfoptions','var')==1
        if isfield(vfoptions,'verbose')==1
            if vfoptions.verbose==1
                sprintf('Permanent type: %i of %i',ii, N_i)
            end
        end
    end
    
    % Go through everything which might be dependent on fixed type (PType)
    % [THIS could be better coded, 'names' are same for all these and just need to be found once outside of ii loop]
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
        if isa(pi_z,'struct')
        names=fieldnames(pi_z);
        pi_z_temp=pi_z.(names{ii});
    end
    ReturnFn_temp=ReturnFn;
    if isa(ReturnFn,'struct')
        names=fieldnames(ReturnFn);
        ReturnFn_temp=ReturnFn.(names{ii});
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
        
    
    if exist('vfoptions','var')==1 % check whether vfoptions was inputted
        [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, vfoptions);
    else
        [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp);
    end
    
    ptstring=['pt',num2str(ii)];
    
    V.(ptstring)=gather(V_ii); % GPU memory is limited, so switch solutions to the cpu
    Policy.(ptstring)=gather(Policy_ii);  % GPU memory is limited, so switch solutions to the cpu
    clear V_ii Policy_ii

end


end