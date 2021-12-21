function [V, Policy]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z, N_j,Names_i,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% N_d=prod(n_d);
% N_a=prod(n_a);
% N_z=prod(n_z);
% N_i=prod(n_i);

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
    else
        vfoptions_temp.verbose=0;
    end 
    
    if vfoptions_temp.verbose==1
        fprintf('Permanent type: %i of %i \n',ii, N_i)
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
    
    [V_ii, Policy_ii]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j,d_grid_temp, a_grid_temp, z_grid_temp, pi_z_temp, ReturnFn_temp, Parameters_temp, DiscountFactorParamNames_temp, ReturnFnParamNames_temp, vfoptions);
        
    V.(Names_i{ii})=gather(V_ii); % GPU memory is limited, so switch solutions to the cpu
    Policy.(Names_i{ii})=gather(Policy_ii);  % GPU memory is limited, so switch solutions to the cpu
    clear V_ii Policy_ii

end


end