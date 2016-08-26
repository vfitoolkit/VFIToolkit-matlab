function [V, Policy]=ValueFnIter_Case2_FHorz_PType(n_d,n_a,n_z,N_j, N_i,d_grid, a_grid, z_grid, pi_z, Phi_aprime, Case2_Type, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, PhiaprimeParamNames, vfoptions)

for ii=1:N_i
    if exist('vfoptions','var')
        if vfoptions.verbose==1
            sprintf('Fixed type: %i of %i',ii, N_i)
        end
    end
    
    % Go through everything which might be dependent on fixed type (PType)
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
        if isa(pi_z,'struct')
        names=fieldnames(pi_z);
        pi_z_temp=pi_z.(names{ii});
    end
    ReturnFn_temp=ReturnFn;
    if isa(ReturnFn,'struct')
        names=fieldnames(ReturnFn);
        ReturnFn_temp=ReturnFn.(names{ii});
    end
    Parameters_temp=Parameters;
    FullParamNames=fieldnames(Parameters);
    nFields=length(FullParamNames);
    for kField=1:nFields
        if isa(Parameters.(FullParamNames{kField}), 'struct')
            names=fieldnames(Parameters.(FullParamNames{kField}));
            Parameters_temp.(FullParamNames{kField})=Parameters.(FullParamNames{kField}).(names{ii});
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
    PhiaprimeParamNames_temp=PhiaprimeParamNames;
    if isa(PhiaprimeParamNames,'struct')
        names=fieldnames(PhiaprimeParamNames);
        PhiaprimeParamNames_temp=PhiaprimeParamNames.(names{ii});
    end
    Phi_aprime_temp=Phi_aprime;
    if isa(Phi_aprime,'struct')
        names=fieldnames(Phi_aprime);
        Phi_aprime_temp=Phi_aprime.(names{ii});
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