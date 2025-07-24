function PolicyValues=PolicyInd2Val_Case1_FHorz_PType(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,vfoptions)
% Can use simoptions or vfoptions. If user is calling it, will probably be
% vfoptions. But internally it gets used with simoptions. The options that
% it checks are all things that will be common to both.

Names_i=fieldnames(PolicyIndexes);
N_i=length(Names_i);

PolicyValues=struct();
for ii=1:N_i
    % Go through everything which might be dependent on fixed type (PType)
    n_d_temp=n_d;
    if isa(n_d,'struct')
        names=fieldnames(n_d);
        n_d_temp=n_d.(names{ii});
    end
    n_a_temp=n_a;
    if isa(n_a,'struct')
        names=fieldnames(n_a);
        n_a_temp=n_a.(names{ii});
    end
    n_z_temp=n_z;
    if isa(n_z,'struct')
        names=fieldnames(n_z);
        n_z_temp=n_z.(names{ii});
    end
    N_j_temp=N_j;
    if isa(N_j,'struct')
        names=fieldnames(N_j);
        N_j_temp=N_j.(names{ii});
    end
    
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
    
    PolicyValues.(Names_i{ii})=PolicyInd2Val_Case1_FHorz(PolicyIndexes.(Names_i{ii}),n_d_temp,n_a_temp,n_z_temp,N_j_temp,d_grid_temp,a_grid_temp,vfoptions);
end


end