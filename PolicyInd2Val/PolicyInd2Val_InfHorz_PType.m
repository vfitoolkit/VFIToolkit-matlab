function PolicyValues=PolicyInd2Val_InfHorz_PType(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,vfoptions)
% Can use simoptions or vfoptions. If user is calling it, will probably be
% vfoptions. But internally it gets used with simoptions. The options that
% it checks are all things that will be common to both.

Names_i=fieldnames(PolicyIndexes);
N_i=length(Names_i);

PolicyValues=struct();
for ii=1:N_i
    iistr=Names_i{ii};
    vfoptions_temp=PType_Options(vfoptions,Names_i,ii);

    % Go through everything which might be dependent on fixed type (PType)
    [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);

    if isstruct(n_z)
        n_z_temp=n_z.(iistr);
    else
        n_z_temp=n_z;
    end

    PolicyValues.(iistr)=PolicyInd2Val_InfHorz(PolicyIndexes.(iistr),n_d_temp,n_a_temp,n_z_temp,d_grid_temp,a_grid_temp,vfoptions_temp);
end


end
