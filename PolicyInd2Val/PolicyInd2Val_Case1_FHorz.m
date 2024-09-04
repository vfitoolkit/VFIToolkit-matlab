function PolicyValues=PolicyInd2Val_Case1_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid)

outputkron=0; % outputkron=1 is just for internal use

% Just redirects
PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,outputkron);

end
