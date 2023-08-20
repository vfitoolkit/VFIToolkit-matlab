function PolicyValues=PolicyInd2Val_FHorz_Case3(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid)

% Case2 and Case3 have same form for Policy, so just redirect it there
% (this function is just to save user from having to know this)
PolicyValues=PolicyInd2Val_FHorz_Case2(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid);

end
