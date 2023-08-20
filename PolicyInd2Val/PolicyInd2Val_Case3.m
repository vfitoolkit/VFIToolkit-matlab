function PolicyValues=PolicyInd2Val_Case3(PolicyIndexes,n_d,n_a,n_z,d_grid)
% Case2 and Case3 have same form for Policy, so just redirect it there
% (this function is just to save user from having to know this)

PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid);

end