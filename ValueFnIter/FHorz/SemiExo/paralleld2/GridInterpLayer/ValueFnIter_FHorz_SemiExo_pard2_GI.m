function [V,Policy]=ValueFnIter_FHorz_SemiExo_pard2_GI(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==0, vfoptions.gridinterplayer==1, vfoptions.pard2==1

error('vfoptions.pard2 with vfoptions.gridinterplayer=1 is not yet implemented')

end
