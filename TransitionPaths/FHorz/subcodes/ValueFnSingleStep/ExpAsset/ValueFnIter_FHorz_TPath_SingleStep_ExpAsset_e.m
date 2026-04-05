function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_e(VKron,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

N_d1=prod(n_d1);
% N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(n_e);

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d1==0
            error("not implemented")
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_nod_e_raw(VKron,n_a, n_z,n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_e_raw(VKron,n_d1,n_d2,n_a1,n_a2,n_z,n_e,N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        error('Have not yet implemented grid interpolation layer for FHorz TPath without fastOLG=1. Ask on forum if you need this.')
    end
else
    error('Have not yet implemented divide and conquer for FHorz TPath without fastOLG=1. Ask on forum if you need this.')
end

%% Policy in transition paths
PolicyKron=UnKronPolicyIndexes_Case1_FHorz_ExpAsset_e(PolicyKron,n_d1,n_d2,n_a1,n_a2,N_z,N_e,N_j,vfoptions);
PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),prod(n_a1)*prod(n_a2),N_z,N_e,N_j]);

end
