function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz_e(VKron,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        error('Have not yet implemented grid interpolation layer for FHorz TPath without fastOLG=1. Ask on forum if you need this.')
    end
else
    error('Have not yet implemented divide and conquer for FHorz TPath without fastOLG=1. Ask on forum if you need this.')
end

%% Policy in transition paths
PolicyKron=UnKronPolicyIndexes_Case1_FHorz(PolicyKron,n_d,n_a,N_e,N_j,vfoptions);
PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),N_a,N_e,N_j]);

end
