function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz(VKron,n_d1,n_d2,n_a1,n_a2,N_j,d_gridvals,d2_grid,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

N_d1=prod(n_d1);
N_a=prod([n_a1,n_a2]);

%%
if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_nod1_noz_raw(VKron,n_d1,n_a1,n_a2, N_j, d_gridvals,d2_grid,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_ExpAsset_noz_raw(VKron,n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals,d2_grid,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        error('Have not yet implemented grid interpolation layer for FHorz TPath ExpAsset without fastOLG=1. Ask on forum if you need this.')
    end
else
    error('Have not yet implemented divide and conquer for FHorz TPath ExpAsset without fastOLG=1. Ask on forum if you need this.')
end

%% Policy in transition paths
if N_d1==0
    PolicyKron=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron,[n_d2,n_a1],N_a,N_j,vfoptions);
else
    PolicyKron=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron,[n_d1,n_d2,n_a1],N_a,N_j,vfoptions);
end
% PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),N_a,N_j]);

end
