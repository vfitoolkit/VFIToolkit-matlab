function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz(VKron,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

N_d=prod(n_d);
N_a=prod(n_a);

%% If get to here then not using exoticpreferences
if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_noz_raw(VKron,n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz_raw(VKron,n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        error('Have not yet implemented grid interpolation layer for FHorz TPath without fastOLG=1. Ask on forum if you need this.')
    end
else
    error('Have not yet implemented divide and conquer for FHorz TPath without fastOLG=1. Ask on forum if you need this.')
end

%% Policy in transition paths
PolicyKron=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron,n_d,n_a,N_j,vfoptions);
PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),N_a,N_j]);

end
