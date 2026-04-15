function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz(VKron,n_d1,n_d2,n_a1,n_a2,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as a-by-j
% Policy is done as a-by-j
% (fastOLG is easy without z)

N_d1=prod(n_d1);
N_a=prod(n_a);
% z and e are handled elsewhere

if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_nod1_noz_raw(VKron,n_d2,n_a1,n_a2, N_j, d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz_raw(VKron,n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            error('None of these are implemented yet')
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_GI_nod1_noz_raw(VKron,n_d2,n_a1,n_a2, N_j, d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames,aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_GI_noz_raw(VKron,n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if vfoptions.gridinterplayer==0
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_nod1_noz_raw(VKron,n_d2,n_a1,n_a2, N_j, d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_noz_raw(VKron,n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            error('None of these are implemented yet')
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_GI_nod1_noz_raw(VKron,n_d2,n_a1,n_a2, N_j, d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_GI_noz_raw(VKron,n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end

%% Policy in transition paths
PolicyKron=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron,n_d,n_a,N_j,vfoptions);
PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),N_a,N_j]);

end
