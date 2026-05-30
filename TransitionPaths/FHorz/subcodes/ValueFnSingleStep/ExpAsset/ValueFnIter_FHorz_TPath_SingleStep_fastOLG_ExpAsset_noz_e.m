function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz_e(VKron,n_d1,n_d2,n_a1,n_a2,n_e,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as (a,j)-by-e [this form makes the expectations easier]
% Policy is done as a-by-j-by-e [this form is easier later, and easier for handling DC1]
% (fastOLG requires swapping order of j and z)

N_d1=prod(n_d1);
N_a=prod(n_a);
N_e=prod(n_e);
% N_z=0 is handled elsewhere
% N_e=0 is handled elsewhere

if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_nod1_noz_e_raw(VKron,n_d2,n_a1,n_a2, n_e, N_j, d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_noz_e_raw(VKron,n_d1,n_d2,n_a1,n_a2, n_e, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            error('None of these are implemented yet')
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_GI1_nod1_noz_e_raw(VKron,n_d2,n_a1,n_a2, n_e, N_j, d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_GI1_noz_e_raw(VKron,n_d1,n_d2,n_a1,n_a2, n_e, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % vfoptions.divideandconquer==1
        if vfoptions.gridinterplayer==0
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_nod1_noz_e_raw(VKron,n_d2,n_a1,n_a2, n_e, N_j, d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_noz_e_raw(VKron,n_d1,n_d2,n_a1,n_a2, n_e, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_GI1_nod1_noz_e_raw(VKron,n_d2,n_a1,n_a2, n_e, N_j, d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAsset_DC1_GI1_noz_e_raw(VKron,n_d1,n_d2,n_a1,n_a2, n_e, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, e_gridvals_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else
    error('Not yet implemented exoticpreferences for transtion paths (email me :)')
end

%% Policy in transition paths
% Treat the e-dim as the z-dim (no z) and use the _FHorz_z UnKron family
if vfoptions.gridinterplayer==0
    if N_d1==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d2,n_a1],N_a,N_e,N_j,vfoptions);
    else
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d1,n_d2,n_a1],N_a,N_e,N_j,vfoptions);
    end
else % vfoptions.gridinterplayer==1
    if N_d1==0
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a1,N_a,N_e,N_j,vfoptions);
    else
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,[n_d1,n_d2],n_a1,N_a,N_e,N_j,vfoptions);
    end
end


end
