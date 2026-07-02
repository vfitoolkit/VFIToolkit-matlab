function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz(VKron,n_d1,n_d2,n_a1,n_a2,n_z,N_j,d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.

% V is done as (a,j)-by-z [this form makes the expectations easier]
% Policy is done as a-by-j-by-z [this form is easier later, and easier for handling DC1]
% (fastOLG requires swapping order of j and z)

N_d1=prod(n_d1);
N_a=prod([n_a1,n_a2]);
N_z=prod(n_z);
% N_z=0 is handled elsewhere
% N_e is handled elsewhere


if strcmp(vfoptions.exoticpreferences,'None')
    if vfoptions.divideandconquer==0
        if vfoptions.gridinterplayer==0
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_nod1_raw(VKron,n_d2,n_a1,n_a2, n_z, N_j, d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_raw(VKron,n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_GI1_nod1_raw(VKron,n_d2,n_a1,n_a2, n_z, N_j, d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_GI1_raw(VKron,n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % vfoptions.divideandconquer==1
        if vfoptions.gridinterplayer==0
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_DC1_nod1_raw(VKron,n_d2,n_a1,n_a2, n_z, N_j, d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_DC1_raw(VKron,n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % vfoptions.gridinterplayer==1
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_DC1_GI1_nod1_raw(VKron,n_d2,n_a1,n_a2, n_z, N_j, d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_ExpAssetz_DC1_GI1_raw(VKron,n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals,d2_gridvals,a1_gridvals,a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end

    end
else
    error('Not yet implemented exoticpreferences for transtion paths (email me :)')
end


%% Policy in transition paths
if vfoptions.gridinterplayer==0
    if N_d1==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d2,n_a1],N_a,N_j,N_z,vfoptions); % fastOLG: pass N_j,N_z (not N_z,N_j) so UnKron reshape is (a,j,z)
    else
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d1,n_d2,n_a1],N_a,N_j,N_z,vfoptions); % fastOLG: pass N_j,N_z (not N_z,N_j) so UnKron reshape is (a,j,z)
    end
else % vfoptions.gridinterplayer==1
    if N_d1==0
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a1,N_a,N_j,N_z,vfoptions); % fastOLG: pass N_j,N_z (not N_z,N_j) so UnKron reshape is (a,j,z)
    else
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,[n_d1,n_d2],n_a1,N_a,N_j,N_z,vfoptions); % fastOLG: pass N_j,N_z (not N_z,N_j) so UnKron reshape is (a,j,z)
    end
end


end
