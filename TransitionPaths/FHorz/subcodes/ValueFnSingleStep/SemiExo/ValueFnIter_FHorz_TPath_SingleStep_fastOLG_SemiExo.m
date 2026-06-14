function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo(VKron,n_d1,n_d2,n_a,n_z,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% 'fastOLG' just means parallelizing across all of the "ages" (j) at once.
% V is done as (a,j)-by-bothz, Policy as a-by-j-by-bothz, where bothz=(semiz,z) with semiz indexing fastest
% pi_z_J is (j,z',z) and z_gridvals_J is (j,N_z,l_z) [fastOLG forms]
% semiz_gridvals_J is (j,N_semiz,l_semiz) [fastOLG form]
% vfoptions.EVpre is needed (=0 for standard transition paths)
% pi_semiz_J is (semiz,semiz',d2,j) [standard form, transition probabilities depend on d2]

N_d1=prod(n_d1);
N_a=prod(n_a);
N_bothz=prod(n_semiz)*prod(n_z);

if ~isscalar(n_a)
    error('Transition paths with semi-exogenous states only allow a single endogenous state (cannot have length(n_a)>1)')
end

if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_GI1_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_GI1_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % vfoptions.divideandconquer==1
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_DC1_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_DC1_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_DC1_GI1_nod1_raw(VKron,n_d2,n_a,n_z,n_semiz,N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_SemiExo_DC1_GI1_raw(VKron,n_d1,n_d2,n_a,n_z,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end

%% Policy in transition paths
% fastOLG raws return Policy as the joint (d1,d2,aprime) index (plain/DC1), or as 4 rows (joint-d,aprimeLower,L2index,L2flag) with grid interpolation
% Note: The actual ordering of N_bothz,N_j is not relevant to how the UnKron commands work, so can just mix them up.
if vfoptions.gridinterplayer==0
    if N_d1==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d2,n_a],N_a,N_j,N_bothz,vfoptions);
    else
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d1,n_d2,n_a],N_a,N_j,N_bothz,vfoptions);
    end
else % vfoptions.gridinterplayer==1
    if N_d1==0
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a,N_a,N_j,N_bothz,vfoptions);
    else
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,[n_d1,n_d2],n_a,N_a,N_j,N_bothz,vfoptions);
    end
end

end
