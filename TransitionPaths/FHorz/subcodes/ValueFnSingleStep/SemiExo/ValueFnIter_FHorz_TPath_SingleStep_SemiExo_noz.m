function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_noz(VKron,n_d1,n_d2,n_a,n_semiz,N_j,d1_gridvals,d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.
% VKron is in kron form: (N_a,N_bothz,N_j), where bothz=(semiz) with semiz indexing fastest
% semiz_gridvals_J is (N_semiz,l_semiz,N_j) [standard form]
% pi_semiz_J is (semiz,semiz',d2,j) [standard form, transition probabilities depend on d2]

N_d1=prod(n_d1);
N_a=prod(n_a);
N_bothz=prod(n_semiz);

if ~isscalar(n_a)
    error('Transition paths with semi-exogenous states only allow a single endogenous state (cannot have length(n_a)>1)')
end

if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_nod1_noz_raw(VKron,n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_noz_raw(VKron,n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_GI1_nod1_noz_raw(VKron,n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_GI1_noz_raw(VKron,n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % vfoptions.divideandconquer==1
    if vfoptions.gridinterplayer==0
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_DC1_nod1_noz_raw(VKron,n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_DC1_noz_raw(VKron,n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if N_d1==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_DC1_GI1_nod1_noz_raw(VKron,n_d2,n_a,n_semiz,N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_SemiExo_DC1_GI1_noz_raw(VKron,n_d1,n_d2,n_a,n_semiz,N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end

%% Policy in transition paths
% slowOLG semiz raws already return Policy with d1/d2/aprime in separate rows (plus aprimeL2index,L2flag rows with grid interpolation)
% The UnKron just unpacks any multi-dimensional d1 (and would unpack multi-dimensional aprime, but semiz only allows scalar n_a)
if N_d1==0
    PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a,N_a,N_bothz,N_j,vfoptions);
else
    PolicyKron=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d1,n_d2,n_a,N_a,N_bothz,N_j,vfoptions);
end

end
