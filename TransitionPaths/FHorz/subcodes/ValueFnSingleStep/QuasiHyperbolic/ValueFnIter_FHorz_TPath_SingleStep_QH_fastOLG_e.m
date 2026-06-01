function varargout=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG_e(VKron,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-Hyperbolic single-step VFI for one transition-path period (fastOLG, z, e).
% VKron in/out is Valt (Naive) or Vunderbar (Sophisticated).
%   Naive:         varargout = {VKron, PolicyKron, PolicyaltKron}
%   Sophisticated: varargout = {VKron, PolicyKron}

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

if ~isscalar(n_a)
    error('QuasiHyperbolic on FHorz TPath fastOLG: 2-endogenous-state DC2A/GI2A not yet supported (Phase 3 scope: 1 endogenous state only)')
end

if vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==0
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==0
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_GI1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_GI1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % DC1 + GI1
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_GI1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_GI1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_GI1_nod_e_raw(VKron,n_a, n_z, n_e, N_j, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_GI1_e_raw(VKron,n_d,n_a,n_z, n_e, N_j, d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end

%% Policy in transition paths
if N_d==0
    PolicyKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_a,N_a,N_j,N_z,N_e,vfoptions);
    if isNaive
        PolicyaltKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,n_a,N_a,N_j,N_z,N_e,vfoptions);
    end
else
    if vfoptions.gridinterplayer==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,[n_d,n_a],N_a,N_j,N_z,N_e,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,[n_d,n_a],N_a,N_j,N_z,N_e,vfoptions);
        end
    else
        PolicyKron=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d,n_a,N_a,N_j,N_z,N_e,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes2_FHorz_z_e(PolicyaltKron,n_d,n_a,N_a,N_j,N_z,N_e,vfoptions);
        end
    end
end

if isNaive
    varargout={VKron,PolicyKron,PolicyaltKron};
else
    varargout={VKron,PolicyKron};
end

end
