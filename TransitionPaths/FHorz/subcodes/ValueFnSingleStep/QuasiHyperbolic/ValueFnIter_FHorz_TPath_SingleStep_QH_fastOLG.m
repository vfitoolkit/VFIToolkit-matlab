function varargout=ValueFnIter_FHorz_TPath_SingleStep_QH_fastOLG(VKron,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-Hyperbolic single-step VFI for one transition-path period (fastOLG, z, no-e).
% VKron in/out is Valt (Naive) or Vunderbar (Sophisticated).
%   Naive:         varargout = {VKron, PolicyKron, PolicyaltKron}
%   Sophisticated: varargout = {VKron, PolicyKron}

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

% Phase 3 scope: fastOLG QH supports plain + DC1 + GI1 + DC1_GI1 (isscalar(n_a) only)
if ~isscalar(n_a)
    error('QuasiHyperbolic on FHorz TPath fastOLG: 2-endogenous-state DC2A/GI2A not yet supported (Phase 3 scope: 1 endogenous state only)')
end

if vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==0
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==0
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % DC1 + GI1
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_fastOLG_DC1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_fastOLG_DC1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end

%% Policy in transition paths
if N_d==0
    PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_j,N_z,vfoptions);
    if isNaive
        PolicyaltKron=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_a,N_a,N_j,N_z,vfoptions);
    end
else
    if vfoptions.gridinterplayer==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_j,N_z,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,[n_d,n_a],N_a,N_j,N_z,vfoptions);
        end
    else
        PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,N_a,N_j,N_z,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron,n_d,n_a,N_a,N_j,N_z,vfoptions);
        end
    end
end

if isNaive
    varargout={VKron,PolicyKron,PolicyaltKron,VtildeKron};
else
    varargout={VKron,PolicyKron,VhatKron};
end

end
