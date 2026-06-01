function varargout=ValueFnIter_FHorz_TPath_SingleStep_QH(VKron,n_d,n_a,n_z,N_j,d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-Hyperbolic single-step VFI for one transition-path period (slowOLG, z, no-e).
% VKron in/out is Valt (Naive) or Vunderbar (Sophisticated).
%   Naive:         varargout = {VKron, PolicyKron, PolicyaltKron}
%   Sophisticated: varargout = {VKron, PolicyKron}

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

if (vfoptions.divideandconquer==1 || vfoptions.gridinterplayer==1) && ~isscalar(n_a)
    error('QuasiHyperbolic on FHorz TPath slowOLG: DC/GI with 2 endogenous states (DC2A/GI2A) not yet supported')
end

if vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==0
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==0
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
else % DC1 + GI1
    if isNaive
        if N_d==0
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,PolicyaltKron,VtildeKron]=ValueFnIter_FHorz_TPath_SingleStep_QHN_DC1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else
        if N_d==0
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_nod_raw(VKron,n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron,PolicyKron,VhatKron]=ValueFnIter_FHorz_TPath_SingleStep_QHS_DC1_GI1_raw(VKron,n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    end
end

%% Policy in transition paths
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_z,N_j,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_a,N_a,N_z,N_j,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_z,N_j,vfoptions);
            if isNaive
                PolicyaltKron=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,[n_d,n_a],N_a,N_z,N_j,vfoptions);
            end
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,N_a,N_z,N_j,vfoptions);
            if isNaive
                PolicyaltKron=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron,n_d,n_a,N_a,N_z,N_j,vfoptions);
            end
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_z,N_j,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_a,N_a,N_z,N_j,vfoptions);
        end
    else
        PolicyKron=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,N_a,N_z,N_j,vfoptions);
        if isNaive
            PolicyaltKron=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron,n_d,n_a1,n_a2,N_a,N_z,N_j,vfoptions);
        end
    end
end

if isNaive
    varargout={VKron,PolicyKron,PolicyaltKron,VtildeKron};
else
    varargout={VKron,PolicyKron,VhatKron};
end

end
