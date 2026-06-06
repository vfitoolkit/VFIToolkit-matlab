function [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz_e(VKron,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% The VKron input is next period value fn, the VKron output is this period.

N_d=prod(n_d);
N_a=prod(n_a);
N_e=prod(n_e);

if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
    % V slot = Valt (Naive) / Vunderbar (Sophisticated); Policy = QH choice; drop QH-only extras
    [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_QH_noz_e(VKron,n_d,n_a,n_e,N_j,d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    return % QH variant already un-Krons Policy
end

%% If get to here then not using exoticpreferences nor StateDependentVariables_z
% N_z==0 is handled by a different command
if vfoptions.divideandconquer==0
    if vfoptions.gridinterplayer==0
        if N_d==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        end
    else % vfoptions.gridinterplayer==1
        if isscalar(n_a)
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_GI2A_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else % vfoptions.divideandconquer==1
    if ~isscalar(n_a) && length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a(2)
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
        end
    end
    if vfoptions.gridinterplayer==0
        if isscalar(n_a)
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % vfoptions.gridinterplayer==1
        if isscalar(n_a)
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC1_GI1_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else % 2 endogenous states
            if N_d==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_nod_noz_e_raw(VKron,n_a,n_e, N_j, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_TPath_SingleStep_DC2A_GI2A_noz_e_raw(VKron,n_d,n_a,n_e, N_j, d_gridvals, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end

%% Policy in transition paths
% Treat the e-dim as the z-dim (no z) and use the _FHorz_z UnKron family
if isscalar(n_a)
    if N_d==0
        PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_e,N_j,vfoptions);
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_e,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,N_a,N_e,N_j,vfoptions);
        end
    end
else
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,N_a,N_e,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_a1,n_a2,N_a,N_e,N_j,vfoptions);
        end
    else
        if vfoptions.gridinterplayer==0
            PolicyKron=UnKronPolicyIndexes1_FHorz_z(PolicyKron,[n_d,n_a],N_a,N_e,N_j,vfoptions);
        else
            PolicyKron=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,N_a,N_e,N_j,vfoptions);
        end
    end
end

end
