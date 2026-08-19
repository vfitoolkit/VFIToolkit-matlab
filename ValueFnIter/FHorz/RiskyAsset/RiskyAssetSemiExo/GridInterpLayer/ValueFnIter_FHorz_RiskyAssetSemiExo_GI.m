function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==0, vfoptions.gridinterplayer==1 (Plain GI) with semiz
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%%
if N_a1==0
    error('Cannot use grid interpolation layer with riskyasset+semiz if there is no standard endogenous state (N_a1==0)')
end
if ~isfield(vfoptions,'ngridinterp')
    vfoptions.ngridinterp=9;
end

% Two standard endogenous assets -> the GI2A raws.
if length(n_a1)>1
    if length(n_a1)>2
        error('riskyasset+semiz gridinterplayer supports at most two standard endogenous assets')
    end
    n_a1_1=n_a1(1); n_a1_2=n_a1(2);
    a1_1_grid=a1_grid(1:n_a1_1);
    a1_2_grid=a1_grid(n_a1_1+1:end);
    % a1->a1_1 (the one the grid interpolation layer refines), a2->a1_2 (folded), a3->the riskyasset
    if N_e==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_nod1_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_e
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_nod1_e_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI2A_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    % Policy channels: nod1 is 5 (d2, d3, d4, a1_1prime, a1_2prime), with d1 it is 6 (d1, d2, d3, d4, a1_1prime, a1_2prime).
    % The grid interpolation layer appends the L2 and L2flag rows, which UnKronPolicyIndexes*
    % passes through unchanged when vfoptions.gridinterplayer==1.
    n_a=[n_a1,n_a2];
    if N_e==0
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, n_semiz, N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes6_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, n_semiz, N_j, vfoptions);
            end
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, [n_semiz,n_z], N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes6_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, [n_semiz,n_z], N_j, vfoptions);
            end
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes6_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            end
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes6_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1_1, n_a1_2, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
    return
end

%% Dispatch
if N_e==0 % no e variable
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_nod1_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_nod1_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else % d1 variable
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI1_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end

%%
if vfoptions.outputkron==0
    n_a=[n_a1,n_a2];
    % Policy from the raw has the choices on the first dimension: (d2,d3,d4,a1prime,L2,L2flag), or (d1,d2,d3,d4,a1prime,L2,L2flag) when there is a d1.
    % vfoptions.gridinterplayer stays 1, so UnKron unpacks rows 1-4/1-5 and passes the L2 and L2flag rows through.
    if N_e==0
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, N_j, vfoptions);
            end
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], N_j, vfoptions);
            end
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            end
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
            if N_d1==0
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            else
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end
