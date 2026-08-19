function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==0
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn
% d4: ReturnFn but not aprimeFn, and determines semiz transitions

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% Divide-and-conquer level1n setup (divide-and-conquer requires the standard endogenous state)
if N_a1==0
    error('Cannot use vfoptions.divideandconquer with riskyasset+semiz if there is no standard endogenous state (N_a1==0)')
end
%% DC2A path: two standard endogenous states -- divide-conquer the first, fold the second; n_a2 is the riskyasset
if length(n_a1)>1
    if length(n_a1)>2
        error('riskyasset+semiz divideandconquer supports at most two standard endogenous assets')
    end
    n_a1_1=n_a1(1); n_a1_2=n_a1(2);
    a1_1_grid=a1_grid(1:n_a1_1);
    a1_2_grid=a1_grid(n_a1_1+1:end);
    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=floor(sqrt(n_a1_1));
    end
    if length(vfoptions.level1n)>1 % level1n must reach the raws as a scalar
        if vfoptions.level1n(2)>=n_a1_2 % only divide-and-conquer on the first standard endogenous state
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With riskyasset+semiz DC2A, can only do divide-and-conquer on the first standard endogenous state')
        end
    end
    vfoptions.level1n=min(vfoptions.level1n,n_a1_1); % level1n is scalar, and with two standard assets it is a1_1 that is divide-conquered
    % a1->a1_1 (divide-conquered), a2->a1_2 (folded), a3->the riskyasset
    if N_e==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_nod1_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_e
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_nod1_e_raw(n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC2A_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1_1,n_a1_2,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_1_grid, a1_2_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    % Policy channels: nod1 is 5 (d2, d3, d4, a1_1prime, a1_2prime), with d1 it is 6 (d1, d2, d3, d4, a1_1prime, a1_2prime).
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

%% Single DC dim -- existing DC1 path
if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a1(1)));
    if n_a1(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a1(1)); % n_a1(1): level1n is scalar, and with two standard assets it is a1_1 that is divide-conquered

%% Dispatch
if N_e==0 % no e variable
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_nod1_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_nod1_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else % d1 variable
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end

%%
if vfoptions.outputkron==0
    n_a=[n_a1,n_a2];
    % Policy from the raw has the choices on the first dimension: (d2,d3,d4,a1prime), or (d1,d2,d3,d4,a1prime) when there is a d1
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
