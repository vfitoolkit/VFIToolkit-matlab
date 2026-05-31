function [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC_GI(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==1
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
    error('Cannot use vfoptions.divideandconquer with riskyasset+semiz DC_GI if there is no standard endogenous state (N_a1==0)')
end
if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=round(sqrt(n_a1(1)));
    if n_a1(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a1);
if ~isfield(vfoptions,'ngridinterp')
    vfoptions.ngridinterp=9;
end

%% Dispatch
if N_e==0 % no e variable
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_nod1_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_nod1_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else % d1 variable
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC1_GI1_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end

%%
if vfoptions.outputkron==0
    if N_d1>0
        n_d=[n_d1,n_d2,n_d3,n_d4];
    else
        n_d=[n_d2,n_d3,n_d4];
    end
    n_a=[n_a1,n_a2];
    n_dForUnKron=[n_d,n_a1,vfoptions.ngridinterp]; % a1prime added as final channel, then GI L2 index
    vfoptions.gridinterplayer=0; % L2 already folded into n_dForUnKron as fake-d; suppress NEW family's L2-passthrough handling
    % Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if N_e==0
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_dForUnKron, n_a, n_semiz, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_dForUnKron, n_a, [n_semiz,n_z], N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_dForUnKron, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_dForUnKron, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end
