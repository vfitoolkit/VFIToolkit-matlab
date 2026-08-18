function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==1
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% Divide-and-conquer level1n setup (divide-and-conquer requires the standard endogenous state)
if N_a1==0
    error('Cannot use vfoptions.divideandconquer with riskyasset DC_GI if there is no standard endogenous state (N_a1==0)')
end
% Two standard endogenous assets -> the DC2A_GI2A raws.
if length(n_a1)>1
    if length(n_a1)>2
        error('riskyasset divideandconquer supports at most two standard endogenous assets')
    end
    n_a1_1=n_a1(1); n_a1_2=n_a1(2);
    a1_1_grid=a1_grid(1:n_a1_1);
    a1_2_grid=a1_grid(n_a1_1+1:end);
    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=floor(sqrt(n_a1_1));
    end
    vfoptions.level1n=min(vfoptions.level1n,n_a1_1); % level1n is scalar, and with two standard assets it is a1_1 that is divide-conquered
    % a1->a1_1 (divide-conquered), a2->a1_2 (folded), a3->the riskyasset
    if N_e==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_nod1_noz_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_nod1_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_noz_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_nod1_noz_e_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_nod1_e_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_noz_e_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC2A_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    % Policy channels: d2, d3, a1_1prime (divide-conquered), a1_2prime (folded) [plus d1 at the
    % front when there is a d1], and then the two grid-interp-layer rows (L2 and L2flag) which
    % UnKronPolicyIndexes*_FHorz_* passes through unchanged when vfoptions.gridinterplayer==1.
    n_a=[n_a1,n_a2];
    if N_d1==0
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_noz(PolicyKron, n_d2, n_d3, n_a1_1, n_a1_2, n_a, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d2, n_d3, n_a1_1, n_a1_2, n_a, n_z, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d2, n_d3, n_a1_1, n_a1_2, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            else
                V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d2, n_d3, n_a1_1, n_a1_2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    else % N_d1
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_noz(PolicyKron, n_d1, n_d2, n_d3, n_a1_1, n_a1_2, n_a, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_a1_1, n_a1_2, n_a, n_z, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_a1_1, n_a1_2, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            else
                V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_a1_1, n_a1_2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
    return
end
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
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else % d1 variable
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_DC1_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

n_a=[n_a1,n_a2];

% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% N_a1
if N_d1==0
    if N_e==0
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_noz(PolicyKron, n_d2,n_d3,n_a1, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d2,n_d3,n_a1, n_a, n_z, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d2,n_d3,n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, n_d2,n_d3,n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end
else % N_d1
    if N_e==0
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes4_FHorz_noz(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, n_z, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end
end


end
