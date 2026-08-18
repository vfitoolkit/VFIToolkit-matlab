function [V,Policy]=ValueFnIter_FHorz_RiskyAsset_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==0, vfoptions.gridinterplayer==1 (Plain GI)
% d1: ReturnFn but not aprimeFn
% d2: aprimeFn but not ReturnFn
% d3: both ReturnFn and aprimeFn

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%%
if N_a1==0
    error('Cannot use grid interpolation layer with riskyasset if there is no standard endogenous state (N_a1==0)')
end
if ~isfield(vfoptions,'ngridinterp')
    vfoptions.ngridinterp=9;
end

% Two standard endogenous assets -> the GI2A raws.
if length(n_a1)>1
    if length(n_a1)>2
        error('riskyasset gridinterplayer supports at most two standard endogenous assets')
    end
    n_a1_1=n_a1(1); n_a1_2=n_a1(2);
    a1_1_grid=a1_grid(1:n_a1_1);
    a1_2_grid=a1_grid(n_a1_1+1:end);
    % a1->a1_1 (the one the grid interpolation layer refines), a2->a1_2 (folded), a3->the riskyasset
    if N_e==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_nod1_noz_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_nod1_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_noz_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_nod1_noz_e_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_nod1_e_raw(n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_noz_e_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1_1,n_a1_2,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_1_grid, a1_2_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    % Policy channels: nod1 is 4 (d2, d3, a1_1prime, a1_2prime), with d1 it is 5 (d1, d2, d3, a1_1prime, a1_2prime).
    % The grid interpolation layer appends the L2 and L2flag rows, which UnKronPolicyIndexes*
    % passes through unchanged when vfoptions.gridinterplayer==1.
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

%% Dispatch
if N_e==0 % no e variable
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else % d1 variable
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
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
