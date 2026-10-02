function [V,Policy]=ValueFnIter_FHorz_ExpAssetu_GI(n_d1,n_d2,n_a1,n_a2,n_z,n_u, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, u_gridvals, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);


%%

if N_a1==0
    error('Cannot use grid interpolation layer if there is no standard endogenous state')
end

%% GI2A path: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, grid interpolation layer), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
if length(n_a1)>1
    % a1_gridvals holds all the standard endogenous states (rows cycle the first fastest); n_a2/a2_grid hold the experience asset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_gridvals=a1_gridvals(1:n_a1(1):end,2:end);
    a1_grid=a1_gridvals(1:n_a1(1),1);
    n_a2=n_a1(2:end);
    n_a1=n_a1(1);

    % noz_e: dedicated _noz_e raw files (no z, i.i.d. e)
    if N_z==0 && N_e>0
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, vfoptions.n_e, n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J, u_gridvals, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_noz_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, vfoptions.n_e, n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J, u_gridvals, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
        if vfoptions.outputkron==1
            V=VKron;
            Policy=PolicyKron;
            return
        end
        n_a=[n_a1,n_a2,n_a3];
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, vfoptions.n_e, N_j, vfoptions);
        return
    end

    % z+e: both Markov z and i.i.d. e present — route to dedicated _e raw files
    if N_z>0 && N_e>0
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_gridvals, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_gridvals, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
        if vfoptions.outputkron==1
            V=VKron;
            Policy=PolicyKron;
            return
        end
        n_a=[n_a1,n_a2,n_a3];
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        return
    end

    if N_z==0
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_nod1_noz_raw(n_d2, n_a1, n_a2, n_a3, n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_noz_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
    else
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z, n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, u_gridvals, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, u_gridvals, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
    end

    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    n_a=[n_a1,n_a2,n_a3];
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_noz(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, N_j, vfoptions);
    end
    return
end

if N_e==0 % no e variable
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_nod1_noz_raw(n_d2,n_a1,n_a2,n_u,N_j, d2_gridvals, a1_gridvals, a2_grid,  u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_nod1_raw(n_d2,n_a1,n_a2,n_z,n_u,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,  u_gridvals, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_noz_raw(n_d1,n_d2,n_a1,n_a2,n_u,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid,  u_gridvals, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_u,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,  u_gridvals, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e,n_u,N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, u_gridvals, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e,n_u,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J,  u_gridvals, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else % d1 variable
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e,n_u,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J,  u_gridvals, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetu_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e,n_u,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J,  u_gridvals, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if n_d1>0
    n_d=[n_d1,n_d2];
else
    n_d=n_d2;
end
n_a=[n_a1,n_a2];

% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron, n_d, n_a1, n_a, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d, n_a1, n_a, n_z, N_j, vfoptions);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d, n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron, n_d, n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
    end
end



end


