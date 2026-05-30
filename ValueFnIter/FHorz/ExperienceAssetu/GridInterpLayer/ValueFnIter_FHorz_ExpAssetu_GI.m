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


