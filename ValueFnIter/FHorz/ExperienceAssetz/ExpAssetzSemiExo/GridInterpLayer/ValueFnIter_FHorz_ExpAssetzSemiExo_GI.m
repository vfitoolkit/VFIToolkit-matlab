function [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state

N_d1=prod(n_d1);
N_a1=prod(n_a1);
% N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%%
if N_a1==0
    error('Have not implemented experience assets with semi-exogenous shocks, without also having a standard asset')
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

    if N_e==0
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI2A_nod1_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI2A_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
    else
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI2A_nod1_e_raw(n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI2A_e_raw(n_d1, n_d2, n_d3, n_a1, n_a2, n_a3, n_z, n_semiz, vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
    end

    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    n_bothz=[n_semiz,n_z];
    n_a=[n_a1,n_a2,n_a3];
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, nDPolicyChannel, n_d3, n_a1, n_a2, n_a, n_bothz, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, nDPolicyChannel, n_d3, n_a1, n_a2, n_a, n_bothz, vfoptions.n_e, N_j, vfoptions);
    end
    return
end

if N_e==0
    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else
    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

n_bothz=[n_semiz,n_z];
n_a=[n_a1,n_a2];

if N_e==0
    V=reshape(VKron,[n_a,n_bothz,N_j]);
    if N_d1==0
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
    else
        Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
    end
else
    V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    if N_d1==0
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    else
        Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
end


end
