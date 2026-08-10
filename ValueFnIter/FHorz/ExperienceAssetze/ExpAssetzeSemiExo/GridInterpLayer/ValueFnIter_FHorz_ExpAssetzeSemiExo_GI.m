function [V,Policy]=ValueFnIter_FHorz_ExpAssetzeSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. (required)

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%%
if N_a1==0
    error('Have not implemented experience assets with semi-exogenous shocks, without also having a standard asset')
end

%% GI2A path: multi-dim n_a1 (first standard endo state GI'd, remaining folded; n_a2 is expasset)
if length(n_a1)>1
    n_a1DC=n_a1(1);
    n_a1fold=n_a1(2:end);
    N_a1DC=prod(n_a1DC);
    a1DC_grid=a1_gridvals(1:N_a1DC,1);
    a1fold_gridvals=a1_gridvals(1:N_a1DC:end,2:end);

    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzeSemiExo_GI2A_nod1_e_raw(n_d2, n_d3, n_a1DC, n_a1fold, n_a2, n_z, n_semiz, vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        nDPolicyChannel=n_d2;
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzeSemiExo_GI2A_e_raw(n_d1, n_d2, n_d3, n_a1DC, n_a1fold, n_a2, n_z, n_semiz, vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        nDPolicyChannel=[n_d1,n_d2];
    end

    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    n_bothz=[n_semiz,n_z];
    n_a=[n_a1,n_a2];
    V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, nDPolicyChannel, n_d3, n_a1DC, n_a1fold, n_a, n_bothz, vfoptions.n_e, N_j, vfoptions);
    return
end

if N_d1==0
    [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzeSemiExo_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
else
    [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzeSemiExo_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

n_bothz=[n_semiz,n_z];
n_a=[n_a1,n_a2];

V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
if N_d1==0
    Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
else
    Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
end


end
