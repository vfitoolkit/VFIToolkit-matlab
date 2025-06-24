function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_grid,d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_z=prod(n_z);
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end

if vfoptions.divideandconquer==1
    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=max(ceil(n_a(1)/50),5); % minimum of 5
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end

    if length(n_a)>1
        error('vfoptions.divideandconquer==1 is currently only possible for one endogenous state (when using semi-exo); contact me if you want this')
    end
end

if isscalar(n_a)
    if N_d1==0
        if N_e==0
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_grid, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_noz_raw(n_d1, n_d2,n_a,n_semiz, N_j, d1_grid, d2_grid, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_raw(n_d1, n_d2,n_a,n_z,n_semiz, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_noz_e_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, vfoptions.n_e, N_j, d1_grid, d2_grid, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else
    error('vfoptions.divideandconquer=1 for semi-exo only supports one endogenous state (currently)')
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% First dimension of Policy3 is (d1,d2,aprime), or if no d1, then (d2,aprime)
if N_z==0
    n_bothz=vfoptions.n_semiz;
else
    n_bothz=[vfoptions.n_semiz,n_z];
end

if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz_semiz_e(Policy3, n_d1,n_d2, n_a, n_bothz, vfoptions.n_e, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1, n_d2, n_a, n_bothz, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=Policy3;
end

    

end
