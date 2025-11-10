function [V,Policy]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d_gridvals , d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_grid, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

N_d1=prod(n_d1);
% N_d2=prod(n_d2);
% N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_z=prod(n_z);
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end

%%
if N_a1==0
    error('Have not implemented experience assets with semi-exogenous shocks, without also having a standard asset')
end

if N_e>0
    if vfoptions.divideandconquer==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else
    if vfoptions.divideandconquer==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_grid, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_GI_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_grid, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_grid, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetuSemiExo_DC1_GI_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,n_u, N_j, d_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, u_grid, pi_z_J, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end



%%
if vfoptions.outputkron==0
    if n_d1>0
        n_d=[n_d1,n_d2,n_d3];
    else 
        n_d=[n_d2,n_d3];
    end
    if n_a1>0
        n_a=[n_a1,n_a2];
        n_d=[n_d,n_a1];
    else
        n_a=n_a2;
    end
    n_d=[n_d,vfoptions.ngridinterp]; % for the L2 indexes
    % Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, [n_z, n_semiz], vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,n_semiz,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_semiz, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, [n_semiz, n_z], N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end


