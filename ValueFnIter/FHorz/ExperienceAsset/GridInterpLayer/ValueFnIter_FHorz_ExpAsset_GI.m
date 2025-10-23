function [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_grid, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end


%%

if N_a1==0
    error('Cannot use grid interpolation layer if there is no standard endogenous state')
end

if N_e>0
    if vfoptions.divideandconquer==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else % no e variable
    if vfoptions.divideandconquer==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_nod1_noz_raw(n_d2,n_a1,n_a2, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_noz_raw(n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_GI_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_nod1_noz_raw(n_d2,n_a1,n_a2, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_noz_raw(n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals, d2_grid, a1_gridvals, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAsset_DC1_GI_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_grid, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end

%%
if vfoptions.outputkron==0
    if n_d1>0
        n_d=[n_d1,n_d2];
    else 
        n_d=n_d2;
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
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end


