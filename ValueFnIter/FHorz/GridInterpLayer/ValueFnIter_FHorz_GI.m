function [V,Policy]=ValueFnIter_FHorz_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% 1 endogenous state
if isscalar(n_a)
    if N_e==0
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GI1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_GI1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GI1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_e
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_GI1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GI1_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_GI1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GI1_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
elseif length(n_a)==2
    %% 2 endogenous states
    if isscalar(vfoptions.ngridinterp)
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_GI2A_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_GI2A_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_GI2A_nod_raw(n_a,n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_GI2A_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_GI2A_nod_noz_e_raw(n_a,vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_GI2A_noz_e_raw(n_d,n_a,vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_GI2A_nod_e_raw(n_a,n_z,vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_GI2A_e_raw(n_d,n_a,n_z,vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    else
        error('vfoptions.gridinterplayer=1 with two endogenous states can only be applied to the first of the two endo states (you have length(vfoptions.ngridinterp)>1)')
    end
else
    error('Cannot use vfoptions.divideandconquer with more than two endogenous states (you have length(n_a)>2)')
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if isscalar(n_a)
    if N_d==0
        if N_e==0
            if N_z==0
                Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_a,n_a,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,n_a,n_z,N_j,vfoptions);
            end
        else
            if N_z==0
                Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,n_a,vfoptions.n_e,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron,n_d,n_a,n_a,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,n_a,n_z,N_j,vfoptions);
            end
        else
            if N_z==0
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,n_a,vfoptions.n_e,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            end
        end
    end
else % length(n_a)
    n_a1=n_a(1);
    n_a2=n_a(2:end);
    if N_d==0
        if N_e==0
            if N_z==0
                Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron,n_a1,n_a2,n_a,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_a1,n_a2,n_a,n_z,N_j,vfoptions);
            end
        else
            if N_z==0
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_a1,n_a2,n_a,vfoptions.n_e,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_a1,n_a2,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                Policy=UnKronPolicyIndexes3_FHorz_noz(PolicyKron,n_d,n_a1,n_a2,n_a,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,n_a,n_z,N_j,vfoptions);
            end
        else
            if N_z==0
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d,n_a1,n_a2,n_a,vfoptions.n_e,N_j,vfoptions);
            else
                Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d,n_a1,n_a2,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            end
        end
    end
end

if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
    end
end



end
