function [V,Policy]=ValueFnIter_FHorz_SemiExo_pard2(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_grid,d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_z=prod(n_z);
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end

if vfoptions.pard2==1
    [V,Policy]=ValueFnIter_FHorz_SemiExo_pard2(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_grid,d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
end

% NOTE: I TRIED PARALLEL OVER d2, BUT IT SEEMED TO BE SLOWER RATHER THAN FASTER. NOT SURE WHY.
if N_d1==0
    error('vfoptions.pard2 not yet implemented for this combo')
else
    if N_e==0
        error('vfoptions.pard2 not yet implemented for this combo')
    else
        if vfoptions.divideandconquer==0
            if N_z==0
                error('vfoptions.pard2 not yet implemented for this combo')
            else
                [VKron, Policy]=ValueFnIter_FHorz_SemiExo_pard2_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        elseif vfoptions.divideandconquer==1
            if N_z==0
                error('vfoptions.pard2 not yet implemented for this combo')
            else
                [VKron, Policy]=ValueFnIter_FHorz_SemiExo_DC1_pard2_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_grid, d2_grid, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end



% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
n_d=[n_d1,n_d2];
if vfoptions.pard2==1
    N_a=prod(n_a);
    N_semiz=prod(n_semiz);
end
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_semiz, vfoptions.n_e,N_j]);
            if vfoptions.pard2==0
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz_e(Policy3, n_d1,n_d2, n_a, vfoptions.n_semiz, vfoptions.n_e, N_j, vfoptions);
            elseif vfoptions.pard2==1
                Policy=UnKronPolicyIndexes_Case1_FHorz_e(Policy, n_d, n_a, vfoptions.n_semiz, vfoptions.n_e, N_j, vfoptions);
            end
        else
            V=reshape(VKron,[n_a,vfoptions.n_semiz,n_z,vfoptions.n_e,N_j]);
            if vfoptions.pard2==0
                if vfoptions.gridinterplayer==0
                    Policy=UnKronPolicyIndexes_Case1_FHorz_semiz_e(Policy3, n_d1,n_d2, n_a, [vfoptions.n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
                elseif vfoptions.gridinterplayer==1
                    Policy=UnKronPolicyIndexes_Case2_FHorz_e(Policy, [n_d,n_a,vfoptions.ngridinterp], n_a, [vfoptions.n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
                end
            elseif vfoptions.pard2==1
                Policy=UnKronPolicyIndexes_Case2_FHorz_e(reshape(Policy,[N_a,N_semiz*N_z,N_e,N_j]), [n_d,n_a], n_a, [vfoptions.n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_semiz,N_j]);
            if vfoptions.pard2==0
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1, n_d2, n_a, vfoptions.n_semiz, N_j, vfoptions);
            elseif vfoptions.pard2==1
                Policy=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, vfoptions.n_semiz, N_j, vfoptions);
            end
        else
            V=reshape(VKron,[n_a,vfoptions.n_semiz,n_z,N_j]);
            if vfoptions.pard2==0
                Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1, n_d2, n_a, [vfoptions.n_semiz,n_z], N_j, vfoptions);
            elseif vfoptions.pard2==1
                Policy=UnKronPolicyIndexes_Case2_FHorz(reshape(Policy,[N_a,N_semiz*N_z,N_e,N_j]), [n_d,n_a], n_a, [vfoptions.n_semiz,n_z], N_j, vfoptions);
            end
        end
    end
else
    V=VKron;
    Policy=Policy3;
end

    

end
