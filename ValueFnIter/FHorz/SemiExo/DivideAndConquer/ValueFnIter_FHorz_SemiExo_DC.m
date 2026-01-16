function [V,Policy]=ValueFnIter_FHorz_SemiExo_DC(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

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
end

if isscalar(n_a)
    if N_d1==0
        if N_e==0
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_noz_raw(n_d1, n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_raw(n_d1, n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_noz_e_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC1_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
%% 2 endogenous states
elseif length(n_a)==2
    if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
        vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
        error('DC2B with semiexo is currently in progress')
        if N_d1==0
            if N_e==0
                if N_z==0
                    [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    if vfoptions.lowmemory==0
                        [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over z

                    end
                end
            else
                if N_z==0
                    if vfoptions.lowmemory==0
                        [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e

                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_nod1_e_raw(n_d2,n_a,n_z,n_semiz,  vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e

                    elseif vfoptions.lowmemory==2 % loop over z & e

                    end
                end
            end
        else
            if N_e==0
                if N_z==0
                    [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_noz_raw(n_d1, n_d2,n_a,n_semiz, N_j, d1_grid, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    if vfoptions.lowmemory==0
                        [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_raw(n_d1, n_d2,n_a,n_z,n_semiz, N_j, d1_grid, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over z

                    end
                end
            else
                if N_z==0
                    if vfoptions.lowmemory==0
                        [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_noz_e_raw(n_d1,n_d2,n_a,vfoptions.n_semiz, vfoptions.n_e, N_j, d1_grid, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e

                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron, Policy3]=ValueFnIter_FHorz_SemiExo_DC2B_e_raw(n_d1,n_d2,n_a,n_z,vfoptions.n_semiz,  vfoptions.n_e, N_j, d1_grid, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e

                    elseif vfoptions.lowmemory==2 % loop over z & e

                    end
                end
            end
        end
    else
        error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
    end
else
    error('Cannot use vfoptions.divideandconquer with more than two endogenous states (you have length(n_a)>2)')
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
% First dimension of Policy3 is (d1,d2,aprime), or if no d1, then (d2,aprime)
if N_z==0
    n_bothz=vfoptions.n_semiz;
else
    n_bothz=[vfoptions.n_semiz,n_z];
end

% Because of how we have N_semiz*N_z together, use the _noz commands to UnKron
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        V=reshape(VKron,[n_a,n_bothz, vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(Policy3, n_d1,n_d2, n_a, n_bothz, vfoptions.n_e, N_j, vfoptions); % pretend e is z (as z is with semiz)
    else
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz_semiz_noz(Policy3, n_d1, n_d2, n_a, n_bothz, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=Policy3;
end



end
