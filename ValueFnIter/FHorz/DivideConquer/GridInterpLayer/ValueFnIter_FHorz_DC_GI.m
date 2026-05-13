function [V,Policy]=ValueFnIter_FHorz_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=max(ceil(n_a(1)/50),5); % minimum of 5
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    elseif length(n_a)==2
        vfoptions.level1n=[max(ceil(n_a(1)/50),5),n_a(2)]; % default is DC2B, min of 5 points in level1 for a1
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a); % Otherwise causes errors

%% 1 endogenous state
if isscalar(n_a)
    if N_e==0
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_e
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
%% 2 endogenous states
elseif length(n_a)==2
    if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
        vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
        if N_e==0
            if N_z==0
                if N_d==0
                    [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    if vfoptions.lowmemory==0
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over z
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_lowmem_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_lowmem_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                end
            end
        else % N_e
            if N_z==0
                if N_d==0
                    if vfoptions.lowmemory==0
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_noz_e_lowmem_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_noz_e_lowmem_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                end
            else
                if N_d==0
                    if vfoptions.lowmemory==0
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_e_lowmem_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==2 % loop over e and z
                        [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_e_lowmem2_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    end
                else
                    if vfoptions.lowmemory==0
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==1 % loop over e
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_e_lowmem_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                    elseif vfoptions.lowmemory==2 % loop over e and z
                        [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_e_lowmem2_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
if N_d==0
    Case2policies=[n_a,vfoptions.ngridinterp];
else
    Case2policies=[n_d,n_a,vfoptions.ngridinterp];
end

if vfoptions.outputkron==0
    if N_e==0
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron, Case2policies, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, Case2policies, n_a, n_z, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, Case2policies, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, Case2policies, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end



end
