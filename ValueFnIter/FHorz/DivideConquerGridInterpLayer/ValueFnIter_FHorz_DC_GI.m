function [V,Policy]=ValueFnIter_FHorz_DC_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=round(sqrt(n_a(1)));
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    elseif length(n_a)==2
        vfoptions.level1n=[round(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end

%% 1 endogenous state
if isscalar(n_a)
    if N_e==0
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_e
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI1_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC1_GI1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC1_GI1_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
%% 2 endogenous states
elseif length(n_a)>=2
    if length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
            vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2A
        else
            error('With two endogenous states, can only do divide-and-conquer in the first endogenous state (not in both)')
        end
    end

    if N_e==0
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else % N_e
        if N_z==0
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_noz_e_raw(n_d,n_a,  vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_DC2A_GI2A_e_raw(n_d,n_a,n_z,  vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
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
else
    % length(n_a)
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
