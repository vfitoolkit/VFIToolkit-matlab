function [V,Policy]=ValueFnIter_FHorz_ExpAssetz_DC_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==1

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_e=prod(vfoptions.n_e);

%% Divide-and-conquer level1n setup (divide-and-conquer requires the standard endogenous state)
if N_a1==0
    error('Cannot use vfoptions.divideandconquer with experience asset if there is no standard endogenous state (N_a1==0)')
end
if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a1(1)));
    if n_a1(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a1);

%% DC2A+GI2A path: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, divide-and-conquer and grid interpolation layer), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
if length(n_a1)>1
    % a1_gridvals holds all the standard endogenous states (rows cycle the first fastest); n_a2/a2_grid hold the experience asset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_gridvals=a1_gridvals(1:n_a1(1):end,2:end);
    a1_grid=a1_gridvals(1:n_a1(1),1);
    n_a2=n_a1(2:end);
    n_a1=n_a1(1);

    if length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a2(1)
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With ExpAssetz DC2A+GI2A, can only do divide-and-conquer on the first standard endogenous state')
        end
    end
    if N_e>0
        if N_d1==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC2A_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=n_d2;
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            nDPolicyChannel=[n_d1,n_d2];
        end
        if vfoptions.outputkron==1
            V=VKron;
            Policy=PolicyKron;
            return
        end
        n_a=[n_a1,n_a2,n_a3];
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        return
    end

    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC2A_GI2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        nDPolicyChannel=n_d2;
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC2A_GI2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        nDPolicyChannel=[n_d1,n_d2];
    end

    if vfoptions.outputkron==1
        V=VKron;
        Policy=PolicyKron;
        return
    end
    n_a=[n_a1,n_a2,n_a3];
    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, N_j, vfoptions);
    return
end

%% Dispatch (single DC dim — existing DC1+GI1 path)
if N_e==0 % no e variable
    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC1_GI1_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC1_GI1_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else % N_e
    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC1_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else % d1 variable
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetz_DC1_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if n_d1>0
    n_d=[n_d1,n_d2];
else
    n_d=n_d2;
end
n_a=[n_a1,n_a2];

% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_e==0
    V=reshape(VKron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d, n_a1, n_a, n_z, N_j, vfoptions);
else
    V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
    Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron, n_d, n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
end

end
