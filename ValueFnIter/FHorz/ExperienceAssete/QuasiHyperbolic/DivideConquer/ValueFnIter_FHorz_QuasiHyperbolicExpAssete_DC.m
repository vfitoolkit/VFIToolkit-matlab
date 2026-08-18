function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssete_DC(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==0
% Quasi-hyperbolic discounting version of ValueFnIter_FHorz_ExpAssete_DC
% Outputs are returned via varargout: {V1, Policy, Valt, Policyalt}
% (Policyalt is [] when Sophisticated).

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

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

%% DC2A path: multi-dim n_a1 (first standard endo state DC, remaining N-2 folded; n_a2 is expasset)
if length(n_a1)>1
    n_a1DC=n_a1(1);
    n_a1fold=n_a1(2:end);
    N_a1DC=prod(n_a1DC);
    % Split a1_gridvals (rows cycle DC fastest) into the DC grid and the folded-states gridvals
    a1DC_grid=a1_gridvals(1:N_a1DC,1);
    a1fold_gridvals=a1_gridvals(1:N_a1DC:end,2:end);

    if length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a1(2) % only DC on the first endo state
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With ExpAssete DC2A, can only do divide-and-conquer on the first standard endogenous state')
        end
    end
    if vfoptions.gridinterplayer==1
        error('vfoptions.gridinterplayer not yet supported with ExpAssete DC2A (multi-dim n_a1)')
    end

    if N_z==0
        if N_d1==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC2A_nod1_noz_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC2A_nod1_noz_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
            nDPolicyChannel=n_d2;
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC2A_noz_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC2A_noz_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
            nDPolicyChannel=[n_d1,n_d2];
        end
    else
        if N_d1==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
            nDPolicyChannel=n_d2;
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
            nDPolicyChannel=[n_d1,n_d2];
        end
    end

    if vfoptions.outputkron==1
        V1=V1Kron;
        Policy=PolicyKron;
        Valt=ValtKron;
        if isNaive
            Policyalt=PolicyaltKron;
        end
    else
        n_a=[n_a1,n_a2];
        if N_z==0
            V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, vfoptions.n_e, N_j, vfoptions); % treat e as z (no z dim present)
            Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, vfoptions.n_e, N_j, vfoptions); % treat e as z (no z dim present)
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    end

    if isNaive
        varargout={V1, Policy, Valt, Policyalt};
    else
        varargout={V1, Policy, Valt, []};
    end
    return
end

%% Dispatch (single DC dim — existing DC1 path)
if N_d1==0
    if N_z==0
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % d1 variable
    if N_z==0
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_DC1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_DC1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end


%%
if vfoptions.outputkron==1
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
    if isNaive
        Policyalt=PolicyaltKron;
    end
else
    if n_d1>0
        n_d=[n_d1,n_d2,n_a1];
    else
        n_d=[n_d2,n_a1];
    end
    n_a=[n_a1,n_a2];

    % Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if N_z==0
        V1=reshape(V1Kron,[n_a,vfoptions.n_e, N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,vfoptions.n_e, N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e, N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e, N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end


end
