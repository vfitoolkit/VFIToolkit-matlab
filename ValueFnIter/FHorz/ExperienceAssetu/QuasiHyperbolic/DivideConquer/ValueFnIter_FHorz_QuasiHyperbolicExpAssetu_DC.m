function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetu_DC(n_d1,n_d2,n_a1,n_a2,n_z,n_u, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==0
% Quasi-hyperbolic discounting version of ValueFnIter_FHorz_ExpAssetu_DC
% Outputs are returned via varargout: {V1, Policy, Valt, Policyalt}
% Naive:         V1=Vtilde (beta0*beta), Policy is its argmax; Valt/Policyalt are the beta pass.
% Sophisticated: V1=Vhat (beta0*beta), Policy is its argmax; the third output is Vunderbar
%                (the beta-RHS gathered at Policy) and Policyalt is [].

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% Divide-and-conquer level1n setup (divide-and-conquer requires the standard endogenous state)
if N_a1==0
    error('Cannot use vfoptions.divideandconquer with experience assetu if there is no standard endogenous state (N_a1==0)')
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

%% DC2A path: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, divide-and-conquer), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
if length(n_a1)>1
    % a1_gridvals holds all the standard endogenous states (rows cycle the first fastest); n_a2/a2_grid hold the experience asset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_gridvals=a1_gridvals(1:n_a1(1):end,2:end);
    a1_grid=a1_gridvals(1:n_a1(1),1);
    n_a2=n_a1(2:end);
    n_a1=n_a1(1);

    if length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a2(1) % only DC on the first endo state
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With ExpAssetu DC2A, can only do divide-and-conquer on the first standard endogenous state')
        end
    end
    if vfoptions.gridinterplayer==1
        error('vfoptions.gridinterplayer not yet supported with ExpAssetu DC2A (two or more standard endogenous states)')
    end

    if N_e==0 % no e variable
        if N_d1==0
            if N_z==0
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_nod1_noz_raw(n_d2, n_a1, n_a2, n_a3,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_nod1_noz_raw(n_d2, n_a1, n_a2, n_a3,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=n_d2;
            else
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=n_d2;
            end
        else % d1 variable
            if N_z==0
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_noz_raw(n_d1, n_d2, n_a1, n_a2, n_a3,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_noz_raw(n_d1, n_d2, n_a1, n_a2, n_a3,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=[n_d1,n_d2];
            else
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=[n_d1,n_d2];
            end
        end
    else % N_e
        if N_d1==0
            if N_z==0
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=n_d2;
            else
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=n_d2;
            end
        else % d1 variable
            if N_z==0
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_noz_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_noz_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=[n_d1,n_d2];
            else
                if isNaive
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
                end
                nDPolicyChannel=[n_d1,n_d2];
            end
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
        n_a=[n_a1,n_a2,n_a3];
        if N_e==0
            if N_z==0
                V1=reshape(V1Kron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_noz(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, N_j, vfoptions);
                Valt=reshape(ValtKron,[n_a,N_j]);
                if isNaive
                    Policyalt=UnKronPolicyIndexes3_FHorz_noz(PolicyaltKron, nDPolicyChannel, n_a1, n_a2, n_a, N_j, vfoptions);
                end
            else
                V1=reshape(V1Kron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, N_j, vfoptions);
                Valt=reshape(ValtKron,[n_a,n_z,N_j]);
                if isNaive
                    Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, N_j, vfoptions);
                end
            end
        else
            if N_z==0
                V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
                Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
                if isNaive
                    Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron, nDPolicyChannel, n_a1, n_a2, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
                end
            else
                V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
                Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
                if isNaive
                    Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
                end
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

%% Dispatch (single DC dim — the DC1 path)
if N_e==0 % no e variable
    if N_d1==0
        if N_z==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_nod1_noz_raw(n_d2,n_a1,n_a2,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_nod1_noz_raw(n_d2,n_a1,n_a2,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_nod1_raw(n_d2,n_a1,n_a2,n_z,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_nod1_raw(n_d2,n_a1,n_a2,n_z,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end
    else % d1 variable
        if N_z==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_noz_raw(n_d1,n_d2,n_a1,n_a2,n_u, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_noz_raw(n_d1,n_d2,n_a1,n_a2,n_u, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid,u_gridvals,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_u, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_raw(n_d1,n_d2,n_a1,n_a2,n_z,n_u, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J,u_gridvals, pi_z_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e,n_u, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end
    else % d1 variable
        if N_z==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J,u_gridvals, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuN_DC1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetuS_DC1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e,n_u, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J,u_gridvals, pi_z_J, vfoptions.pi_e_J,pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
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
    if N_e==0
        if N_z==0
            V1=reshape(V1Kron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_noz(PolicyaltKron, n_d, n_a, N_j, vfoptions);
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron, n_d, n_a, n_z, N_j, vfoptions);
            end
        end
    else
        if N_z==0
            V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end


end
