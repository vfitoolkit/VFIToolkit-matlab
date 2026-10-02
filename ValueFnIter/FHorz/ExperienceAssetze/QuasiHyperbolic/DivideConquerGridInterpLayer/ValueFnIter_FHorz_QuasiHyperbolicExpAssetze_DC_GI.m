function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze_DC_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==1
% Quasi-hyperbolic discounting version of ValueFnIter_FHorz_ExpAssetze_DC_GI
% Outputs are returned via varargout: {V1, Policy, Valt, Policyalt}
% Naive:         V1=Vtilde (beta0*beta), Policy is its argmax; Valt/Policyalt are the beta pass.
% Sophisticated: V1=Vhat (beta0*beta), Policy is its argmax; Valt is Vunderbar (the beta-RHS
%                gathered at Policy) and Policyalt is [].
% e is structural for ExpAssetze (aprimeFn is a function of d2,a2,z,e), so there is only the _e variant.

N_d1=prod(n_d1);
N_a1=prod(n_a1);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

e_gridvals_J=vfoptions.e_gridvals_J;
pi_e_J=vfoptions.pi_e_J;

%% Divide-and-conquer requires the standard endogenous state
if N_a1==0
    error('Cannot use vfoptions.divideandconquer with experience asset if there is no standard endogenous state (N_a1==0)')
end

%% DC2A+GI2A path: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, divide-and-conquer and grid interpolation layer), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
% Moved verbatim from ValueFnIter_FHorz_QuasiHyperbolicExpAssetze; the tier is now chosen by
% the family dispatcher, so only this tier's raw calls remain (indentation kept as it was).
if length(n_a1)>1
    % a1_gridvals holds all the standard endogenous states (rows cycle the first fastest); n_a2/a2_grid hold the experience asset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_gridvals=a1_gridvals(1:n_a1(1):end,2:end);
    a1_grid=a1_gridvals(1:n_a1(1),1);
    n_a2=n_a1(2:end);
    n_a1=n_a1(1);

    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=floor(sqrt(n_a1));
    end
    vfoptions.level1n=min(vfoptions.level1n,n_a1);

        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end

    %% Unkron (DC2A path)
    n_a=[n_a1,n_a2,n_a3];
    if N_d1==0
        nDPolicyChannel=n_d2;
    else
        nDPolicyChannel=[n_d1,n_d2];
    end
    if vfoptions.outputkron==1
        V1=V1Kron; Policy=PolicyKron; Valt=ValtKron;
        if isNaive, Policyalt=PolicyaltKron; end
    else
        V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    end
    if isNaive
        varargout={V1, Policy, Valt, Policyalt};
    else
        varargout={V1, Policy, Valt, []};
    end
    return
end


%% Divide-and-conquer level1n setup (1A: the single standard endogenous state)
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

%% Dispatch (1A: scalar n_a1)
if isNaive
    if N_d1==0
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC1_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC1_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    end
else
    if N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC1_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC1_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    end
end

%% Unkron (1A path)
n_a=[n_a1,n_a2];
if N_d1==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end
if vfoptions.outputkron==1
    V1=V1Kron; Policy=PolicyKron; Valt=ValtKron;
    if isNaive, Policyalt=PolicyaltKron; end
else
    V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
    Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron, n_d, n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
    Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
    if isNaive
        Policyalt=UnKronPolicyIndexes2_FHorz_z_e(PolicyaltKron, n_d, n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
    end
end
if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end


end
