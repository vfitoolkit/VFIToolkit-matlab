function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetz_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% Quasi-hyperbolic ExpAssetz. Handles vfoptions.divideandconquer==0, vfoptions.gridinterplayer==1.
% Split out of ValueFnIter_FHorz_QuasiHyperbolicExpAssetz so each tier UnKrons its own
% results, mirroring the exponential ValueFnIter_FHorz_ExpAssetz_* dispatchers.

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% GI1 path: scalar n_a1 (the single standard endo state is the GI'd dim; n_a2 is the expasset)
if length(n_a1)<=1
    if N_a1==0
        error('Cannot use grid interpolation layer if there is no standard endogenous state')
    end

    if N_e==0 % no e variable
        if N_d1==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI1_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI1_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else % d1 variable
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI1_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI1_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end
    else % N_e
        if N_d1==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else % d1 variable
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end
    end

    % Unkron (GI1 path)
    if vfoptions.outputkron==1
        V1=V1Kron;
        Policy=PolicyKron;
        Valt=ValtKron;
        if isNaive
            Policyalt=PolicyaltKron;
        end
    else
        if n_d1>0
            n_d=[n_d1,n_d2];
        else
            n_d=n_d2;
        end
        n_a=[n_a1,n_a2];

        % Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
        if N_e==0
            V1=reshape(V1Kron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d, n_a1, n_a, n_z, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron, n_d, n_a1, n_a, n_z, N_j, vfoptions);
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron, n_d, n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes2_FHorz_z_e(PolicyaltKron, n_d, n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
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

%% 2A setup: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, DC/GI), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
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

if N_e>0
    e_gridvals_J=vfoptions.e_gridvals_J;
    pi_e_J=vfoptions.pi_e_J;

        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
        end

        % Unkron (DC2A+e)
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

if isNaive
    if N_d1==0
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    end
else
    if N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_nod1_raw(n_d2, n_a1, n_a2, n_a3, n_z, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
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
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
    if isNaive
        Policyalt=PolicyaltKron;
    end
else
    V1=reshape(V1Kron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, N_j, vfoptions);
    Valt=reshape(ValtKron,[n_a,n_z,N_j]);
    if isNaive
        Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron, nDPolicyChannel, n_a1, n_a2, n_a, n_z, N_j, vfoptions);
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end
return

end
