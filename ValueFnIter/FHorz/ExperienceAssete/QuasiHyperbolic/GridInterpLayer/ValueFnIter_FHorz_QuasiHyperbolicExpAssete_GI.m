function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssete_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0)
% vfoptions are already set by ValueFnIter_FHorz()
% Quasi-hyperbolic discounting version of ValueFnIter_FHorz_ExpAssete_GI
% Outputs are returned via varargout: {V1, Policy, Valt, Policyalt}
% (Policyalt is [] when Sophisticated).

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%%

if N_a1==0
    error('Cannot use grid interpolation layer if there is no standard endogenous state')
end

%% GI2A path: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, grid interpolation layer), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
if length(n_a1)>1
    % a1_gridvals holds all the standard endogenous states (rows cycle the first fastest); n_a2/a2_grid hold the experience asset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_gridvals=a1_gridvals(1:n_a1(1):end,2:end);
    a1_grid=a1_gridvals(1:n_a1(1),1);
    n_a2=n_a1(2:end);
    n_a1=n_a1(1);

    if N_z==0
        if N_d1==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI2A_nod1_noz_e_raw(n_d2, n_a1, n_a2, n_a3, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
            nDPolicyChannel=n_d2;
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI2A_noz_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI2A_noz_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
            nDPolicyChannel=[n_d1,n_d2];
        end
    else
        if N_d1==0
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI2A_nod1_e_raw(n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            end
            nDPolicyChannel=n_d2;
        else
            if isNaive
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI2A_e_raw(n_d1, n_d2, n_a1, n_a2, n_a3, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
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
        n_a=[n_a1,n_a2,n_a3];
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

    if isNaive
        varargout={V1, Policy, Valt, Policyalt};
    else
        varargout={V1, Policy, Valt, []};
    end
    return
end

if N_d1==0
    if N_z==0
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        end
    else
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        end
    end
else % d1 variable
    if N_z==0
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        end
    else
        if isNaive
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteN_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAsseteS_GI1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, beta0);
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
        n_d=[n_d1,n_d2];
    else
        n_d=n_d2;
    end
    n_a=[n_a1,n_a2];

    % Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if N_z==0
        V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d, n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron, n_d, n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
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



end
