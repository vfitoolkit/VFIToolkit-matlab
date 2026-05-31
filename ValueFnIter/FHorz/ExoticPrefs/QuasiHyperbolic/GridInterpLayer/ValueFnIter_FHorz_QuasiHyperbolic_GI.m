function varargout=ValueFnIter_FHorz_QuasiHyperbolic_GI(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting wrapper for GI finite-horizon value function iteration.
% Dispatches to ValueFnIter_FHorz_QuasiHyperbolicN_GI1_*_raw or ValueFnIter_FHorz_QuasiHyperbolicS_GI1_*_raw.
%
% Outputs are returned via varargout. Caller may request:
%   Naive:         [V, Policy] or [V, Policy, Valt] or [V, Policy, Valt, Policyalt]
%                  V=Vtilde (QH-discounted), Policy=QH-optimal, Valt=V_std, Policyalt=std-optimal
%   Sophisticated: [V, Policy] or [V, Policy, Valt]
%                  V=Vhat (QH-discounted), Policy=equilibrium, Valt=Vunderbar (realised)

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%%
if isscalar(n_a)
    if isNaive % Output: [V=Vtilde, Policy, Valt=V_std, Policyalt]
        if N_e==0
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_GI1_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    else % Sophisticated. Output: [V=Vhat, Policy, Valt=Vunderbar]
        if N_e==0
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_z==0
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicS_GI1_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    end
else
    error('ValueFnIter_FHorz_GI1_QuasiHyperbolic currently only supports scalar n_a (one endogenous state)')
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
    if isNaive
        Policyalt=PolicyaltKron;
    end
else
    if N_d==0
        if N_e==0
            if N_z==0
                Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_a,n_a,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes1_FHorz_noz(PolicyaltKron,n_a,n_a,N_j,vfoptions);
                end
            else
                Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,n_a,n_z,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_a,n_a,n_z,N_j,vfoptions);
                end
            end
        else
            if N_z==0
                Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_a,n_a,vfoptions.n_e,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_a,n_a,vfoptions.n_e,N_j,vfoptions);
                end
            else
                Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
                end
            end
        end
    else
        if N_e==0
            if N_z==0
                Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron,n_d,n_a,n_a,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes2_FHorz_noz(PolicyaltKron,n_d,n_a,n_a,N_j,vfoptions);
                end
            else
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,n_a,n_z,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron,n_d,n_a,n_a,n_z,N_j,vfoptions);
                end
            end
        else
            if N_z==0
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d,n_a,n_a,vfoptions.n_e,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron,n_d,n_a,n_a,vfoptions.n_e,N_j,vfoptions);
                end
            else
                Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
                if isNaive
                    Policyalt=UnKronPolicyIndexes2_FHorz_z_e(PolicyaltKron,n_d,n_a,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
                end
            end
        end
    end

    if N_e==0
        if N_z==0
            V1=reshape(V1Kron,[n_a,N_j]);
            Valt=reshape(ValtKron,[n_a,N_j]);
        else
            V1=reshape(V1Kron,[n_a,n_z,N_j]);
            Valt=reshape(ValtKron,[n_a,n_z,N_j]);
        end
    else
        if N_z==0
            V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
            Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
        else
            V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
        end
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt};
end


end
