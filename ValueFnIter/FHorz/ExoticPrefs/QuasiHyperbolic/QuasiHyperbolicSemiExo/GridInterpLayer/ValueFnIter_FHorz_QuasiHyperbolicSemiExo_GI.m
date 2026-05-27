function [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_GI(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic + SemiExo + GridInterpLayer sub-dispatcher.
% Mirrors ValueFnIter_FHorz_QuasiHyperbolic_GI.m and ValueFnIter_FHorz_SemiExo_GI.m.
%
% Naive:         varargout = {Vtilde, Policy, V}
% Sophisticated: varargout = {Vhat,   Policy, Vunderbar}

N_d1=prod(n_d1);
N_a1=prod(n_a);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end
if ~isscalar(n_a)
    error('ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC currently only supports scalar n_a (one endogenous state)')
end


%% 8-way nested dispatch on (N_e, N_d1, N_z)
if isscalar(n_a)
    if strcmp(vfoptions.quasi_hyperbolic,'Naive') % Output: [Vtilde, Policy, V]
        if N_e==0
            if N_z==0
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_raw(n_d1, n_d2, n_a, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_z==0
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_nod1_noz_e_raw(n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_noz_e_raw(n_d1, n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_nod1_e_raw(n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_GI1_e_raw(n_d1, n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') % Output: [Vhat, Policy, Vunderbar]
        if N_e==0
            if N_z==0
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_raw(n_d1, n_d2, n_a, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_z==0
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_nod1_noz_e_raw(n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_noz_e_raw(n_d1, n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_nod1_e_raw(n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_GI1_e_raw(n_d1, n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    end
else
    error('ValueFnIter_FHorz_QuasiHyperbolicSemiExo_GI currently only supports scalar n_a (one endogenous state)')
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_z==0
    n_bothz=n_semiz;
else
    n_bothz=[n_semiz,n_z];
end
if N_d1==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end

if vfoptions.outputkron==0
    if N_e==0
        V1=reshape(V1Kron,[n_a,n_bothz,N_j]);
        Valt=reshape(ValtKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, [n_d,n_a,vfoptions.ngridinterp], n_a, n_bothz, N_j, vfoptions);
    else
        V1=reshape(V1Kron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, [n_d,n_a,vfoptions.ngridinterp], n_a, n_bothz, vfoptions.n_e, N_j, vfoptions);
    end
else
    V1=V1Kron;
    Valt=ValtKron;
    Policy=PolicyKron;
end

end
