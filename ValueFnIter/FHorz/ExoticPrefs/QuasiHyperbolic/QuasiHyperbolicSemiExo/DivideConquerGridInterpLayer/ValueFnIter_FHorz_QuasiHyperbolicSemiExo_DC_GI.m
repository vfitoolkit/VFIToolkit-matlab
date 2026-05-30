function [V1,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC_GI(n_d1,n_d2,n_a,n_semiz,n_z,N_j,d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic + SemiExo + DC + GridInterpLayer sub-dispatcher.
% Splices QH dual-V (V1, Valt) with SemiExo d2 outer loop and DC level1n + GI midpoint/L2 search (with L2flag).
% One endogenous state required (N_a1>0); requires vfoptions.level1n and vfoptions.ngridinterp.
%
% Interpretation of output differs by Naive/Sophisticated (same as plain QH DC_GI):
% Naive:         {Vtilde, Policy, V}
% Sophisticated: {Vhat,   Policy, Vunderbar}

N_d1=prod(n_d1);
N_a1=prod(n_a); % only one endogenous state allowed here
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% n_a1>0 / DC level1n / GI ngridinterp validation
if ~isscalar(n_a)
    error('ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC_GI currently only supports scalar n_a (one endogenous state)')
end

if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=round(sqrt(n_a(1)));
    if n_a(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a);

if ~isfield(vfoptions,'ngridinterp')
    error('You must declare vfoptions.ngridinterp when using the grid interpolation layer')
end

%% Dispatch: Naive vs Sophisticated, then N_d1 / N_e / N_z
if strcmp(vfoptions.quasi_hyperbolic,'Naive') % Output: [Vtilde,Policy,V]
    if N_d1==0
        if N_e==0
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_nod1_e_raw(n_d2,n_a,n_z,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_noz_raw(n_d1,n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_raw(n_d1,n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_noz_e_raw(n_d1,n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_DC1_GI1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
elseif strcmp(vfoptions.quasi_hyperbolic,'Sophisticated') % Output: [Vhat,Policy,Vunderbar]
    if N_d1==0
        if N_e==0
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_nod1_noz_raw(n_d2,n_a,n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_nod1_raw(n_d2,n_a,n_z,n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_nod1_noz_e_raw(n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_nod1_e_raw(n_d2,n_a,n_z,n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_e==0
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_noz_raw(n_d1,n_d2,n_a,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_raw(n_d1,n_d2,n_a,n_z,n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_noz_e_raw(n_d1,n_d2,n_a,n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [V1Kron, PolicyKron, ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_DC1_GI1_e_raw(n_d1,n_d2,n_a,n_z,n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
    return
end

% Because of how we have N_semiz*N_z together, use the _z commands to UnKron
if N_z==0
    n_bothz=vfoptions.n_semiz;
else
    n_bothz=[vfoptions.n_semiz,n_z];
end

% First dimension of PolicyKron is (d1,d2,aprime), or if no d1, then (d2,aprime)
if N_d1==0
    if N_e==0
        V1=reshape(V1Kron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,N_j]);
    else
        V1=reshape(V1Kron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    end
else
    if N_e==0
        V1=reshape(V1Kron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,N_j]);
    else
        V1=reshape(V1Kron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    end
end


end
