function varargout=ValueFnIter_FHorz_QuasiHyperbolicSemiExo(n_d, n_a, n_z, n_semiz, N_j, d_grid, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic preferences with semi-exogenous shock.
% Splices the dual-V (V1, Valt) structure of QuasiHyperbolic with the d2 outer loop / pi_bothz=kron(pi_z,pi_semiz) of SemiExo.
%
% Outputs are returned via varargout. Caller may request:
%   Naive:         [V, Policy] or [V, Policy, Valt] or [V, Policy, Valt, Policyalt]
%                  V=Vtilde (QH-discounted), Policy=QH-optimal, Valt=V_std, Policyalt=std-optimal
%   Sophisticated: [V, Policy] or [V, Policy, Valt]
%                  V=Vhat (QH-discounted), Policy=equilibrium, Valt=Vunderbar (realised)
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount gives the name of beta_0 (the additional discount factor parameter)
% vfoptions.l_dsemiz gives the number of decision variables that control the semi-exogenous transitions

% Validate quasi-hyperbolic settings
if ~isfield(vfoptions,'quasi_hyperbolic')
    error('You are using quasi-hyperbolic discounting (vfoptions.exoticpreferences) but you have not specified vfoptions.quasi_hyperbolic, which must be either Naive or Sophisticated')
elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
    error('vfoptions.quasi_hyperbolic must be either Naive or Sophisticated (check spelling and capital letter) \n')
end

if ~isfield(vfoptions,'QHadditionaldiscount')
    error('You must declare vfoptions.QHadditionaldiscount when using quasi-hyperbolic discounting (you have vfoptions.exoticpreferences set to QuasiHyperbolic)')
end

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% Split n_d into n_d1 (other decisions) and n_d2 (semiz controller)
l_dsemiz=vfoptions.l_dsemiz;
if length(n_d)==l_dsemiz
    n_d1=0;
    n_d2=n_d;
else
    n_d1=n_d(1:end-l_dsemiz);
    n_d2=n_d(end-l_dsemiz+1:end);
end

N_d1=prod(n_d1);
% N_d2=prod(n_d2);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

% Split d_gridvals into d1_gridvals and d2_gridvals
if N_d1==0
    d1_gridvals=[];
    d2_grid=d_grid;
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
else
    d1_grid=d_grid(1:n_d1);
    d2_grid=d_grid(n_d1+1:end);
    d1_gridvals=CreateGridvals(n_d1,d1_grid,1);
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
end

%% Sub-dispatch on DC/GI/DC_GI (siblings live in DC/, GI/, DC_GI/ subfolders)
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC_GI(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC_GI(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout={V1, Policy, Valt};
    end
    return
elseif vfoptions.divideandconquer==1
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_DC(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout={V1, Policy, Valt};
    end
    return
elseif vfoptions.gridinterplayer==1
    if isNaive
        [V1, Policy, Valt, Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_GI(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout={V1, Policy, Valt, Policyalt};
    else
        [V1, Policy, Valt]=ValueFnIter_FHorz_QuasiHyperbolicSemiExo_GI(n_d1,n_d2, n_a, n_semiz, n_z, N_j, d1_gridvals,d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        varargout={V1, Policy, Valt};
    end
    return
end


%% 8-way nested dispatch on (N_e, N_d1, N_z)
if isNaive % Output: [V=Vtilde, Policy, Valt=V_std, Policyalt]
    if N_e==0
        if N_z==0
            if N_d1==0
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_raw(n_d1, n_d2, n_a, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_z==0
            if N_d1==0
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_nod1_noz_e_raw(n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_noz_e_raw(n_d1, n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_nod1_e_raw(n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VtildeKron, PolicyKron, ValtKron, PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoN_e_raw(n_d1, n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
else % Sophisticated. Output: [V=Vhat, Policy, Valt=Vunderbar]
    if N_e==0
        if N_z==0
            if N_d1==0
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_nod1_noz_raw(n_d2, n_a, n_semiz, N_j, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_noz_raw(n_d1, n_d2, n_a, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_nod1_raw(n_d2, n_a, n_z, n_semiz, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_raw(n_d1, n_d2, n_a, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    else
        if N_z==0
            if N_d1==0
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_nod1_noz_e_raw(n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_noz_e_raw(n_d1, n_d2, n_a, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_nod1_e_raw(n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                [VhatKron, PolicyKron, VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicSemiExoS_e_raw(n_d1, n_d2, n_a, n_z, n_semiz, vfoptions.n_e, N_j, d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
        end
    end
end

%% Bridge semantic Kron names back to generic V1Kron/ValtKron for downstream reshape
if isNaive
    V1Kron=VtildeKron;
else
    V1Kron=VhatKron;
    ValtKron=VunderbarKron;
end

%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==1
    V1=V1Kron;
    Policy=PolicyKron;
    Valt=ValtKron;
    if isNaive
        Policyalt=PolicyaltKron;
        varargout={V1, Policy, Valt, Policyalt};
    else
        varargout={V1, Policy, Valt};
    end
    return
end

% Because of how we have N_semiz*N_z together, use the _z commands to UnKron
if N_z==0
    n_bothz=n_semiz;
else
    n_bothz=[n_semiz,n_z];
end

% First dimension of PolicyKron is (d1,d2,aprime), or if no d1, then (d2,aprime)
if N_d1==0
    if N_e==0
        V1=reshape(V1Kron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes2_FHorz_z(PolicyaltKron,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
        end
    else
        V1=reshape(V1Kron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes2_FHorz_z_e(PolicyaltKron,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    end
else
    if N_e==0
        V1=reshape(V1Kron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron,n_d1,n_d2,n_a,n_a,n_bothz,N_j,vfoptions);
        end
    else
        V1=reshape(V1Kron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d1,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        if isNaive
            Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron,n_d1,n_d2,n_a,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt};
end

end
