function varargout=ValueFnIter_FHorz_QuasiHyperbolic_DC(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting variant of ValueFnIter_FHorz_DC1_raw.
% Uses divide-and-conquer on a (GPU, parallel==2 only).
%
% Outputs are returned via varargout. Caller may request:
%   Naive:         [V, Policy] or [V, Policy, Valt] or [V, Policy, Valt, Policyalt]
%                  V=Vtilde (QH-discounted), Policy=QH-optimal, Valt=V_std, Policyalt=std-optimal
%   Sophisticated: [V, Policy] or [V, Policy, Valt]
%                  V=Vhat (QH-discounted), Policy=equilibrium, Valt=Vunderbar (realised)
%
% DiscountFactorParamNames is the standard discount factor beta
% vfoptions.QHadditionaldiscount.gives the name of the beta_0 is the additional discount factor parameter

N_d=prod(n_d);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=floor(sqrt(n_a(1)));
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    elseif length(n_a)==2
        vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch
        if n_a(1)<5
            error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
        end
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
else
    if ~isscalar(n_a) && isscalar(vfoptions.level1n)
        vfoptions.level1n=[vfoptions.level1n,n_a(2:end)];
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a);

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%%
if isscalar(n_a)
    if isNaive % Output: [V=Vtilde, Policy, Valt=V_std, Policyalt]
        if N_e==0
            if N_z==0
                if N_d==0
                    [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VtildeKron, PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VtildeKron, PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_z==0
                if N_d==0
                    [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VtildeKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicN_DC1_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    else % Sophisticated. Output: [V=Vhat, Policy, Valt=Vunderbar]
        if N_e==0
            if N_z==0
                if N_d==0
                    [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VhatKron, PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_noz_raw(n_d,n_a, N_j, d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VhatKron, PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_raw(n_d,n_a,n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        else
            if N_z==0
                if N_d==0
                    [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_noz_e_raw(n_d,n_a, vfoptions.n_e, N_j, d_gridvals, a_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            else
                if N_d==0
                    [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                else
                    [VhatKron,PolicyKron,VunderbarKron]=ValueFnIter_FHorz_QuasiHyperbolicS_DC1_e_raw(n_d,n_a, n_z, vfoptions.n_e, N_j, d_gridvals, a_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                end
            end
        end
    end
else
    error('ValueFnIter_FHorz_DC_QuasiHyperbolic currently only supports scalar n_a (one endogenous state)')
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
    end
else
    if N_d==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a];
    end
    if N_e==0
        if N_z==0
            V1=reshape(V1Kron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron,n_daprime,n_a,N_j,vfoptions);
            Valt=reshape(ValtKron,[n_a,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_noz(PolicyaltKron,n_daprime,n_a,N_j,vfoptions);
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_daprime,n_a,n_z,N_j,vfoptions);
            end
        end
    else
        if N_z==0
            V1=reshape(V1Kron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);  % Treat e as z (because no z)
            Valt=reshape(ValtKron,[n_a,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_daprime,n_a,vfoptions.n_e,N_j,vfoptions);
            end
        else
            V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
            end
        end
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt};
end


end
