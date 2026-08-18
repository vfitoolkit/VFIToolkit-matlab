function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetz_DC_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% Quasi-hyperbolic ExpAssetz. Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==1.
% Split out of ValueFnIter_FHorz_QuasiHyperbolicExpAssetz so each tier UnKrons its own
% results, mirroring the exponential ValueFnIter_FHorz_ExpAssetz_* dispatchers.

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);
isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% 1A (scalar n_a1) is not yet implemented for this tier
if length(n_a1)<=1
    error('QuasiHyperbolic+ExpAssetz DC/GI requires multi-dim n_a1 (DC2A path). For scalar n_a1, only the baseline variant is implemented.')
end

%% 2A setup (first standard endo state DC/GI, remaining folded; n_a2 is the expasset)
n_a1DC=n_a1(1);
n_a1fold=n_a1(2:end);
N_a1DC=prod(n_a1DC);
a1DC_grid=a1_gridvals(1:N_a1DC,1);
a1fold_gridvals=a1_gridvals(1:N_a1DC:end,2:end);

if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a1DC));
end
vfoptions.level1n=min(vfoptions.level1n,n_a1DC);

if N_e>0
    e_gridvals_J=vfoptions.e_gridvals_J;
    pi_e_J=vfoptions.pi_e_J;

        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end

        % Unkron (DC2A+e)
        n_a=[n_a1,n_a2];
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
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
            if isNaive
                Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
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
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_GI2A_nod1_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_GI2A_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else
    if N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_GI2A_nod1_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_GI2A_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end

%% Unkron (DC2A path)
n_a=[n_a1,n_a2];
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
    Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, n_z, N_j, vfoptions);
    Valt=reshape(ValtKron,[n_a,n_z,N_j]);
    if isNaive
        Policyalt=UnKronPolicyIndexes3_FHorz_z(PolicyaltKron, nDPolicyChannel, n_a1DC, n_a1fold, n_a, n_z, N_j, vfoptions);
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end
return

end
