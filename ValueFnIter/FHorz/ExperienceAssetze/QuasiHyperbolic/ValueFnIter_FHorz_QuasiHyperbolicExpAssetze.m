function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetze(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting with an experienceassetze state (z+e dependent aprimeFn).
% e is structural (always required for ExpAssetze).
% Mirrors ValueFnIter_FHorz_QuasiHyperbolicExpAssetz dispatcher.

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

l_d2=length(n_d2);
l_a2=length(n_a2);
l_z=length(n_z);
l_e=length(vfoptions.n_e);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z+l_e)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+l_e+1:end}};
else
    aprimeFnParamNames={};
end

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_e==0
    error('experienceassetze requires n_e>0 (e is structural for z+e aprimeFn)')
end

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d_gridvals=[];
end

e_gridvals_J=vfoptions.e_gridvals_J;
pi_e_J=vfoptions.pi_e_J;

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% DC2A / GI2A / DC2A_GI2A branch (multi-dim n_a1)
if length(n_a1)>1 && (vfoptions.divideandconquer==1 || vfoptions.gridinterplayer==1)
    n_a1DC=n_a1(1);
    n_a1fold=n_a1(2:end);
    N_a1DC=prod(n_a1DC);
    a1DC_grid=a1_gridvals(1:N_a1DC,1);
    a1fold_gridvals=a1_gridvals(1:N_a1DC:end,2:end);

    if ~isfield(vfoptions,'level1n')
        vfoptions.level1n=floor(sqrt(n_a1DC));
    end
    vfoptions.level1n=min(vfoptions.level1n,n_a1DC);

    if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==0
        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_DC2A_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
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


%% Baseline (no DC, no GI)
if vfoptions.divideandconquer==1 || vfoptions.gridinterplayer==1
    error('QuasiHyperbolic+ExpAssetze DC/GI requires multi-dim n_a1 (DC2A path).')
end

if isNaive
    if N_a1==0
        error('noa1 variant not yet implemented')
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeN_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else
    if N_a1==0
        error('noa1 variant not yet implemented')
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeS_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end

%% Unkron (baseline)
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
    if prod(n_d)==0
        n_daprime=n_a;
    else
        n_daprime=[n_d,n_a1];
    end
    V1=reshape(V1Kron,[n_a,n_z,vfoptions.n_e,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    Valt=reshape(ValtKron,[n_a,n_z,vfoptions.n_e,N_j]);
    if isNaive
        Policyalt=UnKronPolicyIndexes1_FHorz_z_e(PolicyaltKron,n_daprime,n_a,n_z,vfoptions.n_e,N_j,vfoptions);
    end
end
if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end

end
