function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetz(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting with an experienceassetz state.
% Dispatches to Naive/Sophisticated raw files. Same semantics as
% ValueFnIter_FHorz_QuasiHyperbolic but with EV built via aprimeFn for the
% experienceassetz a2 state (mirrors ValueFnIter_FHorz_ExpAssetz).
%
% Coverage: baseline (no DC, no GI). DC2A/GI2A/DC2A_GI2A when length(n_a1)>1.

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

l_d2=length(n_d2);
l_a2=length(n_a2);
l_z=length(n_z);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+l_z)
    aprimeFnParamNames={temp{l_d2+l_a2+l_z+1:end}};
else
    aprimeFnParamNames={};
end

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d_gridvals=[];
end

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% _e branch: i.i.d. e shock present
if N_e>0
    e_gridvals_J=vfoptions.e_gridvals_J;
    pi_e_J=vfoptions.pi_e_J;

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
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
            if isNaive
                if N_d1==0
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_nod1_e_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_e_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
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

    %% Baseline _e (no DC, no GI)
    if vfoptions.divideandconquer==1 || vfoptions.gridinterplayer==1
        error('QuasiHyperbolic+ExpAssetz+e DC/GI requires multi-dim n_a1 (DC2A path).')
    end
    if isNaive
        if N_a1==0
            error('noa1 _e variant not yet implemented')
        elseif N_d1==0
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_a1==0
            error('noa1 _e variant not yet implemented')
        elseif N_d1==0
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_nod1_e_raw(n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_e_raw(n_d1,n_d2,n_a1,n_a2,n_z,vfoptions.n_e,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end

    % Unkron (baseline+e)
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
    return
end

%% DC2A / GI2A / DC2A_GI2A branch (multi-dim n_a1 + DC or GI requested)
% Raw signature: (..., n_a1=n_a1DC, n_a2=n_a1fold, n_a3=user's n_a2 expasset, ...)
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
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_nod1_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_DC2A_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_nod1_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_DC2A_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==0 && vfoptions.gridinterplayer==1
        if isNaive
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_nod1_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_GI2A_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_nod1_raw(n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_GI2A_raw(n_d1, n_d2, n_a1DC, n_a1fold, n_a2, n_z, N_j, d_gridvals, d2_gridvals, a1DC_grid, a1fold_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
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


%% Baseline dispatch (no DC, no GI; n_a1 may be scalar or multi-dim — collapsed via prod inside raws)
if vfoptions.divideandconquer==1 || vfoptions.gridinterplayer==1
    error('QuasiHyperbolic+ExpAssetz DC/GI requires multi-dim n_a1 (DC2A path). For scalar n_a1, only the baseline variant is implemented.')
end

if isNaive
    if N_a1==0
        error('noa1 variant not yet implemented')
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else
    if N_a1==0
        error('noa1 variant not yet implemented')
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end

%% Unkron (baseline)
n_a=[n_a1,n_a2];
N_a=prod(n_a);
if N_d1==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end
N_d=prod(n_d);

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
        n_daprime=[n_d,n_a1];
    end
    V1=reshape(V1Kron,[n_a,n_z,N_j]);
    Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron,n_daprime,n_a,n_z,N_j,vfoptions);
    Valt=reshape(ValtKron,[n_a,n_z,N_j]);
    if isNaive
        Policyalt=UnKronPolicyIndexes1_FHorz_z(PolicyaltKron,n_daprime,n_a,n_z,N_j,vfoptions);
    end
end

if isNaive
    varargout={V1, Policy, Valt, Policyalt};
else
    varargout={V1, Policy, Valt, []};
end

end
