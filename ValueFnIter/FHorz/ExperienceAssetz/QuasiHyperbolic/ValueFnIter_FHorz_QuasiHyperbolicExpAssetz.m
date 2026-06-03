function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetz(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting with an experienceassetz state.
% Dispatches to Naive/Sophisticated raw files. Same semantics as
% ValueFnIter_FHorz_QuasiHyperbolic but with EV built via aprimeFn for the
% experienceassetz a2 state (mirrors ValueFnIter_FHorz_ExpAssetz).
%
% Phase 1 coverage: baseline only (no DC, no GI, no e). With/without d1.
% Phase 2-4 (GI, DC, DC+GI) and other variants pending.

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

if N_e>0
    error('Phase 1: QuasiHyperbolic+ExpAssetz does not yet support n_e>0; see Phase 5 of the extension plan')
end
if vfoptions.divideandconquer==1 || vfoptions.gridinterplayer==1
    error('Phase 1: QuasiHyperbolic+ExpAssetz baseline only; set vfoptions.divideandconquer=0 and vfoptions.gridinterplayer=0 until Phases 2-4 land')
end

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% Plain baseline dispatch (with z, no e)
if isNaive
    if N_a1==0
        error('Phase 1: noa1 variant not yet implemented')
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzN_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else
    if N_a1==0
        error('Phase 1: noa1 variant not yet implemented')
    elseif N_d1==0
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_nod1_raw(n_d2,n_a1,n_a2,n_z,N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [V1Kron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzS_raw(n_d1,n_d2,n_a1,n_a2,n_z,N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end

%% Unkron
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
        n_daprime=[n_d,n_a1]; % aprime is over a1 only; a2 set by aprimeFn
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
