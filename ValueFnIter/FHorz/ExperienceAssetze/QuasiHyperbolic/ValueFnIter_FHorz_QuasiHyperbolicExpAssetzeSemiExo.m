function varargout=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Quasi-hyperbolic discounting with an experienceassetze state (z+e dependent aprimeFn) and semi-exogenous shocks.
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state, e is i.i.d. (required)
% aprimeFn = aprimeFn(d2, a2, z, e, [params])
% Mirrors ValueFnIter_FHorz_ExpAssetzeSemiExo dispatcher, with varargout for the QH Valt/Policyalt.

% vfoptions are already set by ValueFnIter_FHorz()
if vfoptions.parallel~=2
    error('Can only use experience asset with parallel=2 (gpu)')
end

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames: leading inputs are (d2, a2, z, e)
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
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_z==0
    error('Cannot use experienceassetze with no z variables (z is required)')
end
if N_e==0
    error('Cannot use experienceassetze with no e variables (e is required)')
end

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d12_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d12_gridvals=[]; % not used
end

isNaive=strcmp(vfoptions.quasi_hyperbolic,'Naive');

%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    if isNaive
        [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        varargout={V,Policy,Valt,Policyalt};
    else
        [V,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        varargout={V,Policy,Valt,[]};
    end
    return
elseif vfoptions.divideandconquer==1
    if isNaive
        [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        varargout={V,Policy,Valt,Policyalt};
    else
        [V,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        varargout={V,Policy,Valt,[]};
    end
    return
elseif vfoptions.gridinterplayer==1
    if isNaive
        [V,Policy,Valt,Policyalt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        varargout={V,Policy,Valt,Policyalt};
    else
        [V,Policy,Valt]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        varargout={V,Policy,Valt,[]};
    end
    return
end

%% Plain case: no divide-and-conquer, no grid interpolation layer
if N_a1==0
    error('experienceassetze+semiz requires a standard asset a1')
end

if N_d1==0
    if isNaive
        [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoN_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoS_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else
    if isNaive
        [VKron,PolicyKron,ValtKron,PolicyaltKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoN_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron,PolicyKron,ValtKron]=ValueFnIter_FHorz_QuasiHyperbolicExpAssetzeSemiExoS_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end

%% Kron -> unkron
n_bothz=[n_semiz,n_z];
if N_a1>0
    n_a=[n_a1,n_a2];
else
    n_a=n_a2;
end

if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    Valt=ValtKron;
    if isNaive
        Policyalt=PolicyaltKron;
    end
else
    V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    if N_d1==0
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    else
        Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
    Valt=reshape(ValtKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    if isNaive
        if N_d1==0
            Policyalt=UnKronPolicyIndexes3_FHorz_z_e(PolicyaltKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        else
            Policyalt=UnKronPolicyIndexes4_FHorz_z_e(PolicyaltKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    end
end

if isNaive
    varargout={V,Policy,Valt,Policyalt};
else
    varargout={V,Policy,Valt,[]};
end

end
