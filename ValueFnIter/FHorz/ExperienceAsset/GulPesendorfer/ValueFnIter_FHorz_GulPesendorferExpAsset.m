function [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions)
% Gul-Pesendorfer preferences with an experience asset:
%   V_j = max_{d,a1'} [ u + v + beta*E V_{j+1} ] - max_{d,a1'} v
% where u is the ReturnFn, v the temptation fn (vfoptions.temptationFn, same input signature
% convention as the ReturnFn, own parameters), and the continuation is at
% a2prime=aprimeFn(d2,a2) exactly as in the standard ExpAsset family. The most-tempting term
% is a max of v over the FULL joint choice set ((d1,)d2(,a1prime)) and is subtracted AFTER
% the max in every tier; under grid interpolation it is a max over the FINE a1prime grid,
% found by a two-stage refinement around v's own coarse argmax (see the GP core dispatcher
% ValueFnIter_FHorz_GulPesendorfer for the design decisions). v never touches the
% continuation, so the temptation side needs none of the aprime-probs machinery.
% Dispatched from ValueFnIter_FHorz_GulPesendorfer (which rules out semiz and the other
% experience-asset variants); the DC/GI tier dispatchers rule out multi-dim n_a1.
% vfoptions are already set by ValueFnIter_FHorz()

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2+(l_a2>=2))
    aprimeFnParamNames={temp{l_d2+l_a2+(l_a2>=2)+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
if N_d1>0
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
else
    d_gridvals=[]; % not used
end


%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    % Solve by doing Divide-and-Conquer, and then a grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_DC_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    % Solve using Divide-and-Conquer algorithm
    [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_DC(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    % Solve using grid interpolation layer
    [V,Policy]=ValueFnIter_FHorz_GulPesendorferExpAsset_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals , d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
    return
end


%% Plain case: no divide-and-conquer, no grid interpolation layer
if N_a1==0
    if N_e==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noa1_noz_raw(n_d2,n_a2, N_j, d2_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noa1_raw(n_d2,n_a2,n_z, N_j, d2_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_noa1_noz_raw(n_d1,n_d2,n_a2, N_j, d_gridvals, d2_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_noa1_raw(n_d1,n_d2,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_e>0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noa1_noz_e_raw(n_d2,n_a2, vfoptions.n_e, N_j, d2_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noa1_e_raw(n_d2,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_noa1_noz_e_raw(n_d1,n_d2,n_a2, vfoptions.n_e, N_j , d_gridvals, d2_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_noa1_e_raw(n_d1,n_d2,n_a2,n_z, vfoptions.n_e, N_j , d_gridvals, d2_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else
    %% with a1 (a standard endogenous state)
    if N_e==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noz_raw(n_d2,n_a1,n_a2, N_j, d2_gridvals, a1_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_noz_raw(n_d1,n_d2,n_a1,n_a2, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_e>0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_GulPesendorferExpAsset_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, TemptationFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end


%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if n_d1>0
    n_d=[n_d1,n_d2];
else
    n_d=n_d2;
end
if n_a1>0
    n_d=[n_d,n_a1];
    n_a=[n_a1,n_a2];
else
    % n_d=n_d;
    n_a=n_a2;
end

% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_e==0
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
else
    if N_z==0
        V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
    else
        V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
    end
end


end


