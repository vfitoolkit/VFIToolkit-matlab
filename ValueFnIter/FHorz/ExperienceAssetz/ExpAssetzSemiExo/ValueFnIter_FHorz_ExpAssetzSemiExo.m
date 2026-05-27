function [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state, a2 is experience asset
% z is exogenous markov state (required), semiz is semi-exog state
% aprimeFn = aprimeFn(d2, a2, z, [params])

% vfoptions are already set by ValueFnIter_FHorz()
if vfoptions.parallel~=2
    error('Can only use experience asset with parallel=2 (gpu)')
end

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames: leading inputs are (d2, a2, z)
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
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

if N_z==0
    error('Cannot use experienceassetz with no z variables (z is required)')
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


%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_ExpAssetzSemiExo_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
end


%% Plain case: no divide-and-conquer, no grid interpolation layer
if N_a1==0
    error('Have not implemented experience assets with semi-exogenous shocks, without also having a standard asset')
end

if N_e==0
    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
else % N_e
    if N_d1==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetzSemiExo_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    end
end


%%
if vfoptions.outputkron==0
    if N_e==0
        if N_d1==0 % Policy3
            PolicyKron=shiftdim(PolicyKron(1,:,:,:)+N_d2*(PolicyKron(2,:,:,:)-1)+N_d2*N_d3*(PolicyKron(3,:,:,:)-1),1);
        else % Policy4
            PolicyKron=shiftdim(PolicyKron(1,:,:,:)+N_d1*(PolicyKron(2,:,:,:)-1)+N_d1*N_d2*(PolicyKron(3,:,:,:)-1)+N_d1*N_d2*N_d3*(PolicyKron(4,:,:,:)-1),1);
        end
    else
        if N_d1==0 % Policy3
            PolicyKron=shiftdim(PolicyKron(1,:,:,:,:)+N_d2*(PolicyKron(2,:,:,:,:)-1)+N_d2*N_d3*(PolicyKron(3,:,:,:,:)-1),1);
        else % Policy4
            PolicyKron=shiftdim(PolicyKron(1,:,:,:,:)+N_d1*(PolicyKron(2,:,:,:,:)-1)+N_d1*N_d2*(PolicyKron(3,:,:,:,:)-1)+N_d1*N_d2*N_d3*(PolicyKron(4,:,:,:,:)-1),1);
        end
    end

    n_bothz=[vfoptions.n_semiz,n_z];
    if N_d1>0
        n_d=[n_d1,n_d2,n_d3];
    else
        n_d=[n_d2,n_d3];
    end
    if n_a1>0
        n_a=[n_a1,n_a2];
        n_d=[n_d,n_a1];
    else
        n_a=n_a2;
    end

    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_bothz, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_bothz, vfoptions.n_e, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end
