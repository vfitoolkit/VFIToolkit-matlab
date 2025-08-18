function [V,Policy]=ValueFnIter_FHorz_ExpAssetSemiExo(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d1_grid , d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state

% vfoptions are already set by ValueFnIter_FHorz()
if vfoptions.parallel~=2
    error('Can only use experience asset with parallel=2 (gpu)')
end

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames in same fashion
l_d3=length(n_d3);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d3+l_a2)
    aprimeFnParamNames={temp{l_d3+l_a2+1:end}}; % the first inputs will always be (d3,a2)
else
    aprimeFnParamNames={};
end

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_z=prod(n_z);

% Note: divide-and-conquer is only possible with a1
if N_a1>0 % set up for divide-and-conquer
    if vfoptions.divideandconquer==1
        if ~isfield(vfoptions,'level1n')
            vfoptions.level1n=max(ceil(n_a1(1)/50),5); % minimum of 5
            if n_a1(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
            if vfoptions.verbose==1
                fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
            end
        end
        vfoptions.level1n=min(vfoptions.level1n,n_a1); % Otherwise causes errors
    end
end

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end

if vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_ExpAsset_GI(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
end

if N_d1==0
    d2_gridvals=CreateGridvals(n_d2,d2_grid,1);
else
    d12_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
end


%%
if N_a1==0
    error('Have not implemented experience assets with semi-exogenous shocks, without also having a standard asset')
end

if isfield(vfoptions,'n_e')
    if vfoptions.divideandconquer==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else
    if vfoptions.divideandconquer==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    elseif vfoptions.divideandconquer==1
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d2_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals, d2_grid, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end


%%
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_d1==0 % Policy3
            PolicyKron=shiftdim(PolicyKron(1,:,:,:,:)+N_d2*(PolicyKron(2,:,:,:,:)-1)+N_d2*N_d3*(PolicyKron(3,:,:,:,:)-1),1);
        else % Policy4
            PolicyKron=shiftdim(PolicyKron(1,:,:,:,:)+N_d1*(PolicyKron(2,:,:,:,:)-1)+N_d1*N_d2*(PolicyKron(3,:,:,:,:)-1)+N_d1*N_d2*N_d3*(PolicyKron(4,:,:,:,:)-1),1);
        end
    else
        if N_d1==0 % Policy3
            PolicyKron=shiftdim(PolicyKron(1,:,:,:)+N_d2*(PolicyKron(2,:,:,:)-1)+N_d2*N_d3*(PolicyKron(3,:,:,:)-1),1);
        else % Policy4
            PolicyKron=shiftdim(PolicyKron(1,:,:,:)+N_d1*(PolicyKron(2,:,:,:)-1)+N_d1*N_d2*(PolicyKron(3,:,:,:)-1)+N_d1*N_d2*N_d3*(PolicyKron(4,:,:,:)-1),1);
        end
    end

    if N_z==0
        n_bothz=vfoptions.n_semiz;
    else
        n_bothz=[vfoptions.n_semiz,n_z];
    end
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
    if isfield(vfoptions,'n_e')
        V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_bothz, vfoptions.n_e, N_j, vfoptions);
    else
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_bothz, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end


