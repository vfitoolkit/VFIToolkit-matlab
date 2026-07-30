function [V,Policy]=ValueFnIter_FHorz_ExpAssetSemiExo_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a is endogenous state, a2 is experience asset
% z is exogenous state, semiz is semi-exog state
% vfoptions are already set by ValueFnIter_FHorz()
% Handles vfoptions.divideandconquer==1, vfoptions.gridinterplayer==0

N_d1=prod(n_d1);
N_d2=prod(n_d2);
N_d3=prod(n_d3);
N_a1=prod(n_a1);
N_z=prod(n_z);
N_e=prod(vfoptions.n_e);

%% Divide-and-conquer level1n setup (divide-and-conquer requires the standard endogenous state)
if N_a1==0
    error('Have not implemented experience assets with semi-exogenous shocks, without also having a standard asset')
end
if ~isfield(vfoptions,'level1n')
    vfoptions.level1n=floor(sqrt(n_a1(1)));
    if n_a1(1)<5
        error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
    end
    if vfoptions.verbose==1
        fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a1);

%% DC2A path: two (or more) standard endogenous states -- divide-conquer the first, fold the rest; n_a2 is the experience asset
if length(n_a1)>1
    n_a1DC=n_a1(1);
    n_a1fold=n_a1(2:end);
    N_a1DC=prod(n_a1DC);
    a1DC_grid=a1_gridvals(1:N_a1DC,1);
    a1fold_gridvals=a1_gridvals(1:N_a1DC:end,2:end);
    if length(vfoptions.level1n)>1
        if vfoptions.level1n(2)>=n_a1(2) % only DC on the first standard endogenous state
            vfoptions.level1n=vfoptions.level1n(1);
        else
            error('With ExpAssetSemiExo DC2A, can only do divide-and-conquer on the first standard endogenous state')
        end
    end
    if N_e==0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_nod1_noz_raw(n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_semiz, N_j, d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_noz_raw(n_d1,n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_semiz, N_j, d12_gridvals,d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_nod1_raw(n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_z,n_semiz, N_j, d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, z_gridvals_J,semiz_gridvals_J, pi_z_J,pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_raw(n_d1,n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_z,n_semiz, N_j, d12_gridvals,d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, z_gridvals_J,semiz_gridvals_J, pi_z_J,pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        end
    else % N_e>0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_nod1_noz_e_raw(n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_semiz,vfoptions.n_e, N_j, d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, semiz_gridvals_J,vfoptions.e_gridvals_J, pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_noz_e_raw(n_d1,n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_semiz,vfoptions.n_e, N_j, d12_gridvals,d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, semiz_gridvals_J,vfoptions.e_gridvals_J, pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_nod1_e_raw(n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, z_gridvals_J,semiz_gridvals_J,vfoptions.e_gridvals_J, pi_z_J,pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC2A_e_raw(n_d1,n_d2,n_d3,n_a1DC,n_a1fold,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals,d2_gridvals,d3_grid, a1DC_grid,a1fold_gridvals,a2_grid, z_gridvals_J,semiz_gridvals_J,vfoptions.e_gridvals_J, pi_z_J,pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        end
    end
    % UnKron: reuse existing helpers -- n_a1 is the multi-dim [n_a1DC,n_a1fold], n_bothz=semiz(x z)
    if vfoptions.outputkron==1
        V=VKron; Policy=PolicyKron; return
    end
    if N_z==0
        n_bothz=n_semiz;
    else
        n_bothz=[n_semiz,n_z];
    end
    n_a=[n_a1,n_a2];
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        if N_d1==0
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
        else
            Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        if N_d1==0
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        else
            Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    end
    return
end

%% Dispatch (single DC dim -- existing DC1 path)
if N_e==0
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetSemiExo_DC1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end



%%
if vfoptions.outputkron==1
    V=VKron;
    Policy=PolicyKron;
    return
end

if N_z==0
    n_bothz=n_semiz;
else
    n_bothz=[n_semiz,n_z];
end
if n_a1>0
    n_a=[n_a1,n_a2];
else
    n_a=n_a2;
end

if N_e==0
    V=reshape(VKron,[n_a,n_bothz,N_j]);
    if N_d1==0
        Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
    else
        Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,N_j,vfoptions);
    end
else
    V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
    if N_d1==0
        Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    else
        Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,n_a1,n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
    end
end


end
