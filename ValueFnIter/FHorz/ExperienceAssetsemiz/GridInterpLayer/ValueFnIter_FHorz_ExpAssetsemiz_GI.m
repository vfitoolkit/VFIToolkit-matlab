function [V,Policy]=ValueFnIter_FHorz_ExpAssetsemiz_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals , d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)
% experienceassetsemiz: the experience asset is driven by the semi-exogenous state.
% d1 is any other decision, d2 determines experience asset, d3 determines semi-exog state
% a1 is standard endogenous state (required), a2 is experience asset
% z is an OPTIONAL ordinary exogenous markov state, semiz is the semi-exog state (required)
% Handles vfoptions.gridinterplayer==1, vfoptions.divideandconquer==0
%
% NOTE (incremental build): the noe raws are implemented; the _e raws arrive in a later commit.

N_d1=prod(n_d1);
N_z=prod(n_z);
N_a1=prod(n_a1);
N_e=prod(vfoptions.n_e);

if N_a1==0
    error('experienceassetsemiz with noa1 has no standard endogenous asset to divide-and-conquer or interpolate (turn off vfoptions.divideandconquer and vfoptions.gridinterplayer)')
end
%% GI2A path: two (or more) standard endogenous states. a is divided into a1 (first standard endogenous state, grid interpolation layer), a2 (remaining standard endogenous states, folded) and a3 (experience asset)
% Mirrors the DC2A block in ValueFnIter_FHorz_ExpAssetsemiz_DC; GI2A does not use level1n.
if length(n_a1)>1
    % a1_gridvals holds all the standard endogenous states (rows cycle the first fastest); n_a2/a2_grid hold the experience asset
    n_a3=n_a2;
    a3_grid=a2_grid;
    a2_gridvals=a1_gridvals(1:n_a1(1):end,2:end);
    a1_grid=a1_gridvals(1:n_a1(1),1);
    n_a2=n_a1(2:end);
    n_a1=n_a1(1);
    if N_e==0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_semiz, N_j, d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_semiz, N_j, d12_gridvals,d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz, N_j, d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, z_gridvals_J,semiz_gridvals_J, pi_z_J,pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz, N_j, d12_gridvals,d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, z_gridvals_J,semiz_gridvals_J, pi_z_J,pi_semiz_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        end
    else
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_semiz,vfoptions.n_e, N_j, d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, semiz_gridvals_J,vfoptions.e_gridvals_J, pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_semiz,vfoptions.n_e, N_j, d12_gridvals,d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, semiz_gridvals_J,vfoptions.e_gridvals_J, pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, z_gridvals_J,semiz_gridvals_J,vfoptions.e_gridvals_J, pi_z_J,pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI2A_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_a3,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals,d2_gridvals,d3_grid, a1_grid,a2_gridvals,a3_grid, z_gridvals_J,semiz_gridvals_J,vfoptions.e_gridvals_J, pi_z_J,pi_semiz_J,vfoptions.pi_e_J, ReturnFn,aprimeFn,Parameters,DiscountFactorParamNames,ReturnFnParamNames,aprimeFnParamNames,vfoptions);
            end
        end
    end
    % UnKron: reuse existing helpers -- aprime is Kron over [n_a1,n_a2], n_bothz=semiz(x z)
    if vfoptions.outputkron==1
        V=VKron; Policy=PolicyKron; return
    end
    if N_z==0
        n_bothz=n_semiz;
    else
        n_bothz=[n_semiz,n_z];
    end
    n_a=[n_a1,n_a2,n_a3];
    if N_e==0
        V=reshape(VKron,[n_a,n_bothz,N_j]);
        if N_d1==0
            Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron,n_d2,n_d3,[n_a1,n_a2],n_a,n_bothz,N_j,vfoptions);
        else
            Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron,n_d1,n_d2,n_d3,[n_a1,n_a2],n_a,n_bothz,N_j,vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_bothz,vfoptions.n_e,N_j]);
        if N_d1==0
            Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron,n_d2,n_d3,[n_a1,n_a2],n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        else
            Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron,n_d1,n_d2,n_d3,[n_a1,n_a2],n_a,n_bothz,vfoptions.n_e,N_j,vfoptions);
        end
    end
    return
end

%% Dispatch
if N_e==0
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
else % N_e
    if N_d1==0
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        else
            [VKron, PolicyKron]=ValueFnIter_FHorz_ExpAssetsemiz_GI1_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_semiz,vfoptions.n_e, N_j, d12_gridvals, d2_gridvals, d3_grid, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, pi_semiz_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
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


end
