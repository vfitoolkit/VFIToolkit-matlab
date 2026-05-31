function [V, Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo(n_d,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d_grid, a1_grid, a2_grid, semiz_gridvals_J,z_gridvals_J, u_grid, pi_semiz_J,pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)


%% Get aprimeFnParamNames
l_d=length(n_d); % because it is a risky asset there must be some decision variables
if isfield(vfoptions,'refine_d')
    l_d=l_d-vfoptions.refine_d(1);
    if length(vfoptions.refine_d)==4 % only relevant if using semiz
        l_d=l_d-vfoptions.refine_d(4);
    end
end
l_u=length(n_u);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d+l_u)
    aprimeFnParamNames={temp{l_d+l_u+1:end}}; % the first inputs will always be (d,u)
else
    aprimeFnParamNames={};
end

%%
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_u=gpuArray(pi_u);
u_grid=gpuArray(u_grid);
% Check pi_u and u_grid are the right size
if all(size(pi_u)==[prod(n_u),1])
    % good
elseif all(size(pi_u)==[1,prod(n_u)])
    error('pi_u should be a column vector (it is a row vector, you need to transpose it')
else
    error('pi_u is the wrong size (it should be a column vector of size prod(n_u)-by-1)')
end
if all(size(u_grid)==[prod(n_u),1])
    % good
elseif all(size(u_grid)==[1,prod(n_u)])
    error('u_grid should be a column vector (it is a row vector, you need to transpose it')
else
    error('u_grid is the wrong size (it should be a column vector of size prod(n_u)-by-1)')
end

%% Setup refine
if length(n_a2)>1
    error('Have not yet implemented riskyasset for more than one riskyasset')
end
if sum(vfoptions.refine_d)~=length(n_d)
    error('vfoptions.refine_d seems to be set up wrong, it is inconsistent with n_d')
end
if any(vfoptions.refine_d(2:3)==0)
    error('vfoptions.refine_d cannot contain zeros for d2 or d3 (you can do no d1, but you cannot do no d2 nor no d3)')
end


if vfoptions.refine_d(1)>0
    n_d1=n_d(1:vfoptions.refine_d(1));
else
    n_d1=0;
end
if vfoptions.refine_d(2)>0
    n_d2=n_d(vfoptions.refine_d(1)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2));
else
    n_d2=0;
end
if vfoptions.refine_d(3)>0
    n_d3=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3));
else
    n_d3=0;
end
if vfoptions.refine_d(4)>0
    n_d4=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3)+1:end);
else
    n_d4=0;
end
d1_grid=d_grid(1:sum(n_d1));
d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:sum(n_d1)+sum(n_d2)+sum(n_d3));
d4_grid=d_grid(sum(n_d1)+sum(n_d2)+sum(n_d3)+1:end);

%% Dispatch
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC_GI(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_DC(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
elseif vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAssetSemiExo_GI(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
    return
end

%% Solve
N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_e=prod(vfoptions.n_e);
N_z=prod(n_z);


if N_e==0
    if N_a1==0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noa1_noz_raw(n_d2,n_d3,n_d4,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_noa1_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noa1_raw(n_d2,n_d3,n_d4,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_noa1_raw(n_d1,n_d2,n_d3,n_d4,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_a1>0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noz_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_noz_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, u_grid, pi_semiz_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else % N_e>0
    if N_a1==0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noa1_noz_e_raw(n_d2,n_d3,n_d4,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noa1_e_raw(n_d2,n_d3,n_d4,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_noa1_e_raw(n_d1,n_d2,n_d3,n_d4,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_a1>0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_noz_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_noz_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_nod1_e_raw(n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAssetSemiExo_e_raw(n_d1,n_d2,n_d3,n_d4,n_a1,n_a2,n_semiz,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, d4_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_semiz_J, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
end


%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_a1==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
end
% Raw Policy is multi-channel: (d1,d2,d3,d4,a1prime), or with d1/a1prime dropped.
% Pick the matching UnKronPolicyIndexes family based on N_d1, N_a1.
if vfoptions.outputkron==0
    if N_d1>0 && N_a1>0
        % 5 channels: (d1, d2, d3, d4, a1prime)
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes5_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    elseif N_d1==0 && N_a1>0
        % 4 channels: (d2, d3, d4, a1prime)
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a1, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    elseif N_d1>0 && N_a1==0
        % 4 channels: (d1, d2, d3, d4)
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a, n_semiz, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a, [n_semiz,n_z], N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d1, n_d2, n_d3, n_d4, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    else % N_d1==0 && N_a1==0
        % 3 channels: (d2, d3, d4)
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a, n_semiz, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d2, n_d3, n_d4, n_a, [n_semiz,n_z], N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,n_semiz,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a, n_semiz, vfoptions.n_e, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_semiz,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, n_d2, n_d3, n_d4, n_a, [n_semiz,n_z], vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end



end
