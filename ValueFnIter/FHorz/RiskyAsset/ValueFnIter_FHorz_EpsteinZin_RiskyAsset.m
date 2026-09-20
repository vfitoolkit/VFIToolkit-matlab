function [V, Policy]=ValueFnIter_FHorz_EpsteinZin_RiskyAsset(n_d,n_a1,n_a2,n_z,n_u,N_j,d_grid,a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9)

%% Semi-exogenous state: hand off to the semiz variant
if prod(vfoptions.n_semiz)>0
    [V, Policy]=ValueFnIter_FHorz_EpsteinZin_RiskyAsset_semiz(n_d,n_a1,n_a2,vfoptions.n_semiz,n_z,n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.semiz_gridvals_J,z_gridvals_J, u_grid, vfoptions.pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
    return
end
N_a1=prod(n_a1);
N_z=prod(n_z);

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

%% The Epstein-Zin constants (sj, warmglow, ezc1-ezc9) are computed by ValueFnIter_FHorz_EpsteinZin
% (the level-2 preference router) and arrive as inputs; vfoptions.WarmGlowBequestsFnParamsNames
% was set there too (it travels inside vfoptions).


%% Just do the standard case
if length(n_a2)>1
    error('Have not yet implemented riskyasset for more than one riskyasset')
end
N_e=prod(vfoptions.n_e);

if isfield(vfoptions,'refine_d')
    if vfoptions.refine_d(1)==0
        N_d1=0;
    else
        N_d1=prod(n_d(1:vfoptions.refine_d(1)));
    end
    if vfoptions.refine_d(1)==0 && vfoptions.refine_d(2)==0
        % when only d3, refine does nothing anyway, so just turn it off
        vfoptions=rmfield(vfoptions, 'refine_d');
    end
end

if ~isfield(vfoptions,'refine_d')
    error('When using vfoptions.riskyasset you must also set vfoptions.refine_d')
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
    n_d1=0; % scalar zero, NOT empty: the tier dispatchers recompute N_d1=prod(n_d1), and prod([])=1 would misroute nod1 shapes into the with-d1 raws (mirrors ValueFnIter_FHorz_RiskyAsset)
end
if vfoptions.refine_d(2)>0
    n_d2=n_d(vfoptions.refine_d(1)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2));
else
    n_d2=0;
end
if vfoptions.refine_d(3)>0
    n_d3=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1:end);
else
    n_d3=0;
end
d1_grid=d_grid(1:sum(n_d1));
d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);

%% Dispatch to divide-and-conquer and/or grid interpolation layer subfns (mirrors ValueFnIter_FHorz_RiskyAsset)
% Note: DC and GI require an a1 (there is nothing to refine/interpolate without one); the
% tier dispatchers themselves error on N_a1==0, mirroring the vNM tier dispatchers.
if vfoptions.divideandconquer==1 && vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_DC_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
    return
elseif vfoptions.divideandconquer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_DC(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
    return
elseif vfoptions.gridinterplayer==1
    [V,Policy]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_GI(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
    return
end

%%
if N_e==0
    if N_a1==0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noa1_noz_raw(n_d2,n_d3,n_a2,n_u, N_j, d2_grid, d3_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_noz_raw(n_d1,n_d2,n_d3,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        end
    else % N_a1>0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        else
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        end
    end
else % N_e
    if N_a1==0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noa1_noz_e_raw(n_d2,n_d3,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        else
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noa1_e_raw(n_d2,n_d3, n_a2, n_z, vfoptions.n_e, n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noa1_e_raw(n_d1,n_d2,n_d3, n_a2, n_z, vfoptions.n_e, n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        end
    else % N_a1>0
        if N_z==0
            if N_d1==0
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            end
        else
            if N_d1==0
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_nod1_e_raw(n_d2,n_d3, n_a1,n_a2, n_z, vfoptions.n_e, n_u, N_j, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
            else
                [VKron,PolicyKron]=ValueFnIter_FHorz_RiskyAsset_EpsteinZin_e_raw(n_d1,n_d2,n_d3, n_a1,n_a2, n_z, vfoptions.n_e, n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid,a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc1,ezc2,ezc3,ezc4,ezc5,ezc6,ezc7,ezc8,ezc9);
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

if n_a1>0
    n_a=[n_a1,n_a2];
else
    n_a=n_a2;
end

% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if N_a1==0
    if N_d1==0
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron, n_d2,n_d3, n_a, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d2,n_d3, n_a, n_z, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d2,n_d3, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            else
                V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes2_FHorz_z_e(PolicyKron, n_d2,n_d3, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    else % N_d1
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_noz(PolicyKron, n_d1,n_d2,n_d3, n_a, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d1,n_d2,n_d3, n_a, n_z, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d1,n_d2,n_d3, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            else
                V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, n_d1,n_d2,n_d3, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
else % N_a1
    if N_d1==0
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_noz(PolicyKron, n_d2,n_d3,n_a1, n_a, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d2,n_d3,n_a1, n_a, n_z, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z(PolicyKron, n_d2,n_d3,n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            else
                V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, n_d2,n_d3,n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    else % N_d1
        if N_e==0
            if N_z==0
                V=reshape(VKron,[n_a,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_noz(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, N_j, vfoptions);
            else
                V=reshape(VKron,[n_a,n_z,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, n_z, N_j, vfoptions);
            end
        else
            if N_z==0
                V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
            else
                V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
                Policy=UnKronPolicyIndexes4_FHorz_z_e(PolicyKron, n_d1,n_d2,n_d3,n_a1, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
            end
        end
    end
end