function [V, Policy]=ValueFnIter_Case1_FHorz_RiskyAsset(n_d,n_a1,n_a2,n_z,n_u,N_j,d_grid, a1_grid,a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_a1=prod(n_a1);
N_z=prod(n_z);

%% Get aprimeFnParamNames
l_d=length(n_d); % because it is a risky asset there must be some decision variables
if isfield(vfoptions,'refine_d')
    l_d=l_d-vfoptions.refine_d(1);
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

if vfoptions.verbose==1
    vfoptions
end

%% Deal with Epstein-Zin preferences if relevant
if isfield(vfoptions,'exoticpreferences')
    if strcmp(vfoptions.exoticpreferences,'None')
        % Just ignore and will then continue on.
    elseif strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        error('Quasi-Hyperbolic preferences with a riskyasset have not been implemented')
    elseif strcmp(vfoptions.exoticpreferences,'EpsteinZin')
        [V, Policy]=ValueFnIter_Case1_FHorz_EpsteinZin_RiskyAsset(n_d,n_a1,n_a2,n_z,n_u,N_j,d_grid,a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, aprimeFn, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        return
    end
end

%% Just do the standard case
if length(n_a2)>1
    error('Have not yet implemented riskyasset for more than one riskyasset')
end
if isfield(vfoptions,'n_e')
    N_e=prod(vfoptions.n_e);
else
    N_e=0;
end

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
    warning('When using vfoptions.riskyasset you should also set vfoptions.refine_d (you do not have refine_d, and this is making codes way slower than they should be)')
    if N_e>0
        if N_a1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_noa1_noz_e_raw(n_d,n_a2,vfoptions.n_e,n_u, N_j, d_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_noa1_e_raw(n_d,n_a2,n_z,vfoptions.n_e,n_u, N_j, d_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_noz_e_raw(n_d,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_e_raw(n_d,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % no e
        if N_a1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_noa1_noz_raw(n_d,n_a2,n_u, N_j, d_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_noa1_raw(n_d,n_a2,n_z,n_u, N_j, d_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_noz_raw(n_d,n_a1,n_a2,n_u, N_j, d_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_RiskyAsset_raw(n_d,n_a1,n_a2,n_z,n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    end
else 
    if sum(vfoptions.refine_d)~=length(n_d)
        error('vfoptions.refine_d seems to be set up wrong, it is inconsistent with n_d')
    end
    if any(vfoptions.refine_d(2:3)==0)
        error('vfoptions.refine_d cannot contain zeros for d2 or d3 (you can do no d1, but you cannot do no d2 nor no d3)')
    end
    n_d1=n_d(1:vfoptions.refine_d(1));
    n_d2=n_d(vfoptions.refine_d(1)+1:vfoptions.refine_d(1)+vfoptions.refine_d(2));
    n_d3=n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1:end);
    d1_grid=d_grid(1:sum(n_d1));
    d2_grid=d_grid(sum(n_d1)+1:sum(n_d1)+sum(n_d2));
    d3_grid=d_grid(sum(n_d1)+sum(n_d2)+1:end);
    if N_e>0
        if N_a1==0
            if N_z==0
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_noz_e_raw(n_d2,n_d3,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_noz_e_raw(n_d1,n_d2,n_d3,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_e_raw(n_d2,n_d3,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_e_raw(n_d1,n_d2,n_d3,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        else % N_a1>0
            if N_z==0
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noz_e_raw(n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noz_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, u_grid, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_e_raw(n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_e_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,vfoptions.n_e,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, u_grid, pi_z_J, vfoptions.pi_e_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        end
    else % N_e==0
        if N_a1==0
            if N_z==0
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_noz_raw(n_d2,n_d3,n_a2,n_u, N_j, d2_grid, d3_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_noz_raw(n_d1,n_d2,n_d3,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noa1_raw(n_d2,n_d3,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noa1_raw(n_d1,n_d2,n_d3,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        else % N_a1>0
            if N_z==0
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_noz_raw(n_d2,n_d3,n_a1,n_a2,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_noz_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, u_grid, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_d1==0
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_nod1_raw(n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_FHorz_RiskyAsset_Refine_raw(n_d1,n_d2,n_d3,n_a1,n_a2,n_z,n_u, N_j, d1_grid, d2_grid, d3_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        end
    end
    % PolicyKron=squeeze(PolicyKron);
end
%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if n_a1(1)==0
    n_a=n_a2;
else
    n_a=[n_a1,n_a2];
    n_d=[n_d,n_a1]; % just to UnKron
end
% Note, Policy has same shape as Case2, so just use that command
if vfoptions.outputkron==0
    if N_e>0
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end

% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    fprintf('USING vfoptions to force integer... \n')
    % First, give some output on the size of any changes in Policy as a
    % result of turning the values into integers
    temp=max(max(max(abs(round(Policy)-Policy))));
    while ndims(temp)>1
        temp=max(temp);
    end
    fprintf('  CHECK: Maximum change when rounding values of Policy is %8.6f (if these are not of numerical rounding error size then something is going wrong) \n', temp)
    % Do the actual rounding to integers
    Policy=round(Policy);
    % Somewhat unrelated, but also do a double-check that Policy is now all positive integers
    temp=min(min(min(Policy)));
    while ndims(temp)>1
        temp=min(temp);
    end
    fprintf('  CHECK: Minimum value of Policy is %8.6f (if this is <=0 then something is wrong) \n', temp)
%     Policy=uint64(Policy);
%     Policy=double(Policy);
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

end