function [V, Policy]=ValueFnIter_Case1_FHorz_GulPesendorfer(n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% Gul-Pesendorfer

V=nan;
Policy=nan;

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Some Gul-Pesendorfer specific options need to be set if they are not already declared
if ~isfield(vfoptions,'temptationFn')
    error('When using Gul-Pesendorfer preferences you must declare vfoptions.temptationFn (the temptation function)')
end

% Get the temptation function and the parameters needed to evaluate it
% (Note, this is essentially just copy-paste of handling the return fn)
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);
% [n_d,n_a,n_z]
% [l_d,l_a,l_z]
if n_z(1)==0
    l_z=0;
end
if isfield(vfoptions,'SemiExoStateFn')
    l_z=l_z+length(vfoptions.n_semiz);
end
if isfield(vfoptions,'n_e')
    l_e=length(vfoptions.n_e);
else
    l_e=0;
end
if vfoptions.experienceasset==1
    % One of the endogenous states should only be counted once. I fake this by pretending it is a z rather than a variable
    l_z=l_z+1;
    l_a=l_a-1;
end
if vfoptions.residualasset==1
    % One of the endogenous states should only be counted once. I fake this by pretending it is a z rather than a variable
    l_z=l_z+1;
    l_a=l_a-1;
end
% Figure out TemptationFnParamNames from TemptationFn
temp=getAnonymousFnInputNames(vfoptions.temptationFn);
if length(temp)>(l_d+l_a+l_a+l_z+l_e) % This is largely pointless, the temptationFn is always going to have some parameters
    TemptationFnParamNames={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    TemptationFnParamNames={};
end

%% Just do the standard case
if vfoptions.parallel==2
    if N_d==0
        if isfield(vfoptions,'n_e')
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_nod_noz_e_raw(n_a, vfoptions.n_e, N_j, a_grid, vfoptions.e_grid_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_nod_e_raw(n_a, n_z, vfoptions.n_e, N_j, a_grid, z_gridvals_J, vfoptions.e_grid_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_nod_raw(n_a, n_z, N_j, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        end
    else
        if isfield(vfoptions,'n_e') % UP TO HERE
            if N_z==0
                % [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_noz_e_raw(n_d, n_a, vfoptions.n_e, N_j, d_grid, a_grid, vfoptions.e_grid_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_e_raw(n_d, n_a, n_z, vfoptions.n_e, N_j, d_grid, a_grid, z_gridvals_J, vfoptions.e_grid_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron,PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_noz_raw(n_d, n_a, N_j, d_grid, a_grid, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_GulPesendorfer_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, vfoptions.temptationFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, TemptationFnParamNames, vfoptions);
            end
        end
    end
else
    error('Gul-Pesendorfer only implemented for Parallel=2 (gpu)')
end

if vfoptions.outputkron==0
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
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