function [V,Policy]=ValueFnIter_FHorz_CPU(n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, vfoptions)

N_d=prod(n_d);
% N_a=prod(n_a);
N_z=prod(n_z);

% If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays). This may be completely unnecessary, no-one with a GPU would be using CPU.
d_grid=gather(d_grid);
a_grid=gather(a_grid);
z_grid=gather(z_grid);
pi_z=gather(pi_z);

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end
l_aprime=1; % hardcoded for CPU
l_a=1; % hardcoded for CPU


%% Figure out ReturnFnParamNames from ReturnFn
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_aprime+l_a+l_z) % This is largely pointless, the ReturnFn is always going to have some parameters
    ReturnFnParamNames={temp{l_d+l_aprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z,e)
else
    ReturnFnParamNames={};
end

%% CPU only supports the basics, no options beyond the minimum.
if N_d==0
    if N_z==0
        [VKron,PolicyKron]=ValueFnIter_FHorz_Par1_nod_noz_raw(n_a, N_j, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron,PolicyKron]=ValueFnIter_FHorz_Par1_nod_raw(n_a, n_z, N_j, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
else % N_d
    if N_z==0
        [VKron, PolicyKron]=ValueFnIter_FHorz_Par1_noz_raw(n_d,n_a, N_j, d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    else
        [VKron, PolicyKron]=ValueFnIter_FHorz_Par1_raw(n_d,n_a,n_z, N_j, d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    end
end




%% Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
if vfoptions.outputkron==0
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz_noz(PolicyKron, n_d, n_a, N_j, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z,N_j]);
            Policy=UnKronPolicyIndexes_Case1_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
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
    % First, give some output on the size of any changes in Policy as a result of turning the values into integers
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
elseif vfoptions.policy_forceintegertype==2
    % Do the actual rounding to integers
    Policy=round(Policy);
end

