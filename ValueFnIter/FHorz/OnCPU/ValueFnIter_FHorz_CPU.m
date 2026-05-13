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
    if N_z==0
        V=reshape(VKron,[n_a,N_j]);
        Policy=UnKronPolicyIndexes_FHorz_noz_CPU(PolicyKron, n_d, n_a, N_j);
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_FHorz_CPU(PolicyKron, n_d, n_a, n_z, N_j);
    end
else
    V=VKron;
    Policy=PolicyKron;
end

