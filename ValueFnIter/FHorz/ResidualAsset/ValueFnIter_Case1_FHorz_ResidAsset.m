function [V,Policy]=ValueFnIter_Case1_FHorz_ResidAsset(n_d,n_a1,n_r,n_z, N_j, d_grid, a1_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_Case1_FHorz()
if vfoptions.parallel~=2
    error('Can only use experience asset with parallel=2 (gpu)')
end

if isfield(vfoptions,'rprimeFn')
    rprimeFn=vfoptions.rprimeFn;
else
    error('To use an residual asset you must define vfoptions.rprimeFn')
end

if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a1=length(n_a1);
if prod(n_z)==0
    l_z=0;
else
    l_z=length(n_z);
end
% rprimeFnParamNames in same fashion
temp=getAnonymousFnInputNames(rprimeFn);
if length(temp)>(l_d+l_a1+l_a1+l_z)
    rprimeFnParamNames={temp{l_d+l_a1+l_a1+l_z+1:end}}; % the first inputs will always be (d,a1prime,a1,z)
else
    rprimeFnParamNames={};
end

N_z=prod(n_z);

if isfield(vfoptions,'n_e')
    if n_d==0
        if N_z==0
            error('Have not implemented residual assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ResidAsset_nod_e_raw(n_a1,n_r,n_z, vfoptions.n_e, N_j, a1_grid, r_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, rprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, rprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            error('Have not implemented residual assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            % [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ResidAsset_e_raw(n_d,n_a1,n_r,n_z, vfoptions.n_e, N_j, d_grid, a1_grid, r_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, rprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, rprimeFnParamNames, vfoptions);
        end        
    end
else
    if n_d==0
        if N_z==0
            error('Have not implemented residual assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ResidAsset_nod_raw(n_a1,n_r,n_z, N_j, a1_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, rprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, rprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            error('Have not implemented residual assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ResidAsset_raw(n_d,n_a1,n_r,n_z, N_j, d_grid, a1_grid, r_grid, z_gridvals_J, pi_z_J, ReturnFn, rprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, rprimeFnParamNames, vfoptions);
        end
    end
end


%%
if vfoptions.outputkron==0
    if n_d(1)==0
        n_dmod=n_a1;
    else
        n_dmod=[n_d,n_a1];
    end
    n_a=[n_a1,n_r];
    if n_d==0
        PolicyKron=reshape(PolicyKron,[prod(n_a),prod(n_z),N_j]);
    else
        PolicyKron=reshape(PolicyKron,[2,prod(n_a),prod(n_z),N_j]);
        PolicyKron=shiftdim(PolicyKron(1,:,:,:)+prod(n_d)*(PolicyKron(2,:,:,:)-1),1);
    end
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_dmod, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_dmod, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_dmod, n_a, n_z, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end


