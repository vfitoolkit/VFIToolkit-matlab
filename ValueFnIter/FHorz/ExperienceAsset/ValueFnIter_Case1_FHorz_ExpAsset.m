function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
% vfoptions are already set by ValueFnIter_Case1_FHorz()
if vfoptions.parallel~=2
    error('Can only use experience asset with parallel=2 (gpu)')
end

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_a2=length(n_a2);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+l_a2)
    aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
else
    aprimeFnParamNames={};
end

N_z=prod(n_z);

if isfield(vfoptions,'n_e')
    if isfield(vfoptions,'e_grid_J')
        e_grid=vfoptions.e_grid_J(:,1); % Just a placeholder
    else
        e_grid=vfoptions.e_grid;
    end
    if isfield(vfoptions,'pi_e_J')
        pi_e=vfoptions.pi_e_J(:,1); % Just a placeholder
    else
        pi_e=vfoptions.pi_e;
    end
    if n_d1==0
        if N_z==0
            error('Have not implemented experience assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_e_raw(n_d2,n_a1,n_a2,n_z,  vfoptions.n_e, N_j, d2_grid, a1_grid, a2_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            error('Have not implemented experience assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_grid, e_grid, pi_z, pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end        
    end
else
    if n_d1==0
        if N_z==0
            error('Have not implemented experience assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j , d2_grid, a1_grid, a2_grid, z_grid, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    else
        if N_z==0
            error('Have not implemented experience assets without at least one exogenous variable [you could fake it adding a single-valued z with pi_z=1]')
        else
            [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_grid, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        end
    end
end

size(PolicyKron)
disp('MIN')
min(PolicyKron(:))

%%
if vfoptions.outputkron==0
    if n_d1>0
        n_d=[n_d1,n_d2];
    else 
        n_d=n_d2;
    end
    if n_a1>0
        n_a=[n_a1,n_a2];
        n_d=[n_d,n_a1];
    else
        n_a=n_a2;
    end
    %Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
    if isfield(vfoptions,'n_e')
        if N_z==0
            V=reshape(VKron,[n_a,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, vfoptions.n_e, N_j, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e,N_j]);
            Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, N_j, vfoptions);
        end
    else
        V=reshape(VKron,[n_a,n_z,N_j]);
        Policy=UnKronPolicyIndexes_Case2_FHorz(PolicyKron, n_d, n_a, n_z, N_j, vfoptions);
    end
else
    V=VKron;
    Policy=PolicyKron;
end


end


