function [V,Policy]=ValueFnIter_InfHorz_InheritAsset(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

if isfield(vfoptions,'aprimeFn')
    aprimeFn=vfoptions.aprimeFn;
else
    error('To use an experience asset you must define vfoptions.aprimeFn')
end

% aprimeFnParamNames in same fashion
l_d2=length(n_d2);
l_z=length(n_z);
temp=getAnonymousFnInputNames(aprimeFn);
if length(temp)>(l_d2+2*l_z)
    aprimeFnParamNames={temp{l_d2+2*l_z+1:end}}; % the first inputs will always be (d2,z,zprime)
else
    aprimeFnParamNames={};
end
aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);

% Note: divide-and-conquer is only possible with a1
if N_a1>0 % set up for divide-and-conquer
    if vfoptions.divideandconquer==1
        if ~isfield(vfoptions,'level1n')
            vfoptions.level1n=max(ceil(n_a1(1)/50),5); % minimum of 5
            if n_a1(1)<5
                error('cannot use vfoptions.divideandconquer=1 with less than 5 points in the a variable (you need to turn off divide-and-conquer, or put more points into the a variable)')
            end
            if vfoptions.verbose==1
                fprintf('Suggestion: When using vfoptions.divideandconquer it will be faster or slower if you set different values of vfoptions.level1n (for smaller models 7 or 9 is good, but for larger models something 15 or 21 can be better) \n')
            end
        end
        vfoptions.level1n=min(vfoptions.level1n,n_a1); % Otherwise causes errors
    end
end

if N_a1>0
    a1_gridvals=CreateGridvals(n_a1,a1_grid,1);
end
if N_d1==0
    d_gridvals=CreateGridvals(n_d2,d2_grid,1);
else
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
end

if isfield(vfoptions,'n_e')
    if N_a1==0
        if N_d1==0
            if N_z==0
                error('Have not yet implemented: InfHorz, inheritanceasset, no d1, no a1, no z, e,')
            else
                error('Have not yet implemented: InfHorz, inheritanceasset, no d1, no a1, z, e')
            end
        else
            if N_z==0
                error('Have not yet implemented: InfHorz, inheritanceasset, d1, no a1, no z, e')
            else
                error('Have not yet implemented: InfHorz, inheritanceasset, d1, no a1, z, e')
            end
        end
    else 
        error('Have not yet implemented: InfHorz, inheritanceasset, a1')
    end
else % no e variable
    if N_a1==0
        if N_d1==0
            if N_z==0
                error('Have not yet implemented: InfHorz, inheritanceasset, no d1, no a1, no z, no e,')
            else
                error('Have not yet implemented: InfHorz, inheritanceasset, no d1, no a1, z, no e')
            end
        else
            if N_z==0 % Use Refine for d1
                error('Have not yet implemented: InfHorz, inheritanceasset, d1, no a1, z, no e')
            else
                [VKron, PolicyKron]=ValueFnIter_InfHorz_InheritAsset_noa1_raw(V0,n_d1,n_d2,n_a2,n_z, d_gridvals, d2_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamsVec, vfoptions);
            end
        end
    else % N_a1
        error('Have not yet implemented: InfHorz, inheritanceasset, a1')
    end
end


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
            V=reshape(VKron,[n_a,vfoptions.n_e]);
            Policy=UnKronPolicyIndexes_Case2(PolicyKron, n_d, n_a, vfoptions.n_e, vfoptions); % Treat e as z (because no z)
        else
            V=reshape(VKron,[n_a,n_z,vfoptions.n_e]);
            Policy=UnKronPolicyIndexes_Case2_e(PolicyKron, n_d, n_a, n_z, vfoptions.n_e, vfoptions);
        end
    else
        if N_z==0
            V=reshape(VKron,[n_a,1]);
            Policy=UnKronPolicyIndexes_Case2_noz(PolicyKron, n_d, n_a, vfoptions);
        else
            V=reshape(VKron,[n_a,n_z]);
            Policy=UnKronPolicyIndexes_Case2(PolicyKron, n_d, n_a, n_z, vfoptions);
        end
    end
else
    V=VKron;
    Policy=PolicyKron;
end



end
