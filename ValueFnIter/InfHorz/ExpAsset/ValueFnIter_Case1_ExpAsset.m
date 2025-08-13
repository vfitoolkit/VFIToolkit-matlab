function [V,Policy]=ValueFnIter_Case1_ExpAsset(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

% vfoptions are already set by ValueFnIter_Case1()
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

if isfield(vfoptions,'n_e')
    if N_a1==0
        if N_d1==0
            if N_z==0
                error('Have not yet implemented: InfHorz, no d1, no a1, no z, e,')
            else
                error('Have not yet implemented: InfHorz, no d1, no a1, z, e')
            end
        else
            if N_z==0
                error('Have not yet implemented: InfHorz, d1, no a1, no z, e')
            else
                error('Have not yet implemented: InfHorz, d1, no a1, z, e')
            end
        end
    else % N_a1
        if vfoptions.divideandconquer==0
            if N_d1==0
                if N_z==0
                    error('Have not yet implemented: InfHorz, no d1, a1, no z, e, divideandconquer=0')
                else
                    error('Have not yet implemented: InfHorz, no d1, a1, z, e, divideandconquer=0')
                end
            else
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, e, divideandconquer=0')
                else
                    error('Have not yet implemented: InfHorz, d1, a1, z, e, divideandconquer=0')
                end
            end
        elseif vfoptions.divideandconquer==1
            if N_d1==0
                if N_z==0
                    error('Have not yet implemented: InfHorz, no d1, a1, no z, e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e , d2_grid, a1_grid, a2_grid, vfoptions.e_gridvals, vfoptions.pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                else
                    error('Have not yet implemented: InfHorz, no d1, a1, z, e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, d2_grid, a1_grid, a2_grid, z_gridvals, vfoptions.e_gridvals, pi_z, vfoptions.pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            else % d1 variable
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e , d1_grid, d2_grid, a1_grid, a2_grid, vfoptions.e_gridvals, vfoptions.pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                else
                    error('Have not yet implemented: InfHorz, d1, a1, z, e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, vfoptions.e_gridvals, pi_z, vfoptions.pi_e, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            end
        end
    end
else % no e variable
    if N_a1==0
        if N_d1==0
            if N_z==0
                error('Have not yet implemented: InfHorz, no d1, no a1, no z, no e,')
            else
                error('Have not yet implemented: InfHorz, no d1, no a1, z, no e')
            end
        else
            if N_z==0
                error('Have not yet implemented: InfHorz, d1, no a1, no z, no e')
            else
                error('Have not yet implemented: InfHorz, d1, no a1, z, no e')
            end
        end
    else % N_a1
        if vfoptions.divideandconquer==0
            if N_d1==0
                if N_z==0
                    error('Have not yet implemented: InfHorz, no d1, a1, no z, no e, divideandconquer=0')
                else
                    % error('Have not yet implemented: InfHorz, no d1, a1, z, no e, divideandconquer=0')
                    [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_nod1_raw(V0,n_d2,n_a1,n_a2,n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            else % Use Refine for d1
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, no e, divideandconquer=0')
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_Refine_raw(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            end
        elseif vfoptions.divideandconquer==1
            if N_d1==0
                if N_z==0
                    error('Have not yet implemented: InfHorz, no d1, a1, no z, no e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_nod1_noz_raw(n_d2,n_a1,n_a2 , d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_nod1_raw(V0,n_d2,n_a1,n_a2,n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            else % Use Refine for d1
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, no e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_noz_raw(n_d1,n_d2,n_a1,n_a2 , d1_grid, d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_ExpAsset_DC1_raw(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            end
        end
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