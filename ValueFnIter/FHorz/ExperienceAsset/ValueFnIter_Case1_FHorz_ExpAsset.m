function [V,Policy]=ValueFnIter_Case1_FHorz_ExpAsset(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
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
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_noa1_noz_e_raw(n_d2,n_a2, vfoptions.n_e, N_j , d2_grid, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_noa1_e_raw(n_d2,n_a2,n_z, vfoptions.n_e, N_j , d2_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else % d1 variable
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_noz_e_raw(n_d1,n_d2,n_a2, vfoptions.n_e, N_j , d1_grid, d2_grid, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_e_raw(n_d1,n_d2,n_a2,n_z, vfoptions.n_e, N_j , d1_grid, d2_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_a1
        if vfoptions.divideandconquer==0
            if N_d1==0
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j , d2_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else % d1 variable
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j , d1_grid, d2_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        elseif vfoptions.divideandconquer==1
            if N_d1==0
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_nod1_noz_e_raw(n_d2,n_a1,n_a2, vfoptions.n_e, N_j , d2_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_nod1_e_raw(n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d2_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else % d1 variable
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_noz_e_raw(n_d1,n_d2,n_a1,n_a2, vfoptions.n_e, N_j , d1_grid, d2_grid, a1_grid, a2_grid, vfoptions.e_gridvals_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_e_raw(n_d1,n_d2,n_a1,n_a2,n_z, vfoptions.n_e, N_j, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals_J, vfoptions.e_gridvals_J, pi_z_J, vfoptions.pi_e_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        end
    end
else % no e variable
    if N_a1==0
        if N_d1==0
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_noa1_noz_raw(n_d2,n_a2, N_j , d2_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_noa1_raw(n_d2,n_a2,n_z, N_j , d2_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        else
            if N_z==0
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_noz_raw(n_d1,n_d2,n_a2, N_j , d1_grid, d2_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            else
                [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_noa1_raw(n_d1,n_d2,n_a2,n_z, N_j , d1_grid, d2_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
            end
        end
    else % N_a1
        if vfoptions.divideandconquer==0
            if N_d1==0
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_noz_raw(n_d2,n_a1,n_a2, N_j , d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_noz_raw(n_d1,n_d2,n_a1,n_a2, N_j , d1_grid, d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_ExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            end
        elseif vfoptions.divideandconquer==1
            if N_d1==0
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_nod1_noz_raw(n_d2,n_a1,n_a2, N_j , d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_nod1_raw(n_d2,n_a1,n_a2,n_z, N_j , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                end
            else
                if N_z==0
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_noz_raw(n_d1,n_d2,n_a1,n_a2, N_j , d1_grid, d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_Case1_FHorz_DC1_ExpAsset_raw(n_d1,n_d2,n_a1,n_a2,n_z, N_j, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
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


end


