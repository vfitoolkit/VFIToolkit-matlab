function [V,Policy]=ValueFnIter_InfHorz_ExpAsset_GridInterpLayer(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid , d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions)

N_d1=prod(n_d1);
N_a1=prod(n_a1);
N_z=prod(n_z);

error('NOT WORKING YET')


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
    elseif isscalar(n_a1)
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
                else
                    error('Have not yet implemented: InfHorz, no d1, a1, z, e, divideandconquer=1')
                end
            else % d1 variable
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, e, divideandconquer=1')
                else
                    error('Have not yet implemented: InfHorz, d1, a1, z, e, divideandconquer=1')
                end
            end
        end
    else
        error('Have not yet implemented grid interpolation layer in model with two standard endo states and one experienceasset')
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
    elseif isscalar(n_a1)
        if vfoptions.divideandconquer==0
            if N_d1==0
                if N_z==0
                    error('Have not yet implemented: InfHorz, no d1, a1, no z, no e, divideandconquer=0')
                else
                    % error('Have not yet implemented: InfHorz, no d1, a1, z, no e, divideandconquer=0')
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_ExpAsset_GI_nod1_raw(V0,n_d2,n_a1,n_a2,n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            else % Use Refine for d1
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, no e, divideandconquer=0')
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_ExpAsset_Refine_GI_raw(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            end
        elseif vfoptions.divideandconquer==1
            if N_d1==0
                if N_z==0
                    error('Have not yet implemented: InfHorz, no d1, a1, no z, no e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_InfHorz_ExpAsset_DC1_nod1_noz_raw(n_d2,n_a1,n_a2 , d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_ExpAsset_DC1_GI_nod1_raw(V0,n_d2,n_a1,n_a2,n_z, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            else % Use Refine for d1
                if N_z==0
                    error('Have not yet implemented: InfHorz, d1, a1, no z, no e, divideandconquer=1')
                    % [VKron, PolicyKron]=ValueFnIter_InfHorz_ExpAsset_DC1_noz_raw(n_d1,n_d2,n_a1,n_a2 , d1_grid, d2_grid, a1_grid, a2_grid, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                else
                    [VKron, PolicyKron]=ValueFnIter_InfHorz_ExpAsset_DC1_GI_raw(V0,n_d1,n_d2,n_a1,n_a2,n_z, d1_grid, d2_grid, a1_grid, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamsVec, ReturnFnParamsVec, aprimeFnParamNames, vfoptions);
                end
            end
        end
    else
        error('Have not yet implemented grid interpolation layer in model with two standard endo states and one experienceasset')
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
