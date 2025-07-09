function [V,Policy]=ValueFnIter_DivideConquer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=51;
    elseif length(n_a)==2
        vfoptions.level1n=[21,21];
    else
        error('Cannot use vfoptions.divideandconquer with more than two endogenous states (you have length(n_a)>2)')
    end
end
vfoptions.level1n=min(vfoptions.level1n,n_a);

if prod(n_d)==0
    if isscalar(n_a)
        % This is never actually used/reached as it is too slow to be useful
        [V,Policy]=ValueFnIter_DC1_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    elseif length(n_a)==2
        if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
            vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
            [V,Policy]=ValueFnIter_DC2B_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        else
            [V,Policy]=ValueFnIter_DC2_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        end
    end
else % N_d
    if isscalar(n_a)
        % This is never actually used/reached as it is too slow to be useful
        [V,Policy]=ValueFnIter_DC1_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    elseif length(n_a)==2
        if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the second endogenous state
            vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2B
            [V,Policy]=ValueFnIter_DC2B_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        else
            [V,Policy]=ValueFnIter_DC2_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        end
    end
end



end
