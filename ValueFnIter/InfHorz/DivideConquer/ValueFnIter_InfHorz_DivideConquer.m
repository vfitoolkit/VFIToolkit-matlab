function [V,Policy]=ValueFnIter_InfHorz_DivideConquer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

% Note: vfoptions.lowmemory is not supported on this (divide-and-conquer) path -- none of the
% DC raws below implement it, so setting it has no effect here.
if ~isfield(vfoptions,'level1n')
    if isscalar(n_a)
        vfoptions.level1n=floor(sqrt(n_a(1)));
    elseif length(n_a)>1
        vfoptions.level1n=[floor(sqrt(n_a(1))),n_a(2:end)]; % default DC2A: level1n(2)==n_a(2) triggers DC2A branch (everything after the first endogenous state is Kron'd together)
    end
end
vfoptions.level1n(1)=min(vfoptions.level1n(1),n_a(1)); % Otherwise causes errors. Only clamp the first, else level1n(2)~=n_a(2) would wrongly route to the DC2 raw

if prod(n_d)==0
    if isscalar(n_a)
        % This is never actually used/reached as it is too slow to be useful
        [V,Policy]=ValueFnIter_InfHorz_DC1_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    elseif length(n_a)>1
        if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the endogenous states after the first
            vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2A
            [V,Policy]=ValueFnIter_InfHorz_DC2A_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        else
            [V,Policy]=ValueFnIter_InfHorz_DC2_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        end
    end
else % N_d
    if isscalar(n_a)
        % This is never actually used/reached as it is too slow to be useful
        [V,Policy]=ValueFnIter_InfHorz_DC1_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    elseif length(n_a)>1
        if vfoptions.level1n(2)==n_a(2) % Don't bother with divide-and-conquer on the endogenous states after the first
            vfoptions.level1n=vfoptions.level1n(1); % Only first one is relevant for DC2A
            [V,Policy]=ValueFnIter_InfHorz_DC2A_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        else
            [V,Policy]=ValueFnIter_InfHorz_DC2_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        end
    end
end

%% UnKron the output
if vfoptions.outputkron==0
    V=reshape(V,[n_a,n_z]);
    if prod(n_d)==0
        Policy=UnKronPolicyIndexes1_z(Policy, n_a, n_a, n_z, vfoptions);
    else
        Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, n_z, vfoptions);
    end
end

end
