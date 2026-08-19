function [V,Policy]=ValueFnIter_InfHorz_GridInterpLayer_noz(V0, n_d, n_a, d_gridvals, a_grid, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

N_d=prod(n_d);

% Only implement the four default settings for models without Markov
% ValueFnIter_InfHorz_postGI_nod_noz_raw
% ValueFnIter_InfHorz_Refine_postGI_noz_raw
% ValueFnIter_InfHorz_postGI2A_nod_noz_raw
% ValueFnIter_InfHorz_Refine_postGI2A_noz_raw

%% Use multi-grid approach. Post-GI
% Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid.
% Only consider aprime_grid based on +-vfoptions.maxaprimediff (this is the post-GI)
if vfoptions.preGI==0 % solve of rough grid, and then only consider +- a few aprime points (on rough, with all fine interpolation points)
    if isscalar(n_a)
        if N_d==0
            [V,Policy]=ValueFnIter_InfHorz_postGI_nod_noz_raw(V0, n_a,  a_grid, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
        else % N_d
            [V,Policy]=ValueFnIter_InfHorz_Refine_postGI_noz_raw(V0, n_d, n_a, d_gridvals, a_grid, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        end
    else
        error('NOT YET implemented')
        % if N_d==0
        %     [V,Policy]=ValueFnIter_InfHorz_postGI2A_nod_noz_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
        % else % N_d
        %     [V,Policy]=ValueFnIter_InfHorz_Refine_postGI2A_noz_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        % end
    end
end

%%
V=reshape(V,[n_a,1]);
if N_d==0
    if isscalar(n_a)
        Policy=UnKronPolicyIndexes1_noz(Policy, n_a, n_a, vfoptions);
    else % grid interp layer on first asset only; postGI2A/preGI2A output [a1prime, a2prime, L2, L2flag]
        Policy=UnKronPolicyIndexes2_noz(Policy, n_a(1), n_a(2:end), n_a, vfoptions);
    end
else
    if isscalar(n_a)
        Policy=UnKronPolicyIndexes2_noz(Policy, n_d, n_a, n_a, vfoptions);
    else % [d, a1prime, a2prime, L2, L2flag]
        Policy=UnKronPolicyIndexes3_noz(Policy, n_d, n_a(1), n_a(2:end), n_a, vfoptions);
    end
end


end
