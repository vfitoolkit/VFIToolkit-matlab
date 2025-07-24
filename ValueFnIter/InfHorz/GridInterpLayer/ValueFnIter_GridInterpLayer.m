function [V,Policy]=ValueFnIter_GridInterpLayer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

N_d=prod(n_d);

if ~isfield(vfoptions,'multigridswitch')
    vfoptions.multigridswitch=10000;
    % use a_grid while currdist>multigridswitch*Tolerance
    % then switch to aprime_grid (which includes the interpolation)
end

%% Archived code that tried different approaches to multi-grid
% if ~isfield(vfoptions,'preinterp')
%     vfoptions.preinterp=1; 
%     % =2 is way to slow to be useful
% end
% if isscalar(n_a)
%     if N_d==0
%         if vfoptions.preinterp==0
%             % Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid.
%             % Computes only the parts of ReturnFn for aprime_grid as and when needed.
%             % Turned-out to be not worthwhile
%             [V,Policy]=ValueFnIter_GI_nod_raw(V0, n_a, n_z, a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
%         elseif vfoptions.preinterp==1
%             % Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid. 
%             % Precomputes the entirety of aprime_grid.
%             [V,Policy]=ValueFnIter_preGI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
%         elseif vfoptions.preinterp==2
%             % Precomputes the entirety of aprime_grid and just works with this the entire time. 
%             % [Multi-grid is better. This was just built for testing/understanding runtimes]
%             [V,Policy]=ValueFnIter_pre2GI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
%         end
%     else % N_d
%         % Nowadays, I know that only Refine is worth doing, so just skip to that.
%         if vfoptions.preinterp==1
%             % Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid. 
%             % Precomputes the entirety of aprime_grid.
%             [V,Policy]=ValueFnIter_Refine_preGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
%         end
%     end
% end

%% Use multi-grid approach.
% Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid.
% Precomputes the entirety of aprime_grid.

if isscalar(n_a)
    if N_d==0
        if vfoptions.howardsgreedy==0
            [V,Policy]=ValueFnIter_preGI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
        elseif vfoptions.howardsgreedy==1
            [V,Policy]=ValueFnIter_preGI_HowardGreedy_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
        elseif vfoptions.howardsgreedy==2 % howards iter for a_grid, then howards greedy for aprime_grid
            [V,Policy]=ValueFnIter_preGI_HowardMix_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
        end
    else % N_d
        % Nowadays, I know that only Refine is worth doing, so just skip to that.
        if vfoptions.howardsgreedy==0
            [V,Policy]=ValueFnIter_Refine_preGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        elseif vfoptions.howardsgreedy==1
            [V,Policy]=ValueFnIter_Refine_preGI_HowardGreedy_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        elseif vfoptions.howardsgreedy==2
            [V,Policy]=ValueFnIter_Refine_preGI_HowardMix_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
        end
    end
else
    error('vfoptions.gridinterplayer in Infinite horizion does not yet support two endogenous states')
end

% Note: GI and preGI give slightly different Policy because they deal
% differently with the 'lower grid point' vs 'L2' indexes. But if you
% compare them on the 'fine' grid (for model with no d variable)
% tempA=Policy2a(1,:,:)+((Policy2a(1,:,:)-1)*20)+Policy2a(2,:,:);
% tempC=Policy2c(1,:,:)+((Policy2c(1,:,:)-1)*20)+Policy2c(2,:,:);
% max(abs(tempA(:)-tempC(:)))
% You can see they are exactly the same policy

end
