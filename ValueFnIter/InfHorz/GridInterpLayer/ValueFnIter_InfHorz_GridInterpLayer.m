function [V,Policy]=ValueFnIter_InfHorz_GridInterpLayer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

N_d=prod(n_d);

%% Set default vfoptions for the settings that control the multi-grid approaches to GI in InfHorz
% Pre-GI is 'safer' as it will definitely deliver the correct solution.
% Post-GI is much faster, and as long as vfoptions.multigridswitch is not
% too big and vfoptions.maxaprimediff is not to small, it will give the
% correct answer. That said, you can quite easily set
% vfoptions.maxaprimediff too small in a model with a decision variable,
% and so the default value is quite conservative.

if ~isfield(vfoptions,'multigridswitch')
    % vfoptions.multigridswitch determines when to switch from the coarse
    % grid to the fine grid (is based on how close to the solution convergence tolerance we are)
    if vfoptions.preGI==1
        vfoptions.multigridswitch=10000;
        % use a_grid while currdist>multigridswitch*Tolerance then switch to aprime_grid (which includes the interpolation)
    elseif vfoptions.preGI==0
        vfoptions.multigridswitch=10;
        % need to be very close, as then we will only consider +- a few points on the rough grid
    end
end

% postGI looks locally for the optimal grid interpolation point, in the area around the point that was optimal on the coarse grid. 
% By considering the nearest 'maxaprimediff' points on the original coarse grid the hope is that we set the 'local' sweep wide enough to ensure we catch the global optimum. By repeating this process 'postGIrepeat' times, we can hopefully take steps towards the global optimum of the fine grid problem.
if ~isfield(vfoptions,'postGIrepeat')
    vfoptions.postGIrepeat=1; % Do multiple post-GI layers (this is the number of additional layers)
    % In practice, the local optima appears to get stuck in a 'local basin', and so setting postGIrepeat>1 fails to achieve anything because each repeat remains stuck in the basin and does not get any closer to the global.
    % We therefore set a default of postGIrepeat=1. This can be increased but in tests ran there was little to nothing to be gained from doing so in actual applications.
end

% Set the maximum 'rough grid' change in aprime allowed when solving fine problem, in terms of moving from what was optimal when only solving the rough grid problem.
if ~isfield(vfoptions,'maxaprimediff')
    if prod(n_d)==0
        vfoptions.maxaprimediff=5; % only used for postGI (for vfoptions.preGI=0)
    else
        if n_a(1)<300
            vfoptions.maxaprimediff=ceil(n_a(1)/5);
        else
            vfoptions.maxaprimediff=ceil(n_a(1)/10);
        end
        if vfoptions.verbose_advice==1
            warning('vfoptions.postGI=1 is not guaranteed to converge globally. The default of vfoptions.maxaprimediff is set to a conservatively large value to hopefully achieve global convergence. You can try higher/lower values, and see if the solution is sensitive.')
        end
        % Based on testing models, the default vfoptions.maxaprimediff values set here were sufficient to always acheive global convergence. But this does not guarantee they are for all other models.
        % These defaults are also so conservative that the 'post' step takes more memory than the original coarse grid step. The vast majority of models will solve just fine for the global solution with lower values of vfoptions.maxaprimediff, and will be faster and use less memory.
    end
end

% Note: The defaults mean that only four of the following commands get used:
% ValueFnIter_InfHorz_postGI_nod_raw
% ValueFnIter_InfHorz_Refine_postGI_raw
% ValueFnIter_InfHorz_postGI2A_nod_raw
% ValueFnIter_InfHorz_Refine_postGI2A_raw


%% Below Pre-GI and Post-GI both use a multi-grid approach
% I did test a version where I create the full GI grid from the beginning
% and just use this the entire time. It was slower than the Pre-GI multi-grid approach.

%% Deal with the 'without z' case
if prod(n_z)==0
    [V,Policy]=ValueFnIter_InfHorz_GridInterpLayer_noz(V0, n_d, n_a, d_gridvals, a_grid, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
    % Only implement the four default settings for models without Markov
    return
end

%% Use multi-grid approach. Pre-GI
% Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid.
% Precomputes the entirety of aprime_grid (this is what Pre-GI refers to)

if vfoptions.preGI==1 % precompute ReturnMatrixfine
    if isscalar(n_a)
        if N_d==0
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_preGI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            elseif vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_preGI_HowardGreedy_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardsgreedy==2 % howards greedy for a_grid, then howards iter for aprime_grid (greedy is better at smaller grids)
                [V,Policy]=ValueFnIter_InfHorz_preGI_HowardMix_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
            end
        else % N_d
            % Nowadays, I know that only Refine is worth doing, so just skip to that.
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_Refine_preGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            elseif vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_Refine_preGI_HowardGreedy_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardsgreedy==2 % howards greedy for a_grid, then howards iter for aprime_grid (greedy is better at smaller grids)
                [V,Policy]=ValueFnIter_InfHorz_Refine_preGI_HowardMix_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
            end
        end
    else
        if N_d==0
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_preGI2A_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            else
                error('Based on runtimes for the one endogeneous state models with grid interpolation layer, it seems howards greedy is not worthwhile, so did not bother implementing it (you have vfoptoins.howardsgreedy>0)')
            end
        else % N_d
            % Nowadays, I know that only Refine is worth doing, so just skip to that.
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_Refine_preGI2A_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            else
                error('Based on runtimes for the one endogeneous state models with grid interpolation layer, it seems howards greedy is not worthwhile, so did not bother implementing it (you have vfoptoins.howardsgreedy>0)')
            end
        end
    end
end

%% Use multi-grid approach. Post-GI
% Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid.
% Only consider aprime_grid based on +-vfoptions.maxaprimediff (this is the post-GI)
if vfoptions.preGI==0 % solve of rough grid, and then only consider +- a few aprime points (on rough, with all fine interpolation points)
    if isscalar(n_a)
        if N_d==0
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_postGI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    if vfoptions.lowmemory==0
                        error('vfoptions.howardssparse=1 only implemented for vfoptions.lowmemory=1')
                    elseif vfoptions.lowmemory==1
                        [V,Policy]=ValueFnIter_InfHorz_postGI_sparse_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
                    end
                end
            elseif vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_postGI_HowardGreedy_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardsgreedy==2 % howards greedy for a_grid, then howards iter for aprime_grid
                [V,Policy]=ValueFnIter_InfHorz_postGI_HowardMix_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardsgreedy==3 % howards iter for a_grid, then howards greedy for aprime_grid
                [V,Policy]=ValueFnIter_InfHorz_postGI_HowardMix2_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
            end
        else % N_d
            % Nowadays, I know that only Refine is worth doing, so just skip to that.
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_Refine_postGI_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    if vfoptions.lowmemory==0
                        error('vfoptions.howardssparse=1 only implemented for vfoptions.lowmemory=1')
                    elseif vfoptions.lowmemory==1
                        [V,Policy] = ValueFnIter_InfHorz_postGI_sparse_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
                    end
                end
            elseif vfoptions.howardsgreedy==1
                [V,Policy]=ValueFnIter_InfHorz_Refine_postGI_HowardGreedy_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardsgreedy==2 % howards greedy for a_grid, then howards iter for aprime_grid
                [V,Policy]=ValueFnIter_InfHorz_Refine_postGI_HowardMix_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
            elseif vfoptions.howardsgreedy==3 % howards iter for a_grid, then howards greedy for aprime_grid
                [V,Policy]=ValueFnIter_InfHorz_Refine_postGI_HowardMix2_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
            end
        end
    else
        if N_d==0
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_postGI2A_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            else
                error('Based on runtimes for the one endogeneous state models with grid interpolation layer, it seems howards greedy is not worthwhile, so did not bother implementing it (you have vfoptoins.howardsgreedy>0)')
            end
        else % N_d
            % Nowadays, I know that only Refine is worth doing, so just skip to that.
            if vfoptions.howardsgreedy==0
                if vfoptions.howardssparse==0
                    [V,Policy]=ValueFnIter_InfHorz_Refine_postGI2A_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            else
                error('Based on runtimes for the one endogeneous state models with grid interpolation layer, it seems howards greedy is not worthwhile, so did not bother implementing it (you have vfoptoins.howardsgreedy>0)')
            end
        end
    end
end

%%
V=reshape(V,[n_a,n_z]);
if N_d==0
    if isscalar(n_a)
        Policy=UnKronPolicyIndexes1_z(Policy, n_a, n_a, n_z, vfoptions);
    else % grid interp layer on first asset only; postGI2A/preGI2A output [a1prime, a2prime, L2, L2flag]
        Policy=UnKronPolicyIndexes2_z(Policy, n_a(1), n_a(2:end), n_a, n_z, vfoptions);
    end
else
    if isscalar(n_a)
        Policy=UnKronPolicyIndexes2_z(Policy, n_d, n_a, n_a, n_z, vfoptions);
    else % [d, a1prime, a2prime, L2, L2flag]
        Policy=UnKronPolicyIndexes3_z(Policy, n_d, n_a(1), n_a(2:end), n_a, n_z, vfoptions);
    end
end


end
