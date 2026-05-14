function [V,Policy]=ValueFnIter_InfHorz_GridInterpLayer(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions)

N_d=prod(n_d);

% Pre-GI is 'safer' as it will definitely deliver the correct solution.
% Post-GI is much faster, and as long as vfoptions.multigridswitch is not
% too big and vfoptions.maxaprimediff is not to small, it will give the
% correct answer.

if ~isfield(vfoptions,'multigridswitch')
    if vfoptions.preGI==1
        vfoptions.multigridswitch=10000;
        % use a_grid while currdist>multigridswitch*Tolerance
        % then switch to aprime_grid (which includes the interpolation)
    elseif vfoptions.preGI==0
        vfoptions.multigridswitch=10;
        % need to be very close, as then we will only consider +- a few points on the rough grid
    end
end

if ~isfield(vfoptions,'postGIrepeat')
    vfoptions.postGIrepeat=1; % Do multiple post-GI layers (this is the number of additional layers)
end

% Set the maximum 'rough grid' change in aprime allowed when solving fine problem, in terms of moving from what was optimal when only solving the rough grid problem.
if ~isfield(vfoptions,'maxaprimediff')
    if n_d(1)==0
        if vfoptions.postGIrepeat==0
            vfoptions.maxaprimediff=5; % only used for postGI (for vfoptions.preGI=0)
        elseif vfoptions.postGIrepeat>0
            vfoptions.maxaprimediff=3; % only used for postGI (for vfoptions.preGI=0)
        end
    else
        if vfoptions.postGIrepeat==0
            vfoptions.maxaprimediff=10; % only used for postGI (for vfoptions.preGI=0)
        elseif vfoptions.postGIrepeat>0
            vfoptions.maxaprimediff=5; % only used for postGI (for vfoptions.preGI=0)
        end
    end
end

% Note: The defaults mean that only four of the following commands get used:
% ValueFnIter_InfHorz_postGI_nod_raw
% ValueFnIter_InfHorz_Refine_postGI_raw
% ValueFnIter_InfHorz_postGI2B_nod_raw
% ValueFnIter_InfHorz_Refine_postGI2B_raw


%% Archived code checked if multi-grid was faster than just going directly to using aprime_grid the whole time. Found that multi-grid is faster.
% if ~isfield(vfoptions,'preinterp')
%     vfoptions.preinterp=1;
%     % =2 is way to slow to be useful
% end
% if isscalar(n_a)
%     if N_d==0
%         if vfoptions.preinterp==1
%             % Multi-grid: only considers a_grid, then when nearing convergence switches to considering aprime_grid.
%             % Precomputes the entirety of aprime_grid.
%             [V,Policy]=ValueFnIter_InfHorz_preGI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
%         elseif vfoptions.preinterp==2
%             % Precomputes the entirety of aprime_grid and just works with this the entire time.
%             % [Multi-grid is better. This was just built for testing/understanding runtimes]
%             [V,Policy]=ValueFnIter_InfHorz_pre2GI_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
%         end
%      end
% end

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
                    [V,Policy]=ValueFnIter_InfHorz_preGI2B_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
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
                    [V,Policy]=ValueFnIter_InfHorz_Refine_preGI2B_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
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
                    [V,Policy]=ValueFnIter_InfHorz_postGI2B_nod_raw(V0, n_a, n_z,  a_grid, z_gridvals, pi_z, DiscountFactorParamsVec, ReturnFn, ReturnFnParamsVec, vfoptions);
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
                    [V,Policy]=ValueFnIter_InfHorz_Refine_postGI2B_raw(V0, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals, pi_z, ReturnFn, DiscountFactorParamsVec, ReturnFnParamsVec, vfoptions);
                elseif vfoptions.howardssparse==1
                    error('Not yet implemented')
                end
            else
                error('Based on runtimes for the one endogeneous state models with grid interpolation layer, it seems howards greedy is not worthwhile, so did not bother implementing it (you have vfoptoins.howardsgreedy>0)')
            end
        end
    end
end


%% Reshape
% Grid interp with 2 endo states leaves them separate, so first we have to
% put them together before we can pass it to UnKronPolicyIndexes [nowadays
% with divide-and-conquer and grid interpolation layer, this is common, so
% maybe I should modify UnKronPolicyIndexes to just permit this as an
% alternative case]
if ~isscalar(n_a)
    if N_d==0
        Policy(1,:,:)=Policy(1,:,:)+n_a(1)*(Policy(2,:,:)-1); % aprime (both)
        Policy(2,:,:)=Policy(3,:,:); % L2index
        Policy=Policy(1:2,:,:);
    else
        % Policy(1,:,:)=Policy(1,:,:); % d
        Policy(2,:,:)=Policy(2,:,:)+n_a(1)*(Policy(3,:,:)-1); % aprime (both)
        Policy(3,:,:)=Policy(4,:,:); % L2index
        Policy=Policy(1:3,:,:);
    end
end

V=reshape(V,[n_a,n_z]);
Policy=UnKronPolicyIndexes_Case1(Policy,n_d,n_a,n_z,vfoptions);


end
