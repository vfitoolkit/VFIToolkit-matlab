function [V, Policy] = ValueFnIter_VFHorz_ExpAsset(n_d1, n_d2, n_a1, n_a2, n_z, N_j, d1_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
N_a1 = prod(n_a1);
N_a2 = prod(n_a2);
N_a  = N_a1 * N_a2;
N_z  = prod(n_z);

N_d1_safe = max(1, N_d1);
N_z_safe  = max(1, N_z);

if vfoptions.parallel == 2
    d1_gridvals  = gpuArray(d1_gridvals);
    d2_gridvals  = gpuArray(d2_gridvals);
    a1_gridvals  = gpuArray(a1_gridvals);
    a2_grid      = gpuArray(a2_grid);
    z_gridvals_J = gpuArray(z_gridvals_J);
    pi_z_J       = gpuArray(pi_z_J);
end

aprimeFn = vfoptions.aprimeFn;
num_d2 = length(n_d2);
num_a2 = length(n_a2);
temp = getAnonymousFnInputNames(aprimeFn);
if length(temp) > (num_d2 + num_a2 + (num_a2 >= 2))
    aprimeFnParamNames = {temp{num_d2 + num_a2 + (num_a2 >= 2) + 1:end}};
else
    aprimeFnParamNames = {};
end

num_d1 = length(n_d1);
if N_d1 > 0 && n_d1(1) > 0
    D1_cells = cell(1, num_d1);
    for i = 1:num_d1
        D1_cells{i} = shiftdim(d1_gridvals(:, i), -1);
    end
else
    D1_cells = {};
end

num_a1 = length(n_a1);
if N_a1 > 0 && n_a1(1) > 0
    apr_in = shiftdim(a1_gridvals(:, 1), 0);
    A1_cells = cell(1, num_a1);
    for i = 1:num_a1
        A1_cells{i} = shiftdim(a1_gridvals(:, i), -2);
    end
else
    apr_in = gpuArray(0);
    A1_cells = {};
end

a2_gridvals = CreateGridvals(n_a2, a2_grid, 1);
A2_cells = cell(1, num_a2);
for i = 1:num_a2
    A2_cells{i} = shiftdim(a2_gridvals(:, i), -3);
end

V      = zeros(N_a1, N_a2, N_z_safe, N_j, 'like', a2_grid);
V_next = zeros(N_a1, N_a2, N_z_safe, 'like', a2_grid);

gridinterplayer = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;

if gridinterplayer
    n2short = vfoptions.ngridinterp;
    n2long  = n2short * 2 + 3;
    a1prime_grid = interp1(1:1:N_a1, a1_gridvals(:, 1), linspace(1, N_a1, N_a1 + (N_a1 - 1) * n2short))';
    N_a1prime = length(a1prime_grid);
    PolicyKron = zeros(4, N_a1, N_a2, N_z_safe, N_j, 'like', a2_grid);
    
    if N_z > 0
        z_grid_init = z_gridvals_J(:, 1, 1);
    else
        z_grid_init = gpuArray(0);
    end
    [a1_mesh, a2_mesh, z_mesh] = ndgrid(a1_gridvals(:, 1), a2_gridvals(:, 1), z_grid_init);
    N_state = N_a1 * N_a2 * N_z_safe;
    a1_flat = reshape(a1_mesh, [1, N_state]);
    a2_flat = reshape(a2_mesh, [1, N_state]);
    z_flat  = reshape(z_mesh,  [1, N_state]);
    [~, a2_idx_mesh, z_idx_mesh] = ndgrid(1:N_a1, 1:N_a2, 1:N_z_safe);
    a2_idx_flat = reshape(a2_idx_mesh, [1, N_state]);
    z_idx_flat  = reshape(z_idx_mesh,  [1, N_state]);
else
    PolicyKron = zeros(N_a1, N_a2, N_z_safe, N_j, 'like', a2_grid);
end

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    
    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(gpuArray(vfoptions.V_Jplus1), [N_a1, N_a2, N_z_safe]);
    end
    
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    if ~iscell(ReturnFnParamsVec)
        ReturnFnParamsVec = num2cell(ReturnFnParamsVec);
    end
    aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    
    num_z = length(n_z);
    if N_z > 0
        if size(z_gridvals_J, 3) > 1
            z_work_j = z_gridvals_J(:, :, jj);
        else
            z_work_j = z_gridvals_J(:, :, 1);
        end
        Z_cells = cell(1, num_z);
        for i = 1:num_z
            Z_cells{i} = shiftdim(z_work_j(:, i), -4);
        end
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    else
        Z_cells = {};
        pi_z_j = [];
    end
    
    [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec, 2);
    
    % ... (inside reverse_j loop, after EV interpolation setup) ...

    % Define the Unified GPU Tensor Engine for this time period
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_ExpAsset_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, ...
        N_a1, N_a2, N_d1, N_d2, N_z_safe, N_state, gridinterplayer, n2short, n2long, ...
        beta_j, EV_d2, EV_d2_interp, a1prime_grid, a2primeIndex, a2primeProbs, ...
        a1_flat, a2_flat, z_flat, a2_idx_flat, z_idx_flat, ...
        ReturnFn, D1_cells, d2_gridvals, apr_in, A1_cells, A2_cells, Z_cells, ReturnFnParamsVec);

    % The Time-Loop Router
    if isfield(vfoptions, 'level1n') && vfoptions.level1n > 1
        % Route to Universal DC1 Slicer
        [V_j_max, Pol_apr_max, Pol_d1_max, Pol_L2idx_max, Pol_L2flag_max] = ...
            ValueFnIter_DC1_Slicer(N_a1, N_a1, N_a2, N_z_safe, vfoptions, EvalBlockFn);

        % In DC1, we still need the d2 max across the loop. 
        % (For ExpAsset, d2 is handled inside the block, so Pol_d2_max is returned or implied)
        % Note: To keep things perfectly modular, you'll update EvalBlockFn to loop over d2 internally 
        % and return the winning d2 index as well.
    else
        % Route to Brute Force (Standard _raw)
        [V_j_max, Pol_apr_max, Pol_d1_max, Pol_L2idx_max, Pol_L2flag_max, Pol_d2_max] = ...
            EvalBlockFn(1:N_a1, [], 0);
    end

    % Pack PolicyKron
    d_idx = Pol_d1_max + (Pol_d2_max - 1) * N_d1_safe;
    % ... (Proceed to gridinterplayer packing logic as before) ...
    
    if gridinterplayer
        adjust = (Pol_L2idx_max < 1 + n2short + 1);
        lower_grid_pt = Pol_apr_max - adjust;
        subgrid_step  = adjust .* Pol_L2idx_max + (1 - adjust) .* (Pol_L2idx_max - n2short - 1);
        
        PolicyKron(1, :, :, :, jj) = d_idx;
        PolicyKron(2, :, :, :, jj) = lower_grid_pt;
        PolicyKron(3, :, :, :, jj) = subgrid_step;
        PolicyKron(4, :, :, :, jj) = Pol_L2flag_max;
    else
        PolicyKron_j = d_idx + (Pol_apr_max - 1) * (N_d1_safe * N_d2);
        PolicyKron(:, :, :, jj) = PolicyKron_j;
    end
    
    V(:, :, :, jj) = V_j_max;
    V_next = V_j_max;
end

V = reshape(V, [N_a, N_z_safe, N_j]);

if gridinterplayer
    PolicyKron = reshape(PolicyKron, [4, N_a, N_z_safe, N_j]);
else
    PolicyKron = reshape(PolicyKron, [N_a, N_z_safe, N_j]);
end

if vfoptions.outputkron == 1
    Policy = PolicyKron;
else
    if N_d1 > 0 && n_d1(1) > 0
        n_d_vec = [n_d1, n_d2];
    else
        n_d_vec = n_d2;
    end
    
    if gridinterplayer
        n_a_vec = [n_a1, n_a2];
        if N_z == 0
            V = reshape(V, [N_a, N_j]);
            Policy = UnKronPolicyIndexes2_FHorz_noz(PolicyKron, n_d_vec, n_a1, n_a_vec, N_j, vfoptions);
        else
            Policy = UnKronPolicyIndexes2_FHorz_z(PolicyKron, n_d_vec, n_a1, n_a_vec, n_z, N_j, vfoptions);
        end
    else
        PolicyKron = shiftdim(PolicyKron, -1);
        
        if N_a1 > 0 && n_a1(1) > 0
            n_d_vec_disc = [n_d_vec, n_a1];
            n_a_vec_disc = [n_a1, n_a2];
        else
            n_a_vec_disc = n_a2;
        end
        
        if N_z == 0
            V = reshape(V, [N_a, N_j]);
            Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_d_vec_disc, n_a_vec_disc, N_j, vfoptions);
        else
            Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d_vec_disc, n_a_vec_disc, n_z, N_j, vfoptions);
        end
    end
end


end