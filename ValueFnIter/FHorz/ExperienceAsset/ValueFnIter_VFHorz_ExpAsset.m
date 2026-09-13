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
else
    PolicyKron = zeros(N_a1, N_a2, N_z_safe, N_j, 'like', a2_grid);
    n2short = 0;
    n2long  = 0;
    a1prime_grid = [];
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

    a1_work_local = a1_gridvals(:, 1);

    % Define the Unified GPU Tensor Engine for this time period
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_ExpAsset_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, ...
        N_a1, N_a2, N_d1, N_d2, N_z_safe, gridinterplayer, n2short, n2long, ...
        beta_j, V_next, pi_z_j, a1prime_grid, a2primeIndex, a2primeProbs, ...
        a1_work_local, ...
        ReturnFn, D1_cells, d2_gridvals, A1_cells, A2_cells, Z_cells, ReturnFnParamsVec);

    % The Time-Loop Router
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        % Extract the first dimension for the Slicer
        vfoptions.level1n = vfoptions.level1n(1);

        [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = ...
            ValueFnIter_DC1_Slicer(N_a1, N_a1, N_a2, N_z_safe, vfoptions, EvalBlockFn);
    else
        % Route to Brute Force (Standard _raw)
        [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = ...
            EvalBlockFn(1:N_a1, [], 0);
    end

    % Pack PolicyKron (d_idx is now natively tracked by the combo index!)
    d_idx = Pol_d_combo;

    if gridinterplayer
        adjust = (Pol_L2idx_max < 1 + n2short + 1);
        lower_grid_pt = Pol_apr_max - adjust;

        % SAFETY CLAMP: Prevent 0-index for completely invalid (-Inf) states
        lower_grid_pt = max(lower_grid_pt, 1);

        subgrid_step  = adjust .* Pol_L2idx_max + (1 - adjust) .* (Pol_L2idx_max - n2short - 1);

        % Apply toolkit safety clamp to the WINNING choices
        G_segments = n2short + 1;
        at_top = (lower_grid_pt >= N_a1);

        lower_grid_pt(at_top) = N_a1 - 1;
        subgrid_step(at_top)  = G_segments + 1;

        PolicyKron(1, :, :, :, jj) = d_idx;
        PolicyKron(2, :, :, :, jj) = lower_grid_pt;
        PolicyKron(3, :, :, :, jj) = subgrid_step;
        PolicyKron(4, :, :, :, jj) = Pol_L2flag_max;
    else
        % Safety clamp for coarse grid as well
        PolicyKron_j = d_idx + (max(Pol_apr_max, 1) - 1) * (N_d1_safe * N_d2);
        PolicyKron(:, :, :, jj) = PolicyKron_j;
    end

    V(:, :, :, jj) = V_j_max;
    V_next = V_j_max;
end

% =========================================================
% Unpack 4D/5D Structure [n_a1, n_a2, n_z, N_j]
% =========================================================
n_a_vec = [n_a1, n_a2];
if n_d1 > 0 && n_d1(1) > 0
    n_d_vec = [n_d1, n_d2];
else
    n_d_vec = n_d2;
end

if vfoptions.outputkron == 1
    if N_z == 0
        V = reshape(V, [n_a_vec, N_j]);
        if gridinterplayer
            Policy = reshape(PolicyKron, [4, n_a_vec, N_j]);
        else
            Policy = reshape(PolicyKron, [n_a_vec, N_j]);
        end
    else
        V = reshape(V, [n_a_vec, n_z, N_j]);
        if gridinterplayer
            Policy = reshape(PolicyKron, [4, n_a_vec, n_z, N_j]);
        else
            Policy = reshape(PolicyKron, [n_a_vec, n_z, N_j]);
        end
    end
    return
end

if gridinterplayer
    % UnKron2 expects the asset dimensions of PolicyKron to be flattened
    PolicyKron_flat = reshape(PolicyKron, [4, N_a, N_z_safe, N_j]);

    if N_z == 0
        V = reshape(V, [n_a_vec, N_j]);
        Policy = UnKronPolicyIndexes2_FHorz_noz(PolicyKron_flat, n_d_vec, n_a1, n_a_vec, N_j, vfoptions);
    else
        V = reshape(V, [n_a_vec, n_z, N_j]);
        Policy = UnKronPolicyIndexes2_FHorz_z(PolicyKron_flat, n_d_vec, n_a1, n_a_vec, n_z, N_j, vfoptions);
    end
else
    % UnKron1 expects the asset dimensions of PolicyKron to be flattened
    PolicyKron_flat = reshape(PolicyKron, [1, N_a, N_z_safe, N_j]);
    n_d_vec_disc = [n_d_vec, n_a1];

    if N_z == 0
        V = reshape(V, [n_a_vec, N_j]);
        Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron_flat, n_d_vec_disc, n_a_vec, N_j, vfoptions);
    else
        V = reshape(V, [n_a_vec, n_z, N_j]);
        Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron_flat, n_d_vec_disc, n_a_vec, n_z, N_j, vfoptions);
    end
end


end

function [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_ExpAsset_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, ...
    N_a1, N_a2, N_d1, N_d2, N_z_safe, gridinterplayer, n2short, n2long, ...
    beta_j, V_next, pi_z_j, a1prime_grid, a2primeIndex, a2primeProbs, ...
    a1_work_local, ...
    ReturnFn, D1_cells, d2_gridvals, A1_cells, A2_cells, Z_cells, ReturnFnParamsVec)

N_block = length(state_idx);
N_d1_safe = max(1, N_d1);

% Preallocate outputs for this block
V_j_max     = -inf(N_block, N_a2, N_z_safe, 'like', V_next);
Pol_apr_max = ones(N_block, N_a2, N_z_safe, 'like', V_next);
Pol_d1_max  = ones(N_block, N_a2, N_z_safe, 'like', V_next);
Pol_d2_max  = ones(N_block, N_a2, N_z_safe, 'like', V_next);
if gridinterplayer
    Pol_L2idx_max  = ones(N_block, N_a2, N_z_safe, 'like', V_next);
    Pol_L2flag_max = 2 * ones(N_block, N_a2, N_z_safe, 'like', V_next);
else
    Pol_L2idx_max = [];
    Pol_L2flag_max = [];
end

% Slice the A1 cell array so the ReturnFn only broadcasts to the requested block
A1_cells_block = cell(size(A1_cells));
for i = 1:length(A1_cells)
    A1_cells_block{i} = A1_cells{i}(1, 1, state_idx, :);
end

% --- 0. Choice Grid Setup (Implicit Dimensions) ---
if isempty(loweredge_matrix)
    N_choice = N_a1;
    apr_idx_tensor = reshape(1:N_a1, [N_choice, 1, 1, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar);
    N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [N_choice, 1, 1, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, 1, N_a2, N_z_safe]);
    apr_idx_tensor = base_edge + offset;
end
apr_in = reshape(a1_work_local(apr_idx_tensor(:)), size(apr_idx_tensor));

% Pre-calculate tensor offsets for linear indexing
a2_offset = reshape(0:N_a2-1, [1, 1, 1, N_a2, 1]) .* N_a1;
z_offset  = reshape(0:N_z_safe-1, [1, 1, 1, 1, N_z_safe]) .* (N_a1 * N_a2);

for i_d2 = 1:N_d2
    % --- 1. Compute Full Expected Value (EV) for this d2 choice ---
    idx   = a2primeIndex(i_d2, :);
    probs = a2primeProbs(i_d2, :);

    Vlower = V_next(:, idx, :);
    Vupper = V_next(:, min(idx + 1, N_a2), :);

    % Expand probabilities to perfectly match the tensor size!
    % (This completely avoids MATLAB's logical indexing broadcast trap)
    probs_full = repmat(reshape(probs, [1, N_a2, 1]), [N_a1, 1, N_z_safe]);

    % Toolkit exact rule: Skip interpolation if upper and lower are equal
    skipinterp = (Vlower == Vupper);
    probs_full(skipinterp) = 0;

    EV_interp = probs_full .* Vlower + (1 - probs_full) .* Vupper;

    % Apply the toolkit's exact 0 and 1 protections using our full-sized masks
    mask0 = (probs_full == 0);
    mask1 = (probs_full == 1);
    EV_interp(mask0) = Vupper(mask0);
    EV_interp(mask1) = Vlower(mask1);

    % Catch any lingering 0 * -Inf = NaN
    EV_interp(isnan(EV_interp)) = -Inf;

    if N_z_safe > 1
        EV_flat = reshape(EV_interp, [N_a1 * N_a2, N_z_safe]);

        % THE Z SHIELD: Matrix multiplication converts 0 * -Inf to NaN.
        % Temporarily use -1e15 to safely survive the dot product...
        EV_flat(EV_flat == -Inf) = -1e15;
        EV_d2_full = reshape(EV_flat * pi_z_j', [N_a1, N_a2, N_z_safe]);

        % ...then STRICTLY restore anything infected by -1e15 back to -Inf!
        % (This ensures the Grid Interpolator's boundary checks work perfectly)
        EV_d2_full(EV_d2_full <= -1e5) = -Inf;
    else
        EV_d2_full = EV_interp;
    end

    % --- 2. Build Coarse RHS ---
    D2_cells = cell(1, numel(d2_gridvals(i_d2,:)));
    for i = 1:length(D2_cells)
        D2_cells{i} = d2_gridvals(i_d2, i);
    end

    F_tensor = ReturnFn(D1_cells{:}, D2_cells{:}, apr_in, A1_cells_block{:}, A2_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});

    % Extract the exact subset of EV bounds requested by the DC Slicer
    lin_idx = apr_idx_tensor + a2_offset + z_offset;
    EV_d2_bc = reshape(EV_d2_full(lin_idx(:)), size(lin_idx));

    RHS = F_tensor + beta_j .* EV_d2_bc;

    expected_sz = [N_choice, N_d1_safe, N_block, N_a2, N_z_safe];
    if ~isequal(size(RHS), expected_sz)
        RHS = RHS + zeros(expected_sz, 'like', V_next);
    end

    RHS_flat = reshape(RHS, [N_choice * N_d1_safe, N_block * N_a2 * N_z_safe]);
    [V_sub_coarse, Pol_sub_idx_coarse] = max(RHS_flat, [], 1);

    if N_d1 > 0
        apr_idx_local = mod(Pol_sub_idx_coarse - 1, N_choice) + 1;
        d1_idx_coarse = ceil(Pol_sub_idx_coarse / N_choice);
    else
        apr_idx_local = Pol_sub_idx_coarse;
        d1_idx_coarse = ones(size(Pol_sub_idx_coarse), 'like', Pol_sub_idx_coarse);
    end

    if isempty(loweredge_matrix)
        apr_idx_coarse = apr_idx_local;
    else
        loweredge_flat = repmat(reshape(loweredge_matrix, [1, N_a2 * N_z_safe]), [N_block, 1]);
        apr_idx_local_2d = reshape(apr_idx_local, [N_block, N_a2 * N_z_safe]);
        apr_idx_coarse = loweredge_flat + apr_idx_local_2d - 1;
    end
    apr_idx_coarse = reshape(apr_idx_coarse, [N_block, N_a2, N_z_safe]);
    d1_idx_coarse  = reshape(d1_idx_coarse,  [N_block, N_a2, N_z_safe]);

    % --- 3. The Continuous Sub-Grid Refinement ---
    if gridinterplayer
        midpoint = max(min(apr_idx_coarse, N_a1 - 1), 2);
        base_idx = midpoint + (midpoint - 1) * n2short;

        % Build fine indices natively mapped to Dim 1!
        base_idx_tensor = reshape(base_idx, [1, 1, N_block, N_a2, N_z_safe]);
        offset_fine = reshape(-n2short-1 : n2short+1, [n2long, 1, 1, 1, 1]);
        fine_idx_tensor = base_idx_tensor + offset_fine;

        apr_in_fine = reshape(a1prime_grid(fine_idx_tensor(:)), size(fine_idx_tensor));

        if N_d1 > 0
            % Keep d1 fixed at the optimal coarse choice
            D1_fine = cell(size(D1_cells));
            for i_d = 1:numel(D1_cells)
                D1_val = d1_gridvals(d1_idx_coarse(:), i_d);
                D1_fine{i_d} = reshape(D1_val, [1, 1, N_block, N_a2, N_z_safe]);
            end
        else
            D1_fine = {};
        end

        F_tensor_fine = ReturnFn(D1_fine{:}, D2_cells{:}, apr_in_fine, A1_cells_block{:}, A2_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});

        % Vectorized 1D interpolation of EV across all non-sliced states
        % SHIELD: interp1 creates NaNs if it touches -Inf. Use -1e15 temporarily!
        EV_d2_for_interp = EV_d2_full;
        EV_d2_for_interp(EV_d2_for_interp == -Inf) = -1e15;

        % Vectorized 1D interpolation of EV across all non-sliced states
        % SHIELD: interp1 creates NaNs if it touches -Inf. Use -1e15 temporarily!
        EV_d2_for_interp = EV_d2_full;
        EV_d2_for_interp(EV_d2_for_interp == -Inf) = -1e15; 
        
        % BUG FIX: Use the actual coarse asset grid (a1_work_local) as the X-axis!
        EV_d2_interp = interp1(a1_work_local, reshape(EV_d2_for_interp, [N_a1, N_a2 * N_z_safe]), a1prime_grid);

        a2_col = reshape(1:N_a2, [1, 1, 1, N_a2, 1]);
        z_col  = reshape(0:N_z_safe-1, [1, 1, 1, 1, N_z_safe]) .* N_a2;
        col_idx = a2_col + z_col;
        fine_lin_idx = fine_idx_tensor + (col_idx - 1) .* length(a1prime_grid);

        EV_fine = reshape(EV_d2_interp(fine_lin_idx(:)), size(fine_lin_idx));
        RHS_fine = F_tensor_fine + beta_j .* EV_fine;

        expected_sz_fine = [n2long, 1, N_block, N_a2, N_z_safe];
        if ~isequal(size(RHS_fine), expected_sz_fine)
            RHS_fine = RHS_fine + zeros(expected_sz_fine, 'like', V_next);
        end

        RHS_fine_flat = reshape(RHS_fine, [n2long, N_block * N_a2 * N_z_safe]);
        [V_sub_fine, maxindexL2] = max(RHS_fine_flat, [], 1);

        % GI BOUNDARY CHECK: Look for the proxy infinity (-1e10) instead of strict -Inf
        isInfLower = (RHS_fine_flat(1, :) <= -1e10);
        isInfUpper = (RHS_fine_flat(end, :) <= -1e10);

        inLowerStrict = (maxindexL2 >= 2) & (maxindexL2 <= n2short + 1);
        inUpperStrict = (maxindexL2 >= n2short + 3) & (maxindexL2 <= n2long - 1);
        L2flag_fine = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

        V_sub   = reshape(V_sub_fine,     [N_block, N_a2, N_z_safe]);
        apr_idx = reshape(midpoint,       [N_block, N_a2, N_z_safe]);
        d1_idx  = reshape(d1_idx_coarse,  [N_block, N_a2, N_z_safe]);
        L2idx   = reshape(maxindexL2,     [N_block, N_a2, N_z_safe]);
        L2flag  = reshape(L2flag_fine,    [N_block, N_a2, N_z_safe]);
    else
        V_sub   = reshape(V_sub_coarse,   [N_block, N_a2, N_z_safe]);
        apr_idx = reshape(apr_idx_coarse, [N_block, N_a2, N_z_safe]);
        d1_idx  = reshape(d1_idx_coarse,  [N_block, N_a2, N_z_safe]);
    end

    % --- 4. Loop Max Tracking ---
    if i_d2 == 1
        update_mask = true(N_block, N_a2, N_z_safe);
    else
        update_mask = V_sub > V_j_max;
    end

    V_j_max(update_mask)     = V_sub(update_mask);
    Pol_apr_max(update_mask) = apr_idx(update_mask);
    Pol_d1_max(update_mask)  = d1_idx(update_mask);
    Pol_d2_max(update_mask)  = i_d2;

    if gridinterplayer
        Pol_L2idx_max(update_mask)  = L2idx(update_mask);
        Pol_L2flag_max(update_mask) = L2flag(update_mask);
    end
end % End of i_d2 loop

% Pack d1 and d2 into a single combo index so the universal DC1 slicer
% can track both dimensions implicitly!
Pol_d_combo = Pol_d1_max + (Pol_d2_max - 1) * N_d1_safe;

% Restore strict -Inf bounds so the next time period's EV calculates correctly
V_j_max(V_j_max <= -1e10) = -Inf;


end


