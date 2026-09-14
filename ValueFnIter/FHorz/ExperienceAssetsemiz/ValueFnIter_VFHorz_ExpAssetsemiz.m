function [V, Policy] = ValueFnIter_VFHorz_ExpAssetsemiz(n_d1, n_d2, n_d3, n_a1, n_a2, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, d3_gridvals, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Safe Dimension Setup ---
N_d1 = max(1, prod(n_d1)); N_d2 = max(1, prod(n_d2)); N_d3 = max(1, prod(n_d3));
N_a1 = max(1, prod(n_a1)); N_a2 = max(1, prod(n_a2));
N_z = prod(n_z); N_z_safe = max(1, N_z); 
N_semiz_safe = max(1, prod(n_semiz));

% --- 2. GPU Transfer ---
if vfoptions.parallel == 2
    d1_gridvals = gpuArray(d1_gridvals); d2_gridvals = gpuArray(d2_gridvals); d3_gridvals = gpuArray(d3_gridvals);
    a1_gridvals = gpuArray(a1_gridvals); a2_grid = gpuArray(a2_grid);
    z_gridvals_J = gpuArray(z_gridvals_J); semiz_gridvals_J = gpuArray(semiz_gridvals_J);
    pi_z_J = gpuArray(pi_z_J); pi_semiz_J = gpuArray(pi_semiz_J);
end

% --- 3. EZ Context ---
is_EZ = isfield(vfoptions, 'exoticpreferences') && strcmp(vfoptions.exoticpreferences, 'EpsteinZin');
if is_EZ
    ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
    ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;
else
    ezc2 = ones(N_j,1); ezc3 = 1; ezc4 = 1; ezc5 = ones(N_j,1);
    ezc6 = ones(N_j,1); ezc7 = ones(N_j,1); ezc8 = ones(N_j,1);
end

% --- 4. Setup Grids and Functions ---
aprimeFn = vfoptions.aprimeFn;
num_d2 = length(n_d2); num_a2 = length(n_a2); num_semiz = length(n_semiz);
input_names = getAnonymousFnInputNames(aprimeFn);
aprimeFnParamNames = input_names(isfield(Parameters, input_names));

num_d1 = length(n_d1);
if N_d1 > 0 && n_d1(1) > 0
    D1_cells = cell(1, num_d1);
    for i = 1:num_d1, D1_cells{i} = shiftdim(d1_gridvals(:, i), -1); end
else
    D1_cells = {};
end

num_a1 = length(n_a1);
A1_cells = cell(1, num_a1);
for i = 1:num_a1, A1_cells{i} = shiftdim(a1_gridvals(:, i), -2); end
a1_work_local = a1_gridvals;

a2_gridvals_full = CreateGridvals(n_a2, a2_grid, 1);
A2_cells = cell(1, num_a2);
for i = 1:num_a2, A2_cells{i} = shiftdim(a2_gridvals_full(:, i), -3); end

Semiz_cells = cell(1, num_semiz);
for i = 1:num_semiz, Semiz_cells{i} = shiftdim(semiz_gridvals_J(:, i, 1), -4); end

num_z = length(n_z);
if N_z_safe > 1
    Z_cells = cell(1, num_z);
    for i = 1:num_z, Z_cells{i} = shiftdim(z_gridvals_J(:, i, 1), -5); end
else
    Z_cells = {};
end

V = zeros(N_a1, N_a2, N_semiz_safe, N_z_safe, N_j, 'like', a2_grid);
V_next = zeros(N_a1, N_a2, N_semiz_safe, N_z_safe, 'like', a2_grid);
PolicyKron = zeros(N_a1, N_a2, N_semiz_safe, N_z_safe, N_j, 'like', a2_grid);

gridinterplayer = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer(1) == 1;
if gridinterplayer
    error('Grid interpolation not yet supported in V-World ExpAssetsemiz.');
end

% =========================================================
% TIME LOOP
% =========================================================
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    if ~iscell(ReturnFnParamsVec), ReturnFnParamsVec = num2cell(ReturnFnParamsVec); end

    aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    [a2primeIndex, a2primeProbs] = CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec, 2);

    if N_z_safe > 1
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    else
        pi_z_j = [];
    end
    pi_semiz_j = pi_semiz_J(:, :, :, min(jj, size(pi_semiz_J, 4)));

    % 2. Apply EZ Curvature
    valid_V = isfinite(V_next) & (V_next ~= 0);
    V_transformed = V_next;
    if ezc5(jj) == 1
        V_transformed(valid_V) = ezc4 * V_next(valid_V);
    else
        V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj);
    end
    V_transformed(V_next == 0) = 0;

    % 3. Pre-integrate purely exogenous Z
    if prod(n_z) > 0
        V_flat = reshape(V_transformed, [N_a1 * N_a2 * N_semiz_safe, N_z_safe]);
        EV_z_pre = reshape(V_flat * pi_z_j', [N_a1, N_a2, N_semiz_safe, N_z_safe]);
    else
        EV_z_pre = V_transformed;
    end

    % 4. Call the Tensor Block
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_ExpAssetSemiZ_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d1, N_d2, N_d3, N_semiz_safe, N_z_safe, num_a1, ...
        beta_j, EV_z_pre, pi_semiz_j, a2primeIndex, a2primeProbs, a1_work_local, ...
        ReturnFn, D1_cells, d2_gridvals, d3_gridvals, A1_cells, A2_cells, Z_cells, Semiz_cells, ReturnFnParamsVec, ...
        ezc2(jj), ezc3, ezc4, ezc6(jj), ezc7(jj), ezc8(jj));

    % 5. Route (DC Slicer vs Brute Force)
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        vfoptions.level1n = vfoptions.level1n(1);
        [V_j_max, Pol_apr_max, Pol_d_combo] = ...
            ValueFnIter_DC1_Slicer(N_a1, N_a1, N_a2, N_semiz_safe * N_z_safe, vfoptions, EvalBlockFn);
    else
        [V_j_max, Pol_apr_max, Pol_d_combo] = EvalBlockFn(1:N_a1, [], 0);
    end

    V(:,:,:,:,jj) = V_j_max;
    V_next = V_j_max;

    PolicyKron_j = Pol_d_combo + (max(Pol_apr_max, 1) - 1) * (N_d1 * N_d2 * N_d3);
    PolicyKron(:, :, :, :, jj) = PolicyKron_j;
end

% =========================================================
% OUTPUT UNPACKING
% =========================================================
n_a_vec = [n_a1, n_a2];
if N_d1 > 0 && n_d1(1) > 0
    n_d_vec = [n_d1, n_d2, n_d3];
else
    n_d_vec = [n_d2, n_d3];
end

if vfoptions.outputkron == 1
    if N_z == 0
        V = reshape(V, [n_a_vec, n_semiz, N_j]);
        Policy = reshape(PolicyKron, [n_a_vec, n_semiz, N_j]);
    else
        V = reshape(V, [n_a_vec, n_semiz, n_z, N_j]);
        Policy = reshape(PolicyKron, [n_a_vec, n_semiz, n_z, N_j]);
    end
    return
end

PolicyKron_flat = reshape(PolicyKron, [1, N_a1 * N_a2, N_semiz_safe * N_z_safe, N_j]);
n_d_vec_disc = [n_d_vec, n_a1];
if N_z == 0
    V = reshape(V, [n_a_vec, n_semiz, N_j]);
    Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron_flat, n_d_vec_disc, n_a_vec, n_semiz, N_j, vfoptions);
else
    V = reshape(V, [n_a_vec, n_semiz, n_z, N_j]);
    n_bothz = [n_semiz, n_z];
    Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron_flat, n_d_vec_disc, n_a_vec, n_bothz, N_j, vfoptions);
end


end

function [V_j_max, Pol_apr_max, Pol_d_combo] = Evaluate_ExpAssetSemiZ_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d1, N_d2, N_d3, N_semiz_safe, N_z_safe, num_a1, ...
    beta_j, EV_z_pre, pi_semiz_j, a2primeIndex, a2primeProbs, a1_work_local, ...
    ReturnFn, D1_cells, d2_gridvals, d3_gridvals, A1_cells, A2_cells, Z_cells, Semiz_cells, ReturnFnParamsVec, ...
    ezc2_j, ezc3, ezc4, ezc6_j, ezc7_j, ezc8_j)

N_block = length(state_idx);
N_d1_safe = max(1, N_d1);

V_j_max = -inf(N_block, N_a2, N_semiz_safe, N_z_safe, 'like', EV_z_pre);
Pol_apr_max = ones(N_block, N_a2, N_semiz_safe, N_z_safe, 'like', EV_z_pre);
Pol_d1_max = ones(N_block, N_a2, N_semiz_safe, N_z_safe, 'like', EV_z_pre);
Pol_d2_max = ones(N_block, N_a2, N_semiz_safe, N_z_safe, 'like', EV_z_pre);
Pol_d3_max = ones(N_block, N_a2, N_semiz_safe, N_z_safe, 'like', EV_z_pre);

A1_cells_block = cell(size(A1_cells));
for i = 1:length(A1_cells), A1_cells_block{i} = A1_cells{i}(1, 1, state_idx, :); end

% --- 0. Choice Grid Setup (Proper 6D Alignment) ---
% Dim 1: N_choice, Dim 2: N_d1, Dim 3: N_block, Dim 4: N_a2, Dim 5: N_semiz, Dim 6: N_z
if isempty(loweredge_matrix)
    N_choice = N_a1;
    apr_idx_tensor = reshape(1:N_a1, [N_choice, 1, 1, 1, 1, 1]);
else
    N_choice = maxgap_scalar + 1;
    offset = reshape(gpuArray(0:maxgap_scalar), [N_choice, 1, 1, 1, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, N_block, N_a2, N_semiz_safe, N_z_safe]);
    apr_idx_tensor = base_edge + offset;
end

A1prime_cells = cell(1, num_a1);
for i = 1:num_a1
    A1prime_cells{i} = reshape(a1_work_local(apr_idx_tensor(:), i), size(apr_idx_tensor));
end

% SHIFTED: a2 is Dim 4, semiz is Dim 5, z is Dim 6!
a2_offset    = reshape(0:N_a2-1,         [1, 1, 1, N_a2, 1, 1]) .* N_a1;
semiz_offset = reshape(0:N_semiz_safe-1, [1, 1, 1, 1, N_semiz_safe, 1]) .* (N_a1 * N_a2);
z_offset     = reshape(0:N_z_safe-1,     [1, 1, 1, 1, 1, N_z_safe]) .* (N_a1 * N_a2 * N_semiz_safe);
lin_idx = apr_idx_tensor + a2_offset + semiz_offset + z_offset;

for i_d3 = 1:N_d3
    % --- A. Integrate Semi-Exogenous State ---
    if N_semiz_safe > 1
        pi_semiz_d3 = pi_semiz_j(:, :, i_d3); % [N_semiz, N_semiz_next]
        EV_flat = reshape(EV_z_pre, [N_a1 * N_a2, N_semiz_safe, N_z_safe]);
        EV_perm = permute(EV_flat, [1, 3, 2]);
        EV_reshaped = reshape(EV_perm, [N_a1 * N_a2 * N_z_safe, N_semiz_safe]);
        EV_integrated = EV_reshaped * pi_semiz_d3';
        EV_semiz = permute(reshape(EV_integrated, [N_a1 * N_a2, N_z_safe, N_semiz_safe]), [1, 3, 2]);
        EV_semiz = reshape(EV_semiz, [N_a1, N_a2, N_semiz_safe, N_z_safe]);
    else
        EV_semiz = EV_z_pre;
    end

    % --- B. EZ Certainty Equivalent Unpacking ---
    valid_EV = isfinite(EV_semiz) & (EV_semiz ~= 0);
    if ezc6_j ~= 1
        EV_semiz(valid_EV) = max(EV_semiz(valid_EV), 0).^ezc6_j;
    end
    if ezc8_j ~= 1
        EV_semiz(valid_EV) = max(EV_semiz(valid_EV), 0).^ezc8_j;
    end

    D3_cells = cell(1, numel(d3_gridvals(i_d3,:)));
    for i = 1:length(D3_cells), D3_cells{i} = d3_gridvals(i_d3, i); end

    for i_d2 = 1:N_d2
        % --- C. Experience Asset Interpolation ---
        idx = a2primeIndex(i_d2, :, :);     % [1, N_a2, N_semiz]
        probs = a2primeProbs(i_d2, :, :);   % [1, N_a2, N_semiz]

        idx_full = repmat(reshape(idx, [1, N_a2, N_semiz_safe, 1]), [N_a1, 1, 1, N_z_safe]);
        probs_full = repmat(reshape(probs, [1, N_a2, N_semiz_safe, 1]), [N_a1, 1, 1, N_z_safe]);

        ev_semiz_offset = reshape(0:N_semiz_safe-1, [1, 1, N_semiz_safe, 1]) .* (N_a1 * N_a2);
        ev_z_offset = reshape(0:N_z_safe-1, [1, 1, 1, N_z_safe]) .* (N_a1 * N_a2 * N_semiz_safe);
        a1_col = reshape(1:N_a1, [N_a1, 1, 1, 1]);

        lin_lower = a1_col + (idx_full - 1) .* N_a1 + ev_semiz_offset + ev_z_offset;
        lin_upper = a1_col + (min(idx_full + 1, N_a2) - 1) .* N_a1 + ev_semiz_offset + ev_z_offset;

        EV_lower = EV_semiz(lin_lower);
        EV_upper = EV_semiz(lin_upper);

        EV_interp = probs_full .* EV_lower + (1 - probs_full) .* EV_upper;
        skipinterp = (EV_lower == EV_upper);
        probs_full(skipinterp) = 0;
        EV_interp(probs_full == 0) = EV_upper(probs_full == 0);
        EV_interp(probs_full == 1) = EV_lower(probs_full == 1);
        EV_interp(isnan(EV_interp)) = -Inf;

        % --- D. Coarse RHS Assembly ---
        D2_cells = cell(1, numel(d2_gridvals(i_d2,:)));
        for i = 1:length(D2_cells), D2_cells{i} = d2_gridvals(i_d2, i); end

        F_tensor = ReturnFn(D1_cells{:}, D2_cells{:}, D3_cells{:}, A1prime_cells{:}, A1_cells_block{:}, A2_cells{:}, Semiz_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});

        EV_bc = reshape(EV_interp(lin_idx(:)), size(lin_idx));

        RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bc, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);

        expected_sz = [N_choice, N_d1_safe, N_block, N_a2, N_semiz_safe, N_z_safe];
        if ~isequal(size(RHS), expected_sz)
            RHS = RHS + zeros(expected_sz, 'like', EV_z_pre);
        end

        % Safely flatten by combining N_choice * N_d1 for the max() lookup
        RHS_flat = reshape(RHS, [N_choice * N_d1_safe, N_block * N_a2 * N_semiz_safe * N_z_safe]);
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
            loweredge_flat = repmat(reshape(loweredge_matrix, [1, N_a2 * N_semiz_safe * N_z_safe]), [N_block, 1]);
            apr_idx_local_2d = reshape(apr_idx_local, [N_block, N_a2 * N_semiz_safe * N_z_safe]);
            apr_idx_coarse = loweredge_flat + apr_idx_local_2d - 1;
        end

        V_sub = reshape(V_sub_coarse, [N_block, N_a2, N_semiz_safe, N_z_safe]);
        apr_idx = reshape(apr_idx_coarse, [N_block, N_a2, N_semiz_safe, N_z_safe]);
        d1_idx = reshape(d1_idx_coarse, [N_block, N_a2, N_semiz_safe, N_z_safe]);

        if i_d2 == 1 && i_d3 == 1
            update_mask = true(N_block, N_a2, N_semiz_safe, N_z_safe);
        else
            update_mask = V_sub > V_j_max;
        end

        V_j_max(update_mask) = V_sub(update_mask);
        Pol_apr_max(update_mask) = apr_idx(update_mask);
        Pol_d1_max(update_mask) = d1_idx(update_mask);
        Pol_d2_max(update_mask) = i_d2;
        Pol_d3_max(update_mask) = i_d3;
    end
end

Pol_d_combo = Pol_d1_max + (Pol_d2_max - 1) * N_d1_safe + (Pol_d3_max - 1) * N_d1_safe * max(1, N_d2);


end