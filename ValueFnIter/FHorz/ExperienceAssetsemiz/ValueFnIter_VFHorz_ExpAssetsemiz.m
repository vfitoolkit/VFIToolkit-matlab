function [V, Policy] = ValueFnIter_VFHorz_ExpAssetsemiz(n_d1, n_d2, n_d3, n_a1, n_a2, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, d3_gridvals, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Safe Dimension Setup ---
N_d1 = max(1, prod(n_d1)); N_d2 = max(1, prod(n_d2)); N_d3 = max(1, prod(n_d3));
N_a1 = max(1, prod(n_a1)); N_a2 = max(1, prod(n_a2));
N_z_safe = max(1, prod(n_z)); N_semiz_safe = max(1, prod(n_semiz));

% --- 2. GPU Transfer (if parallel) ---
if vfoptions.parallel == 2
    d1_gridvals = gpuArray(d1_gridvals); d2_gridvals = gpuArray(d2_gridvals); d3_gridvals = gpuArray(d3_gridvals);
    a1_gridvals = gpuArray(a1_gridvals); a2_grid = gpuArray(a2_grid);
    z_gridvals_J = gpuArray(z_gridvals_J); semiz_gridvals_J = gpuArray(semiz_gridvals_J);
    pi_z_J = gpuArray(pi_z_J); pi_semiz_J = gpuArray(pi_semiz_J);
end

% --- 3. EZ Universal Mix-In Context ---
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
% (Insert the standard boilerplate here for extracting aprimeFnParamNames,
% ReturnFnParamNames, and building D1_cells, D2_cells, D3_cells, A1_cells, A2_cells exactly like ExpAsset)
% ... [Standard toolkit cell extraction setup] ...

V = zeros(N_a1, N_a2, N_semiz_safe, N_z_safe, N_j, 'like', a2_grid);
V_next = zeros(N_a1, N_a2, N_semiz_safe, N_z_safe, 'like', a2_grid);
PolicyKron = zeros(N_a1, N_a2, N_semiz_safe, N_z_safe, N_j, 'like', a2_grid);
gridinterplayer = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;
% ... [Standard n2short / n2long / a1prime_grid setup] ...

% =========================================================
% TIME LOOP
% =========================================================
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    % ... [Extract beta_j, pi_z_j, and ReturnFnParamsVec] ...

    % 1. Create the ExpAsset Transitions (Note: depends on semiz!)
    aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    [a2primeIndex, a2primeProbs] = CreateExperienceAssetsemizFnMatrix(aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, semiz_gridvals_J(:,:,jj), aprimeFnParamsVec, 2);

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
        state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d1, N_d2, N_d3, N_semiz_safe, N_z_safe, ...
        beta_j, EV_z_pre, pi_semiz_J(:,:,:,jj), a1prime_grid, a2primeIndex, a2primeProbs, a1_work_local, ...
        ReturnFn, D1_cells, d2_gridvals, d3_gridvals, A1_cells, A2_cells, Z_cells, Semiz_cells, ReturnFnParamsVec, ...
        ezc2(jj), ezc3, ezc4, ezc6(jj), ezc7(jj), ezc8(jj));

    % 5. Route (DC Slicer vs Brute Force)
    % ... [Standard if divideandconquer == 1 block] ...

    V(:,:,:,:,jj) = V_j_max;
    V_next = V_j_max;
end

% ... [Standard UnKron Output Packing] ...
end


function [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_ExpAssetSemiZ_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d1, N_d2, N_d3, N_semiz_safe, N_z_safe, ...
    beta_j, EV_z_pre, pi_semiz_j, a1prime_grid, a2primeIndex, a2primeProbs, a1_work_local, ...
    ReturnFn, D1_cells, d2_gridvals, d3_gridvals, A1_cells, A2_cells, Z_cells, Semiz_cells, ReturnFnParamsVec, ...
    ezc2_j, ezc3, ezc4, ezc6_j, ezc7_j, ezc8_j)

N_block = length(state_idx);
V_j_max = -inf(N_block, N_a2, N_semiz_safe, N_z_safe, 'like', EV_z_pre);
% ... [Preallocate Pol_apr_max, Pol_d1, Pol_d2, Pol_d3] ...

for i_d3 = 1:N_d3

    % --- A. Integrate Semi-Exogenous State ---
    if N_semiz_safe > 1
        pi_semiz_d3 = pi_semiz_j(:, :, i_d3); % [N_semiz, N_semiz_next]
        EV_flat = reshape(EV_z_pre, [N_a1 * N_a2, N_semiz_safe, N_z_safe]);

        % Permute to multiply: [N_a1*N_a2, N_z, N_semiz] * pi'
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

    for i_d2 = 1:N_d2

        % --- C. Experience Asset Interpolation ---
        idx = a2primeIndex(i_d2, :, :);     % [1, N_a2, N_semiz]
        probs = a2primeProbs(i_d2, :, :);   % [1, N_a2, N_semiz]

        idx_full = repmat(reshape(idx, [1, N_a2, N_semiz_safe, 1]), [N_a1, 1, 1, N_z_safe]);
        probs_full = repmat(reshape(probs, [1, N_a2, N_semiz_safe, 1]), [N_a1, 1, 1, N_z_safe]);

        % Extract linear bounds from EV_semiz
        % (Insert standard 1D linear indexing extraction for EV_lower and EV_upper here)

        EV_interp = probs_full .* EV_lower + (1 - probs_full) .* EV_upper;
        % (Apply Toolkit mask protections for probs==0 and probs==1 here)

        % --- D. Coarse RHS Assembly ---
        F_tensor = ReturnFn(...); % (Call with D1, D2(i_d2), D3(i_d3), a1, a2, semiz, z)

        % Slice the requested block of expected values for the DC Slicer
        EV_bc = reshape(EV_interp(lin_idx(:)), size(lin_idx));

        % Call our Universal Helper!
        RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bc, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);

        % (Run standard max() and Sub-Grid Refinement updates)
        % ... [Standard Update Mask Logic for V_j_max] ...
    end
end

% Pack combo index
Pol_d_combo = Pol_d1_max + (Pol_d2_max - 1) * N_d1_safe + (Pol_d3_max - 1) * N_d1_safe * N_d2_safe;


end