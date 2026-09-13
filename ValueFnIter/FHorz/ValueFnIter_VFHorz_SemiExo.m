function [V, Policy] = ValueFnIter_VFHorz_SemiExo(n_d1, n_d2, n_a, n_semiz, n_z, N_j, ...
    d1_gridvals, d2_gridvals, a_grid, z_gridvals_J, semiz_gridvals_J, ...
    pi_z_J, pi_semiz_J, ReturnFn, Parameters, ...
    DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Dimensions
n_d = [n_d1, n_d2];
N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
N_d = N_d1 * N_d2;
N_a = prod(n_a);
N_semiz = prod(n_semiz);
N_z = prod(n_z);
n_all_z = [n_semiz, n_z];
N_bothz = prod(n_all_z);
has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
if has_e, error('Vectorized SemiExo does not currently support e shocks.'); end

% --- AGGRESSIVE GPU ALLOCATION ---
if vfoptions.parallel == 2
    pi_semiz_J = gpuArray(pi_semiz_J);
    semiz_gridvals_J = gpuArray(semiz_gridvals_J);
    pi_z_J = gpuArray(pi_z_J);
    z_gridvals_J = gpuArray(z_gridvals_J);
    d1_gridvals = gpuArray(d1_gridvals);
    d2_gridvals = gpuArray(d2_gridvals);

    if isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        vfoptions.V_Jplus1 = gpuArray(vfoptions.V_Jplus1);
    end
end
% ---------------------------------

% 2. Choice Grid (D_cells)
d_gridvals = [repmat(d1_gridvals, N_d2, 1), repelem(d2_gridvals, N_d1, 1)];
num_d = length(n_d);
D_cells = cell(1, num_d);
for i_d = 1:num_d
    D_cells{i_d} = shiftdim(d_gridvals(:, i_d), -3); % Dim 4
end

% 3. Endogenous Grid (A_mat)
num_a = length(n_a);
if num_a > 1
    a_grids_1d = cell(1, num_a); offset = 0;
    for i_a = 1:num_a
        a_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a(i_a)));
        offset = offset + n_a(i_a);
    end
    [A_mesh_raw{1:num_a}] = ndgrid(a_grids_1d{:});
    A_mat = zeros(N_a, num_a, 'like', a_grid);
    for i_a = 1:num_a, A_mat(:, i_a) = A_mesh_raw{i_a}(:); end
else
    A_mat = a_grid(:);
end
a_work = A_mat(:, 1);

% 4. Preallocate
V = zeros(N_a, N_bothz, N_j, 'like', a_grid);
if vfoptions.gridinterplayer == 1
    PolicyKron = zeros(5, N_a, N_bothz, N_j, 'like', a_grid); % Must be 5 rows for GI!
else
    PolicyKron = zeros(3, N_a, N_bothz, N_j, 'like', a_grid);
end
V_next = zeros(N_a, N_bothz, 'like', a_grid);

% 5. Backward Induction
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    beta_j = prod(CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj));
    ReturnFnParamsVec = num2cell(CreateVectorFromParams(Parameters, ReturnFnParamNames, jj));

    % Joint Z Grid for this period
    bothz_gridvals_j = [repmat(semiz_gridvals_J(:,:,jj), N_z, 1), repelem(z_gridvals_J(:,:,jj), N_semiz, 1)];
    num_z = length(n_all_z);
    Z_cells = cell(1, num_z);
    for i_z = 1:num_z
        Z_cells{i_z} = shiftdim(bothz_gridvals_j(:, i_z), -1); % Dim 2
    end

    % --- EXPECTATIONS ---
    if jj == N_j
        if isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
            temp_V_rs = reshape(vfoptions.V_Jplus1, [N_a * N_semiz, N_z]);
            pi_z_j = pi_z_J(:,:,min(jj, size(pi_z_J, 3)));
            EV_raw_rs = temp_V_rs * (pi_z_j');
            EV_slice = reshape(EV_raw_rs, [N_a, N_semiz, N_z]);
        else
            EV_slice = zeros(N_a, N_semiz, N_z, 'like', a_grid);
            pi_z_j = eye(N_z, 'like', a_grid); % Dummy for dispatchers
        end
    else
        % Integrate out Exogenous Z (independent of choices)
        pi_z_j = pi_z_J(:,:,jj);
        temp_V_rs = reshape(V_next, [N_a * N_semiz, N_z]);
        EV_raw_rs = temp_V_rs * (pi_z_j');
        EV_slice = reshape(EV_raw_rs, [N_a, N_semiz, N_z]);
    end
    % --- GRID INTERPOLATION SETUP ---
    if vfoptions.gridinterplayer == 1
        n2short = vfoptions.ngridinterp;
        n2long  = n2short * 2 + 3;
        a1prime_grid = interp1(1:1:N_a, a_work, linspace(1, N_a, N_a + (N_a - 1) * n2short))';
    else
        n2short = 0;
        n2long  = 0;
        a1prime_grid = [];
    end

    pi_semiz_j = pi_semiz_J(:,:,:,min(jj, size(pi_semiz_J, 4)));

    % --- Determine Exogenous Memory Chunking (lowmemory) ---
    lowmem_level = 0;
    if isfield(vfoptions, 'lowmemory') && ~isempty(vfoptions.lowmemory)
        lowmem_level = vfoptions.lowmemory;
    end
    if lowmem_level == 0
        ze_chunks = {1:N_bothz};
    elseif lowmem_level == 1
        % z & semiz present: parallel over semiz, loop over z
        if N_z > 1 && N_semiz > 1
            ze_chunks = cell(1, N_z);
            for iz = 1:N_z
                ze_chunks{iz} = (iz - 1) * N_semiz + 1 : iz * N_semiz;
            end
        else
            ze_chunks = num2cell(1:N_bothz);
        end
    else
        ze_chunks = num2cell(1:N_bothz);
    end

    % Preallocate running max for this period
    V_j_max     = -inf(N_a, N_bothz, 'like', a_grid);
    Pol_d1_max  = ones(N_a, N_bothz, 'like', a_grid);
    Pol_d2_max  = ones(N_a, N_bothz, 'like', a_grid);
    Pol_apr_max = ones(N_a, N_bothz, 'like', a_grid);
    if vfoptions.gridinterplayer == 1
        Pol_tau_max = ones(N_a, N_bothz, 'like', a_grid);
        Pol_L2_max  = 2 * ones(N_a, N_bothz, 'like', a_grid);
    end

    % =========================================================
    % TILED MAP-REDUCE: Chunking over d2 and bothz
    % =========================================================
    for i_d2 = 1:N_d2
        % 1. Choice-Dependent Expectation for this d2
        temp_EV = reshape(EV_slice, [N_a, N_semiz, N_z]);
        temp_EV_flat = reshape(permute(temp_EV, [1, 3, 2]), [N_a * N_z, N_semiz]);
        EV_d2_flat = temp_EV_flat * pi_semiz_j(:,:,i_d2);
        EV_d2 = permute(reshape(EV_d2_flat, [N_a, N_z, N_semiz]), [1, 3, 2]);
        EV_d2_slice = reshape(EV_d2, [N_a, N_bothz]);

        % 2. Choice Grids for THIS d2 slice only (Size N_d1)
        d_gridvals_slice = [d1_gridvals, repmat(d2_gridvals(i_d2), N_d1, 1)];
        N_d_safe = max(1, N_d1);
        D_cells_block = cell(1, num_d);
        for i_d = 1:num_d
            D_cells_block{i_d} = reshape(d_gridvals_slice(:, i_d), [N_d_safe, 1, 1, 1]);
        end

        % 3. Exogenous Chunking (lowmemory)
        for i_ze = 1:length(ze_chunks)
            curr_ze = ze_chunks{i_ze};
            N_ze_local = length(curr_ze);

            % Build Z_cells for this chunk
            num_z_vars = size(bothz_gridvals_j, 2);
            Z_cells_local = cell(1, num_z_vars);
            for iz = 1:num_z_vars
                Z_cells_local{iz} = reshape(bothz_gridvals_j(curr_ze, iz), [1, 1, 1, N_ze_local]);
            end

            % Build EV variables for this chunk
            EV_local = EV_d2_slice(:, curr_ze);
            z_offset_local = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, N_ze_local]);
            if vfoptions.gridinterplayer
                EV_interp_local = interp1(a_work, EV_local, a1prime_grid);
                z_offset_fine_local = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, N_ze_local]);
            else
                EV_interp_local = [];
                z_offset_fine_local = [];
            end

            % Create localized closure targeting SemiExo Block
            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_SemiExo_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, N_a, N_d_safe, N_ze_local, ...
                Z_cells_local, {}, D_cells_block, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a_work, a1prime_grid, ...
                ReturnFn, ReturnFnParamsVec);

            % Dispatch to Slicer or Brute Force
            if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
                vfoptions.level1n = vfoptions.level1n(1);
                [v, p_apr, p_d1, p_l2idx, p_l2flag] = ValueFnIter_DC1_Slicer(N_a, N_a, 1, N_ze_local, vfoptions, LocalBlockFn);
            else
                [v, p_apr, p_d1, p_l2idx, p_l2flag] = LocalBlockFn(1:N_a, [], 0);
            end

            % 4. Reduce against global max
            update_mask = v > V_j_max(:, curr_ze);

            if any(update_mask, 'all')
                V_slice = V_j_max(:, curr_ze);
                V_slice(update_mask) = v(update_mask);
                V_j_max(:, curr_ze) = V_slice;

                d1_slice = Pol_d1_max(:, curr_ze);
                d1_slice(update_mask) = p_d1(update_mask);
                Pol_d1_max(:, curr_ze) = d1_slice;

                d2_slice = Pol_d2_max(:, curr_ze);
                d2_slice(update_mask) = i_d2;
                Pol_d2_max(:, curr_ze) = d2_slice;

                apr_slice = Pol_apr_max(:, curr_ze);
                apr_slice(update_mask) = p_apr(update_mask);
                Pol_apr_max(:, curr_ze) = apr_slice;

                if vfoptions.gridinterplayer == 1
                    tau_slice = Pol_tau_max(:, curr_ze);
                    tau_slice(update_mask) = p_l2idx(update_mask);
                    Pol_tau_max(:, curr_ze) = tau_slice;

                    L2_slice = Pol_L2_max(:, curr_ze);
                    L2_slice(update_mask) = p_l2flag(update_mask);
                    Pol_L2_max(:, curr_ze) = L2_slice;
                end
            end
        end % End Exogenous Chunking Loop
    end % End Map-Reduce Loop

    % Assign to global containers
    V(:,:,jj) = V_j_max;
    PolicyKron(1,:,:,jj) = Pol_d1_max;
    PolicyKron(2,:,:,jj) = Pol_d2_max;

    if vfoptions.gridinterplayer == 1
        % Apply toolkit safety clamp to the WINNING choices
        G_segments = vfoptions.ngridinterp + 1;
        at_top = (Pol_apr_max == N_a);

        Pol_apr_max(at_top) = N_a - 1;
        Pol_tau_max(at_top) = G_segments + 1;

        PolicyKron(3,:,:,jj) = Pol_apr_max;
        PolicyKron(4,:,:,jj) = Pol_tau_max;
        PolicyKron(5,:,:,jj) = Pol_L2_max;
    else
        PolicyKron(3,:,:,jj) = Pol_apr_max;
    end
    V_next = V(:,:,jj);
end
Policy = PolicyKron;
end

function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_SemiExo_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, ...
    gridinterplayer, n2short, n2long, beta_j, EV_local, EV_interp_local, z_offset_local, z_offset_fine_local, a_work_local, a1prime_grid, ...
    ReturnFn, ReturnFnParamsVec)

N_block = length(state_idx);

% --- 1. Choice Grid Setup (Implicit Dimensions) ---
if isempty(loweredge_matrix)
    N_choice = N_a;
    apr_idx_tensor = reshape(1:N_a, [1, N_choice, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar);
    N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [1, N_choice, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, 1, N_ze_local]);
    apr_idx_tensor = base_edge + offset;
end

% --- 2. State & Choice Tensor Construction ---
apr_in = reshape(a_work_local(apr_idx_tensor(:)), size(apr_idx_tensor));
a_in = reshape(a_work_local(state_idx), [1, 1, N_block, 1]);

% --- 3. Evaluate Return Function & Coarse RHS ---
F_tensor = ReturnFn(D_cells_block{:}, apr_in, a_in, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsVec{:});

EV_flat = reshape(EV_local, [N_a * N_ze_local, 1]);
linear_idx = apr_idx_tensor + z_offset_local;
EV_bounded = reshape(EV_flat(linear_idx(:)), size(linear_idx));

RHS = F_tensor + beta_j .* EV_bounded;

expected_sz = [N_d_safe, N_choice, N_block, N_ze_local];
if ~isequal(size(RHS), expected_sz)
    RHS = RHS + zeros(expected_sz, 'like', EV_local);
end

RHS_flat = reshape(RHS, [N_d_safe * N_choice, N_block * N_ze_local]);
[V_sub_coarse, Pol_sub_idx] = max(RHS_flat, [], 1);

d_idx_local   = mod(Pol_sub_idx - 1, N_d_safe) + 1;
apr_idx_local = ceil(Pol_sub_idx / N_d_safe);

if isempty(loweredge_matrix)
    apr_idx_coarse = apr_idx_local;
else
    loweredge_2d = repmat(reshape(loweredge_matrix, [1, N_ze_local]), [N_block, 1]);
    apr_idx_local_2d = reshape(apr_idx_local, [N_block, N_ze_local]);
    apr_idx_coarse = loweredge_2d + apr_idx_local_2d - 1;
end

apr_idx_coarse = reshape(apr_idx_coarse, [N_block, N_ze_local]);
d_idx_coarse   = reshape(d_idx_local, [N_block, N_ze_local]);

% --- 4. The Continuous Sub-Grid Refinement (GI1) ---
if gridinterplayer
    midpoint = max(min(apr_idx_coarse, N_a - 1), 2);
    base_idx = midpoint + (midpoint - 1) * n2short;
    offset   = (-n2short-1 : 1 : n2short+1)';
    fine_idx = base_idx(:)' + offset;

    fine_idx_4d = reshape(fine_idx, [1, n2long, N_block, N_ze_local]);
    apr_in_fine = reshape(a1prime_grid(fine_idx(:)), [1, n2long, N_block, N_ze_local]);
    a_in_fine   = reshape(a_work_local(state_idx), [1, 1, N_block, 1]);

    F_tensor_fine = ReturnFn(D_cells_block{:}, apr_in_fine, a_in_fine, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsVec{:});

    linear_fine_idx = fine_idx_4d + z_offset_fine_local;
    EV_fine = reshape(EV_interp_local(linear_fine_idx(:)), size(linear_fine_idx));

    RHS_fine = F_tensor_fine + beta_j .* EV_fine;

    expected_sz_fine = [N_d_safe, n2long, N_block, N_ze_local];
    if ~isequal(size(RHS_fine), expected_sz_fine)
        RHS_fine = RHS_fine + zeros(expected_sz_fine, 'like', EV_local);
    end

    RHS_fine_flat = reshape(RHS_fine, [N_d_safe * n2long, N_block * N_ze_local]);
    [V_sub_fine, maxindexL2] = max(RHS_fine_flat, [], 1);

    d_idx_fine    = mod(maxindexL2 - 1, N_d_safe) + 1;
    apr_step_fine = ceil(maxindexL2 / N_d_safe);

    isInfLower    = (RHS_fine_flat(1:N_d_safe, :) == -Inf);
    isInfUpper    = (RHS_fine_flat(end-N_d_safe+1:end, :) == -Inf);

    inLowerStrict = (apr_step_fine >= 2) & (apr_step_fine <= n2short + 1);
    inUpperStrict = (apr_step_fine >= n2short + 3) & (apr_step_fine <= n2long - 1);

    linear_win_d = d_idx_fine + (0:N_block*N_ze_local-1)*N_d_safe;
    L2flag_fine = 2 + (inLowerStrict & isInfLower(linear_win_d)) - (inUpperStrict & isInfUpper(linear_win_d));

    V_j_max        = reshape(V_sub_fine,    [N_block, N_ze_local]);
    Pol_apr_max    = reshape(midpoint,      [N_block, N_ze_local]);
    Pol_d_max      = reshape(d_idx_fine,    [N_block, N_ze_local]);
    Pol_L2idx_max  = reshape(apr_step_fine, [N_block, N_ze_local]);
    Pol_L2flag_max = reshape(L2flag_fine,   [N_block, N_ze_local]);
else
    V_j_max        = reshape(V_sub_coarse,   [N_block, N_ze_local]);
    Pol_apr_max    = reshape(apr_idx_coarse, [N_block, N_ze_local]);
    Pol_d_max      = d_idx_coarse;
    Pol_L2idx_max  = [];
    Pol_L2flag_max = [];
end


end