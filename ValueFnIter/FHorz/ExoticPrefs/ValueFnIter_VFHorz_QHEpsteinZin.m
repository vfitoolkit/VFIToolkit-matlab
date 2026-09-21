function [V, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

%% 1. Pre-computation and Setup
vfoptions.precision = underlyingType(a_grid);

% Ensure EZ Parameters are properly initialized
vfoptions = EpsteinZinSetup_VFHorz(N_j, Parameters, ReturnFnParamNames, DiscountFactorParamNames, vfoptions);
ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;

% Universal Packer Setup
l_a2 = 0;
if vfoptions.experienceasset > 0; l_a2 = vfoptions.experienceasset; end
if vfoptions.experienceassetz > 0; l_a2 = vfoptions.experienceassetz; end
if l_a2 > 0
    n_a1 = n_a(1:end-l_a2);
    n_a2 = n_a(end-l_a2+1:end);
else
    n_a1 = n_a;
    n_a2 = [];
end
N_a1 = prod(max(1, n_a1));
N_a2 = prod(max(1, n_a2));
N_a  = prod(max(1, n_a));
N_d  = prod(n_d); % Leave raw for boolean checks (N_d > 0)
N_d_safe = prod(max(1, n_d));
N_z  = prod(n_z); 
N_z_safe = prod(max(1, n_z));

% Build grids and cells
a1_grid_len = sum(n_a1);
a1_grid_vals = a_grid(1:a1_grid_len);
a2_grid_vals = a_grid(a1_grid_len+1:end);

has_e = isfield(vfoptions, 'n_e') && prod(vfoptions.n_e) > 0;
n_e_pass = 0;
if has_e; n_e_pass = vfoptions.n_e; e_work = vfoptions.e_grid; else; e_work = ones(1, 1, 'like', a_grid); end

[TensorReturnFn, D_cells_block, A1_cells, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_d, n_a1, n_z, n_e_pass, d_grid, a1_grid_vals, [], []);

if l_a2 > 0
    [TensoraprimeFn, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, 0, n_a2, 0, 0, [], a2_grid_vals, [], []);
else
    TensoraprimeFn = []; A2_cells = {};
end

A1_mat = zeros(N_a1, length(n_a1), 'like', a_grid);
for i_a = 1:length(n_a1); A1_mat(:, i_a) = A1_cells{i_a}(:); end
A2_mat = zeros(N_a2, length(n_a2), 'like', a_grid);
a2_grids_1d = cell(1, length(n_a2));
offset = 0;
for i_a = 1:length(n_a2)
    A2_mat(:, i_a) = A2_cells{i_a}(:);
    a2_grids_1d{i_a} = a2_grid_vals((offset + 1):(offset + n_a2(i_a)));
    offset = offset + n_a2(i_a);
end
for i_d = 1:length(D_cells_block); D_cells_block{i_d} = reshape(D_cells_block{i_d}, [N_d_safe, 1, 1, 1, 1]); end

aprimeFnParamsCell = {};
if l_a2 > 0
    aprimeFnParamNames = vfoptions.aprimeFnParamNames(isfield(Parameters, vfoptions.aprimeFnParamNames));
end

has_semiz = isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0;
if has_semiz
    N_semiz = prod(vfoptions.n_semiz);
    N_z_exog = max(1, prod(n_z) / N_semiz);
else
    N_semiz = 1; N_z_exog = max(1, prod(n_z));
end
n_z_work = N_semiz * N_z_exog;
n_e_work = max(1, prod(n_e_pass));
N_ze = n_z_work * n_e_work;

V = zeros(N_a, n_z_work, n_e_work, N_j, 'like', a_grid);
Valt = zeros(N_a, n_z_work, n_e_work, N_j, 'like', a_grid);
Valt_next = zeros(N_a, n_z_work, n_e_work, 'like', a_grid);
Policyalt = []; % Planner policy tracks primary policy in standard QH EZ

if vfoptions.gridinterplayer(1) == 1
    PolicyKron = zeros(3, N_a, n_z_work, n_e_work, N_j, 'like', a_grid);
    n2short = vfoptions.ngridinterp; n2long = n2short * 2 + 3;
    a1_work = A1_cells{1}(:);
    a1prime_grid = interp1(1:1:N_a1, a1_work, linspace(1, N_a1, N_a1 + (N_a1 - 1) * n2short))';
    idx = discretize(a1prime_grid, a1_work);
    idx(isnan(idx) | idx == length(a1_work)) = length(a1_work) - 1;
    interp_left_idx = idx(:); interp_right_idx = idx(:) + 1;
    a1_left = a1_work(interp_left_idx); a1_right = a1_work(interp_right_idx);
    interp_weights = (a1prime_grid(:) - a1_left) ./ (a1_right - a1_left);
    interp_weights(a1_right == a1_left) = 0;
    if vfoptions.parallel == 2
        interp_left_idx = gpuArray(interp_left_idx);
        interp_right_idx = gpuArray(interp_right_idx);
        interp_weights = gpuArray(interp_weights);
    end
else
    PolicyKron = zeros(N_a, n_z_work, n_e_work, N_j, 'like', a_grid);
    n2short = 0; n2long = 0; a1prime_grid = []; interp_left_idx = []; interp_right_idx = []; interp_weights = [];
end

% Chunker Setup
ze_chunks = {1:N_ze};
if ismember(vfoptions.lowmemory, [4, 5]) && l_a2 > 0; a2_chunks = num2cell(1:N_a2); else; a2_chunks = {1:N_a2}; end

chunk_meta = cell(1, length(ze_chunks));
for i_ze = 1:length(ze_chunks)
    c_ze = ze_chunks{i_ze};
    c_ze_cpu = gather(c_ze);
    [z_ind, e_ind] = ind2sub([n_z_work, n_e_work], c_ze_cpu);
    meta.z_vals = unique(z_ind); meta.e_vals = unique(e_ind);
    meta.n_z_loc = length(meta.z_vals); meta.n_e_loc = length(meta.e_vals);
    meta.N_ze_local = length(c_ze);
    chunk_meta{i_ze} = meta;
end

base_ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, 1, vfoptions.precision);
is_age_dependent = false(1, length(ReturnFnParamNames));
for ip = 1:length(ReturnFnParamNames)
    if numel(Parameters.(ReturnFnParamNames{ip})) == N_j; is_age_dependent(ip) = true; end
    if vfoptions.parallel == 2 && isnumeric(base_ReturnFnParamsCell{ip}) && ~isa(base_ReturnFnParamsCell{ip}, 'gpuArray')
        base_ReturnFnParamsCell{ip} = gpuArray(base_ReturnFnParamsCell{ip});
    end
end

if isfield(vfoptions,'survivalprobability'); sj = Parameters.(vfoptions.survivalprobability); else; sj = ones(N_j,1); end
warmglow = isfield(vfoptions,'WarmGlowBequestsFn');
N_dsemiz = 1; if has_semiz && N_d > 0; N_dsemiz = n_d(end); end

%% 2. Reverse Time Loop
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    if vfoptions.verbose == 1; fprintf('Finite horizon QHEZ: %i of %i \n', jj, N_j); end

    ReturnFnParamsCell = base_ReturnFnParamsCell;
    for ip = find(is_age_dependent)
        val = cast(Parameters.(ReturnFnParamNames{ip})(jj), vfoptions.precision);
        if vfoptions.parallel == 2; ReturnFnParamsCell{ip} = gpuArray(val); else; ReturnFnParamsCell{ip} = val; end
    end
    if l_a2 > 0; aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj); end

    % QHEZ specific Discount Factors extraction
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = DiscountFactorParamsVec(1);
    delta_j = DiscountFactorParamsVec(2);
    sj_val = sj(jj);

    pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    if has_e; pi_e_j = vfoptions.pi_e_J(:, min(jj + 1, size(vfoptions.pi_e_J, 2))); end
    if vfoptions.parallel == 2 && has_e && ~isa(pi_e_j, 'gpuArray'); pi_e_j = gpuArray(pi_e_j); end

    if jj == N_j && (~isfield(vfoptions, 'V_Jplus1') || isempty(vfoptions.V_Jplus1))
        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', a_grid); end
            valid_wg = isfinite(WG_eval) & (WG_eval ~= 0); WG_transformed = WG_eval;
            if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
            WG_transformed(WG_eval == 0) = 0;
            EV_Valt_base = repmat(reshape(WG_transformed, [N_a, 1, 1, 1]), [1, N_semiz * N_z_exog, n_e_work, N_dsemiz]);
        else
            EV_Valt_base = zeros(N_a, N_semiz * N_z_exog, n_e_work, N_dsemiz, 'like', a_grid);
        end
    else
        if jj == N_j
            Valt_next = reshape(vfoptions.V_Jplus1, [N_a, n_z_work, n_e_work]);
            if vfoptions.parallel == 2 && ~isa(Valt_next, 'gpuArray'); Valt_next = gpuArray(Valt_next); end
        end

        % EZ Transform applied strictly to Valt (the future self's value function)
        valid_V = isfinite(Valt_next) & (Valt_next ~= 0);
        V_transformed = Valt_next;
        if ezc5(jj) == 1; V_transformed(valid_V) = ezc4 * Valt_next(valid_V); else; V_transformed(valid_V) = max(ezc4 * Valt_next(valid_V), 0).^ezc5(jj); end
        V_transformed(Valt_next == 0) = 0;

        if has_e
            V_trans_flat = reshape(V_transformed, [N_a * n_z_work, n_e_work]);
            V_inf_mask = (V_trans_flat == -Inf); V_safe = V_trans_flat; V_safe(V_inf_mask) = -1e250;
            V_expected_e = V_safe * pi_e_j(:);
            inf_restore = (V_inf_mask * (pi_e_j(:) > 0)) > 0; V_expected_e(inf_restore) = -Inf;
            V_transformed = repmat(reshape(V_expected_e, [N_a, n_z_work, 1]), [1, 1, n_e_work]);
        end

        EV_Valt_base = zeros(N_a, N_semiz * N_z_exog, n_e_work, N_dsemiz, 'like', Valt_next);
        for ie = 1:n_e_work
            V_curr = V_transformed(:,:,ie);
            if N_z_exog > 1 && prod(n_z) > 0
                V_slice = reshape(V_curr, [N_a * N_semiz, N_z_exog]);
                V_inf_mask = (V_slice == -Inf); V_safe = V_slice; V_safe(V_inf_mask) = -1e250;
                V_z_eval = V_safe * pi_z_j';
                inf_restore = (V_inf_mask * (pi_z_j' > 0)) > 0; V_z_eval(inf_restore) = -Inf;
                V_z_eval = reshape(V_z_eval, [N_a, N_semiz, N_z_exog]);
            else
                V_z_eval = reshape(V_curr, [N_a, N_semiz, N_z_exog]);
            end

            if has_semiz
                pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));
                V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz, N_a * N_z_exog]);
                V_inf_mask = (V_perm == -Inf); V_safe = V_perm; V_safe(V_inf_mask) = -1e250;
                for idsemiz = 1:N_dsemiz
                    pi_semiz_d = pi_semiz_j(:, :, idsemiz);
                    EV_perm = pi_semiz_d * V_safe;
                    inf_restore = (pi_semiz_d > 0) * V_inf_mask > 0; EV_perm(inf_restore) = -Inf;
                    EV_d = permute(reshape(EV_perm, [N_semiz, N_a, N_z_exog]), [2, 1, 3]);
                    EV_Valt_base(:,:,ie,idsemiz) = reshape(EV_d, [N_a, N_semiz * N_z_exog]);
                end
            else
                EV_Valt_base(:,:,ie,1) = reshape(V_z_eval, [N_a, N_semiz * N_z_exog]);
            end
        end

        if warmglow == 1
            wg_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
            WG_eval = vfoptions.WarmGlowBequestsFn(a_grid, wg_params{:});
            if isscalar(WG_eval); WG_eval = WG_eval * ones(size(a_grid), 'like', a_grid); end
            valid_wg = isfinite(WG_eval) & (WG_eval ~= 0); WG_transformed = WG_eval;
            if ezc5(jj) == 1; WG_transformed(valid_wg) = ezc4 * WG_eval(valid_wg); else; WG_transformed(valid_wg) = max(ezc4 * WG_eval(valid_wg), 0).^ezc5(jj); end
            WG_transformed(WG_eval == 0) = 0; WG_eval = WG_transformed;
            WG_eval = reshape(WG_eval, [N_a, 1, 1, 1]);
            EV_Valt_base = EV_Valt_base * sj_val + (1 - sj_val) * WG_eval;
        end
    end

    % Reverse EZ transform to get the raw Certainty Equivalent
    valid_EV = isfinite(EV_Valt_base) & (EV_Valt_base ~= 0);
    if ezc6(jj) ~= 1; EV_Valt_base(valid_EV) = max(EV_Valt_base(valid_EV), 0).^ezc6(jj); end
    if ezc8(jj) ~= 1; EV_Valt_base(valid_EV) = max(EV_Valt_base(valid_EV), 0).^ezc8(jj); end

    EV_flat_ze = reshape(EV_Valt_base, [N_a, N_ze, N_dsemiz]);

    V_j_max        = zeros(N_a, N_ze, 'like', Valt_next);
    Valt_j_max     = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_apr_max    = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_d_max      = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_L2idx_max  = zeros(N_a, N_ze, 'like', Valt_next);
    Pol_L2flag_max = zeros(N_a, N_ze, 'like', Valt_next);

    if N_dsemiz > 1
        if isfield(vfoptions, 'l_dsemiz'); N_d_prefix = prod(max(1, n_d(1:end-vfoptions.l_dsemiz))); else; N_d_prefix = prod(max(1, n_d(1:end-1))); end
        dsemiz_idx = ceil((1:N_d_safe)' / N_d_prefix); dsemiz_idx_tensor = reshape(dsemiz_idx, [N_d_safe, 1, 1, 1]);
    else
        dsemiz_idx_tensor = ones(N_d_safe, 1, 1, 1);
    end

    %% 3. The QHEZ Master Orchestrator
    if vfoptions.divideandconquer == 1
        for i_ze = 1:length(ze_chunks)
            meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
            curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);
            EV_local = EV_flat_ze(:, curr_ze, :);

            num_z_vars = length(n_z); Z_cells_local = cell(1, num_z_vars);
            if size(z_gridvals_J, 2) ~= num_z_vars
                z_inflated = reshape(z_gridvals_J, [prod(n_z), num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
            else
                for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
            end

            if has_e
                num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                for ie_var = 1:num_e_vars; E_cells_local{ie_var} = reshape(e_work(meta.e_vals, ie_var), [1, 1, 1, 1, n_e_loc]); end
            else
                E_cells_local = {};
            end

            if vfoptions.gridinterplayer(1) == 1
                N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                EV_2d = reshape(EV_local, [N_a1, N_cols]);
                EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_ze_local, N_dsemiz]);
            else
                EV_interp_local = [];
            end

            if l_a2 == 0
                EV_reshaped = reshape(EV_local, [N_a1, n_z_loc, n_e_loc, N_dsemiz]);
                EV_d_sliced = EV_reshaped(:, :, :, dsemiz_idx_tensor(:));
                EV_bounded_pre = permute(EV_d_sliced, [4, 1, 5, 2, 3]);
                d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]); z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1), [1, 1, 1, n_z_loc, 1]);
                e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1 * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
            else
                EV_bounded_pre = []; static_EV_offset = [];
            end

            LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, dc_mode_override) Evaluate_QHEZ_TensorBlock(...
                state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d_safe, N_ze_local, ...
                Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
                vfoptions.gridinterplayer, n2short, n2long, beta_j, delta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override);

            full_state_chunk = 1:(N_a1 * N_a2);
            vfoptions.level1n = vfoptions.level1n(1);

            if vfoptions.gridinterplayer(1) == 1
                % TENSOR BRIDGE FIX: Stage 1 DC+GI Override
                temp_vfoptions = vfoptions; temp_vfoptions.gridinterplayer = 0;
                [~, p_apr_coarse] = ValueFnIter_DC1_Slicer(N_a1 * N_a2, N_a, 1, N_ze_local, temp_vfoptions, LocalBlockFn);
                [v, p_apr, p_d, p_l2idx, p_l2flag, valt] = LocalBlockFn(full_state_chunk, p_apr_coarse, n2long - 1, 0);
            else
                [v, p_apr, p_d] = ValueFnIter_DC1_Slicer(N_a1 * N_a2, N_a, 1, N_ze_local, vfoptions, LocalBlockFn);
                % QHEZ Counterfactual Fix: Evaluate Valt exactly at the chosen DC policy
                [~, ~, ~, p_l2idx, p_l2flag, valt] = LocalBlockFn(full_state_chunk, p_apr, 0, 0);
            end

            V_j_max(:, curr_ze)     = reshape(v,     [N_a1 * N_a2, N_ze_local]);
            Valt_j_max(:, curr_ze)  = reshape(valt,  [N_a1 * N_a2, N_ze_local]);
            Pol_apr_max(:, curr_ze) = reshape(p_apr, [N_a1 * N_a2, N_ze_local]);
            Pol_d_max(:, curr_ze)   = reshape(p_d,   [N_a1 * N_a2, N_ze_local]);
            if vfoptions.gridinterplayer(1) == 1
                Pol_L2idx_max(:, curr_ze)  = reshape(p_l2idx,  [N_a1 * N_a2, N_ze_local]);
                Pol_L2flag_max(:, curr_ze) = reshape(p_l2flag, [N_a1 * N_a2, N_ze_local]);
            end
        end
    else
        for i_a2 = 1:length(a2_chunks)
            curr_a2 = a2_chunks{i_a2}; N_a2_local = length(curr_a2);
            start_a_idx = (min(curr_a2) - 1) * N_a1 + 1; end_a_idx = max(curr_a2) * N_a1;
            for i_ze = 1:length(ze_chunks)
                curr_ze = ze_chunks{i_ze}; N_ze_local = length(curr_ze);
                meta = chunk_meta{i_ze}; n_z_loc = meta.n_z_loc; n_e_loc = meta.n_e_loc;
                if l_a2 > 0; A2_local = A2_mat(curr_a2, :); else; A2_local = []; end
                EV_local = EV_flat_ze(:, curr_ze, :);

                num_z_vars = length(n_z); Z_cells_local = cell(1, num_z_vars);
                if size(z_gridvals_J, 2) ~= num_z_vars
                    z_inflated = reshape(z_gridvals_J, [prod(n_z), num_z_vars, size(z_gridvals_J, ndims(z_gridvals_J))]);
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_inflated(meta.z_vals, iz, min(jj, size(z_inflated,3))), [1, 1, 1, n_z_loc, 1]); end
                else
                    for iz = 1:num_z_vars; Z_cells_local{iz} = reshape(z_gridvals_J(meta.z_vals, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, n_z_loc, 1]); end
                end
                if has_e
                    num_e_vars = size(e_work, 2); E_cells_local = cell(1, num_e_vars);
                    for ie = 1:num_e_vars; E_cells_local{ie} = reshape(e_work(meta.e_vals, ie), [1, 1, 1, 1, n_e_loc]); end
                else; E_cells_local = {}; end

                if vfoptions.gridinterplayer(1) == 1
                    N_cols = N_ze_local * N_dsemiz; zero_weights = (interp_weights == 0); one_weights = (interp_weights == 1);
                    EV_2d = reshape(EV_local, [N_a1, N_cols]);
                    EV_left_val = EV_2d(interp_left_idx, :); EV_right_val = EV_2d(interp_right_idx, :);
                    EV_interp_flat = EV_left_val + interp_weights .* (EV_right_val - EV_left_val);
                    EV_interp_flat(zero_weights, :) = EV_left_val(zero_weights, :); EV_interp_flat(one_weights, :) = EV_right_val(one_weights, :);
                    EV_interp_flat(isnan(EV_interp_flat)) = -Inf;
                    EV_interp_local = reshape(EV_interp_flat, [length(a1prime_grid), N_ze_local, N_dsemiz]);
                else; EV_interp_local = []; end

                if l_a2 == 0
                    apr_idx_tensor = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
                    z_offset_broadcast = reshape((0:N_ze_local-1) * N_a, [1, 1, 1, n_z_loc, n_e_loc]);
                    idx_base = apr_idx_tensor + z_offset_broadcast;
                    max_idx_row = N_a1 * n_z_work * n_e_work;
                    linear_idx_pre = idx_base + (dsemiz_idx_tensor - 1) * max_idx_row;
                    EV_bounded_pre = reshape(EV_local(linear_idx_pre(:)), [N_d_safe, N_a1, 1, n_z_loc, n_e_loc]);
                    d_vec = reshape(0:N_d_safe-1, [N_d_safe, 1, 1, 1, 1]); z_vec = reshape((0:n_z_loc-1) * (N_d_safe * N_a1), [1, 1, 1, n_z_loc, 1]);
                    e_vec = reshape((0:n_e_loc-1) * (N_d_safe * N_a1 * n_z_loc), [1, 1, 1, 1, n_e_loc]);
                    static_EV_offset = cast(d_vec + 1 + z_vec + e_vec, 'like', EV_bounded_pre);
                else; EV_bounded_pre = []; static_EV_offset = []; end

                LocalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar, dc_mode_override) Evaluate_QHEZ_TensorBlock(...
                    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2_local, N_d_safe, N_ze_local, ...
                    Z_cells_local, E_cells_local, D_cells_block, A1_mat, A2_local, a2_grids_1d, l_a2, ...
                    vfoptions.gridinterplayer, n2short, n2long, beta_j, delta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
                    TensorReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), ...
                    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, dc_mode_override);

                % TENSOR BRIDGE FIX: Memory chunking locked to coarse grid
                state_list = start_a_idx:end_a_idx; total_states = length(state_list);
                flat_choices = max(1, N_d_safe) * N_a1;
                max_states_per_chunk = max(1, floor(50000000 / (flat_choices * n_z_loc * n_e_loc)));

                v_concat = []; valt_concat = []; p_apr_concat = []; p_d_concat = []; p_l2idx_concat = []; p_l2flag_concat = [];
                for chunk_start = 1:max_states_per_chunk:total_states
                    chunk_end = min(total_states, chunk_start + max_states_per_chunk - 1);
                    state_chunk = state_list(chunk_start:chunk_end);
                    if vfoptions.gridinterplayer(1) == 1
                        [~, p_apr_coarse, ~, ~, ~, ~] = LocalBlockFn(state_chunk, [], 0, 2);
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c, valt_c] = LocalBlockFn(state_chunk, p_apr_coarse, n2long - 1, 0);
                    else
                        [v_c, p_apr_c, p_d_c, p_l2idx_c, p_l2flag_c, valt_c] = LocalBlockFn(state_chunk, [], 0, 0);
                    end
                    v_concat = [v_concat; v_c]; valt_concat = [valt_concat; valt_c]; p_apr_concat = [p_apr_concat; p_apr_c]; p_d_concat = [p_d_concat; p_d_c];
                    if vfoptions.gridinterplayer(1) == 1; p_l2idx_concat = [p_l2idx_concat; p_l2idx_c]; p_l2flag_concat = [p_l2flag_concat; p_l2flag_c]; end
                end

                V_j_max(start_a_idx:end_a_idx, curr_ze)     = reshape(v_concat,    [N_a1 * N_a2_local, N_ze_local]);
                Valt_j_max(start_a_idx:end_a_idx, curr_ze)  = reshape(valt_concat, [N_a1 * N_a2_local, N_ze_local]);
                Pol_apr_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_apr_concat,[N_a1 * N_a2_local, N_ze_local]);
                Pol_d_max(start_a_idx:end_a_idx, curr_ze)   = reshape(p_d_concat,  [N_a1 * N_a2_local, N_ze_local]);
                if vfoptions.gridinterplayer(1) == 1
                    Pol_L2idx_max(start_a_idx:end_a_idx, curr_ze)  = reshape(p_l2idx_concat,  [N_a1 * N_a2_local, N_ze_local]);
                    Pol_L2flag_max(start_a_idx:end_a_idx, curr_ze) = reshape(p_l2flag_concat, [N_a1 * N_a2_local, N_ze_local]);
                end
            end
        end
    end

    V_j_max     = reshape(V_j_max,     [N_a, n_z_work, n_e_work]);
    Valt_j_max  = reshape(Valt_j_max,  [N_a, n_z_work, n_e_work]);
    Pol_apr_max = reshape(Pol_apr_max, [N_a, n_z_work, n_e_work]);
    Pol_d_max   = reshape(Pol_d_max,   [N_a, n_z_work, n_e_work]);
    if vfoptions.gridinterplayer(1) == 1
        Pol_L2idx_max  = reshape(Pol_L2idx_max,  [N_a, n_z_work, n_e_work]);
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_a, n_z_work, n_e_work]);
        lower_grid_pt = Pol_apr_max; subgrid_step  = Pol_L2idx_max;
        if N_d > 0; PolicyKron(1, :, :, :, jj) = (lower_grid_pt - 1) * N_d + Pol_d_max; else; PolicyKron(1, :, :, :, jj) = lower_grid_pt; end
        PolicyKron(2, :, :, :, jj) = subgrid_step; PolicyKron(3, :, :, :, jj) = Pol_L2flag_max;
    else
        if N_d > 0; PolicyKron(:, :, :, jj) = (Pol_apr_max - 1) * N_d + Pol_d_max; else; PolicyKron(:, :, :, jj) = Pol_apr_max; end
    end
    V(:, :, :, jj) = V_j_max; Valt(:, :, :, jj) = Valt_j_max; Valt_next = Valt_j_max;
end

if N_z == 0; V = squeeze(V); Valt = squeeze(Valt); end
if N_d == 0; n_daprime = n_a(1:length(n_a)); else; n_daprime = [n_d, n_a(1:length(n_a))]; end
if vfoptions.gridinterplayer(1) ~= 1; PolicyKron = shiftdim(PolicyKron, -1); end

disp('Unpacking QHEZ Policy tensor to System RAM...');
num_pol_vars = length(n_daprime); n_daprime_col = n_daprime(:); divisors = cumprod([1; n_daprime_col(1:end-1)]);
if vfoptions.gridinterplayer(1) == 1
    BaseIndexKron = PolicyKron(1, :, :, :, :);
    P_base_gpu = mod(floor((BaseIndexKron - 1) ./ divisors), n_daprime_col) + 1;
    Policy_flat = gather([P_base_gpu; PolicyKron(2:3, :, :, :, :)]);
else
    Policy_flat = gather(mod(floor((PolicyKron - 1) ./ divisors), n_daprime_col) + 1);
end

out_pol_vars = size(Policy_flat, 1);

% Filter out 0 placeholders
out_n_a = n_a(n_a > 0);
if isempty(out_n_a); out_n_a = 1; end

out_n_z = n_z(n_z > 0);
if isempty(out_n_z); out_n_z = 1; end

% Dynamically build the state dimension vector
state_shape = out_n_a;
if prod(n_z) > 0
    state_shape = [state_shape, out_n_z];
end
if has_e
    state_shape = [state_shape, n_e_pass];
end
state_shape = [state_shape, N_j]; % Append the time dimension

% Execute clean, single-call unpacks
Policy = reshape(Policy_flat, [out_pol_vars, state_shape]);
V = reshape(gather(V), state_shape);
Valt = reshape(gather(Valt), state_shape);


end


function [V_j_max, Pol_apr_max, Pol_d_max, Pol_L2idx_max, Pol_L2flag_max, Valt_j_max] = Evaluate_QHEZ_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_a1, N_a2, N_d_safe, N_ze_local, ...
    Z_cells_block, E_cells_block, D_cells_block, A1_mat, A2_mat, a2_grids_1d, l_a2, ...
    gridinterplayer, n2short, n2long, beta_j, delta_j, EV_local, EV_bounded_pre, EV_interp_local, a1prime_grid, ...
    TensorReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, ...
    TensoraprimeFn, aprimeFnParamsCell, N_dsemiz, dsemiz_idx_tensor, n_z_loc, n_e_loc, static_EV_offset, is_dc_mode)

N_states = length(state_idx);
if l_a2 > 0; [a1_sub, a2_sub] = ind2sub([N_a1, size(A2_mat, 1)], state_idx); else; a1_sub = state_idx; end

num_a1 = size(A1_mat, 2); A1_cells = cell(1, num_a1);
for ia = 1:num_a1; A1_cells{ia} = reshape(A1_mat(a1_sub, ia), [1, 1, N_states, 1, 1]); end
if l_a2 > 0
    num_a2 = size(A2_mat, 2); A2_cells = cell(1, num_a2);
    for ia = 1:num_a2; A2_cells{ia} = reshape(A2_mat(a2_sub, ia), [1, 1, N_states, 1, 1]); end
else; A2_cells = {}; end

if isempty(loweredge_matrix)
    if gridinterplayer(1) == 0 || is_dc_mode == 2
        % =================================================================
        % BRANCH 1A: COARSE EVALUATION
        % =================================================================
        Apr_cells = cell(1, num_a1);
        for ia = 1:num_a1; Apr_cells{ia} = reshape(A1_mat(:,ia), [1, N_a1, 1, 1, 1]); end

        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
        EV_bounded_base = EV_bounded_pre;

        FLAT_CHOICES = max(1, N_d_safe) * N_a1; FLAT_STATES = N_states * N_ze_local;

        % 1. Compute V (using beta * delta)
        EV_bounded_V = (beta_j * delta_j) .* EV_bounded_base;
        RHS_V = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded_V, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        [V_sub_coarse, Pol_sub_idx] = max(reshape(RHS_V, [FLAT_CHOICES, FLAT_STATES]), [], 1);
        clear RHS_V EV_bounded_V; % VRAM Garbage Collection

        % 2. Compute Valt (using delta) exactly at the chosen optimal V policy
        EV_bounded_Valt = delta_j .* EV_bounded_base;
        RHS_Valt = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded_Valt, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        RHS_Valt_flat = reshape(RHS_Valt, [FLAT_CHOICES, FLAT_STATES]);
        lin_idx_valt = Pol_sub_idx + (0:FLAT_STATES-1) * FLAT_CHOICES;
        Valt_sub_coarse = RHS_Valt_flat(lin_idx_valt);
        clear RHS_Valt RHS_Valt_flat EV_bounded_Valt EV_bounded_base;

        d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_idx_local = ceil(Pol_sub_idx / max(1, N_d_safe));
        V_j_max     = reshape(V_sub_coarse,  [N_states, N_ze_local]);
        Valt_j_max  = reshape(Valt_sub_coarse,  [N_states, N_ze_local]);
        Pol_apr_max = reshape(apr_idx_local, [N_states, N_ze_local]);
        Pol_d_max   = reshape(d_idx_local,   [N_states, N_ze_local]);
        Pol_L2idx_max = []; Pol_L2flag_max = [];
    else
        % =================================================================
        % BRANCH 1B: FULL FINE GRID EVALUATION (1-Step Brute Force)
        % =================================================================
        num_choices = length(a1prime_grid); choice_idx = reshape(1:num_choices, [1, num_choices, 1, 1, 1]);
        Apr_cells = cell(1, num_a1); for ia = 1:num_a1; Apr_cells{ia} = reshape(a1prime_grid, [1, num_choices, 1, 1, 1]); end

        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
        ze_offset = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, n_z_loc, n_e_loc]);
        L2_linear_idx = choice_idx + ze_offset;
        if N_dsemiz > 1; L2_linear_idx = L2_linear_idx + (dsemiz_idx_tensor - 1) * (length(a1prime_grid) * N_ze_local); end
        EV_bounded_base = EV_interp_local(L2_linear_idx);

        FLAT_CHOICES = max(1, N_d_safe) * num_choices; FLAT_STATES = N_states * N_ze_local;

        EV_bounded_V = (beta_j * delta_j) .* EV_bounded_base;
        RHS_V = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded_V, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        [V_sub_fine, Pol_sub_idx] = max(reshape(RHS_V, [FLAT_CHOICES, FLAT_STATES]), [], 1);
        clear RHS_V EV_bounded_V;

        EV_bounded_Valt = delta_j .* EV_bounded_base;
        RHS_Valt = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded_Valt, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
        RHS_Valt_flat = reshape(RHS_Valt, [FLAT_CHOICES, FLAT_STATES]);
        lin_idx_valt = Pol_sub_idx + (0:FLAT_STATES-1) * FLAT_CHOICES;
        Valt_sub_fine = RHS_Valt_flat(lin_idx_valt);
        clear RHS_Valt RHS_Valt_flat EV_bounded_Valt EV_bounded_base;

        d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_offset = ceil(Pol_sub_idx / max(1, N_d_safe));
        V_j_max    = reshape(V_sub_fine, [N_states, N_ze_local]);
        Valt_j_max = reshape(Valt_sub_fine, [N_states, N_ze_local]);
        Pol_d_max  = reshape(d_idx_local, [N_states, N_ze_local]);

        Pol_apr_max = floor((apr_offset - 1) / (n2short + 1)) + 1; Pol_apr_max = min(Pol_apr_max, N_a1 - 1);
        Pol_L2idx_max = apr_offset - (Pol_apr_max - 1) * (n2short + 1);
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]); Pol_L2idx_max = reshape(Pol_L2idx_max, [N_states, N_ze_local]);

        % TENSOR BRIDGE FIX: Dynamic Boundary Repeller
        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', V_j_max);
        idx_lower_coarse = (Pol_apr_max(:)' - 1) * (n2short + 1) + 1; idx_upper_coarse = min(num_choices, idx_lower_coarse + (n2short + 1));
        F_flat = reshape(F_tensor, [FLAT_CHOICES, FLAT_STATES]);
        lin_lower = d_idx_local(:)' + (idx_lower_coarse - 1) * max(1, N_d_safe) + (0:FLAT_STATES-1) * FLAT_CHOICES;
        lin_upper = d_idx_local(:)' + (idx_upper_coarse - 1) * max(1, N_d_safe) + (0:FLAT_STATES-1) * FLAT_CHOICES;
        isInfLower = (F_flat(lin_lower) == -Inf); isInfUpper = (F_flat(lin_upper) == -Inf);
        clear F_flat;

        isInnerOrUpper = (Pol_L2idx_max(:)' > 1); isInnerOrLower = (Pol_L2idx_max(:)' < n2short + 2);
        Pol_L2flag_max(isInnerOrUpper & isInfLower) = 3; Pol_L2flag_max(isInnerOrLower & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);
    end
else
    % =================================================================
    % BRANCH 2: ZOOM PHASE (loweredge_matrix provided)
    % =================================================================
    num_states_lower = size(loweredge_matrix, 1);
    if num_states_lower == 1 && N_states > 1; loweredge_matrix = repmat(loweredge_matrix, N_states, 1); end
    if gridinterplayer(1) == 0
        num_choices = maxgap_scalar + 1;
        base_idx = reshape(loweredge_matrix, [1, 1, N_states, n_z_loc, n_e_loc]);
        choice_idx = max(1, min(base_idx + reshape(0:maxgap_scalar, [1, num_choices, 1, 1, 1]), N_a1));
        Apr_cells = cell(1, num_a1); for ia = 1:num_a1; Apr_cells{ia} = A1_mat(choice_idx, ia); end

        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
        EV_bounded_base = EV_bounded_pre(static_EV_offset + (choice_idx - 1) * N_d_safe);
    else
        num_choices = n2long;
        loweredge_matrix = max(2, min(loweredge_matrix, N_a1 - 1));
        base_idx = reshape((loweredge_matrix - 1) * (n2short + 1) + 1, [1, 1, N_states, n_z_loc, n_e_loc]);
        start_offset = -(n2short + 1);
        raw_choice_idx = base_idx + reshape(start_offset:(n2short + 1), [1, num_choices, 1, 1, 1]);
        out_of_bounds = (raw_choice_idx < 1) | (raw_choice_idx > length(a1prime_grid));
        choice_idx = max(1, min(raw_choice_idx, length(a1prime_grid)));
        Apr_cells = cell(1, num_a1); for ia = 1:num_a1; Apr_cells{ia} = reshape(a1prime_grid(choice_idx), [1, num_choices, N_states, n_z_loc, n_e_loc]); end

        F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, E_cells_block{:}, ReturnFnParamsCell{:});
        ze_offset = reshape((0:N_ze_local-1) * length(a1prime_grid), [1, 1, 1, n_z_loc, n_e_loc]);
        choice_idx_cast = repmat(choice_idx, [1, 1, 1, 1, 1]); ze_offset_cast = repmat(ze_offset, [1, num_choices, N_states, 1, 1]);
        L2_linear_idx = choice_idx_cast(:) + ze_offset_cast(:);
        if N_dsemiz > 1; L2_linear_idx = L2_linear_idx + repmat((dsemiz_idx_tensor - 1) * (length(a1prime_grid) * N_ze_local), [1, num_choices, N_states, n_z_loc, n_e_loc]); end
        EV_bounded_base = reshape(EV_interp_local(L2_linear_idx), [1, num_choices, N_states, n_z_loc, n_e_loc]);
        EV_bounded_base(out_of_bounds) = -Inf;
    end

    FLAT_CHOICES = max(1, N_d_safe) * num_choices; FLAT_STATES = N_states * N_ze_local;

    EV_bounded_V = (beta_j * delta_j) .* EV_bounded_base;
    RHS_V = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded_V, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_sub_fine, Pol_sub_idx] = max(reshape(RHS_V, [FLAT_CHOICES, FLAT_STATES]), [], 1);
    clear RHS_V EV_bounded_V;

    EV_bounded_Valt = delta_j .* EV_bounded_base;
    RHS_Valt = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded_Valt, 1, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    RHS_Valt_flat = reshape(RHS_Valt, [FLAT_CHOICES, FLAT_STATES]);
    lin_idx_valt = Pol_sub_idx + (0:FLAT_STATES-1) * FLAT_CHOICES;
    Valt_sub_fine = RHS_Valt_flat(lin_idx_valt);
    clear RHS_Valt RHS_Valt_flat EV_bounded_Valt EV_bounded_base;

    d_idx_local = mod(Pol_sub_idx - 1, max(1, N_d_safe)) + 1; apr_offset = ceil(Pol_sub_idx / max(1, N_d_safe));
    V_j_max    = reshape(V_sub_fine, [N_states, N_ze_local]);
    Valt_j_max = reshape(Valt_sub_fine, [N_states, N_ze_local]);
    Pol_d_max  = reshape(d_idx_local, [N_states, N_ze_local]);

    if gridinterplayer(1) == 0
        Pol_apr_max = reshape(reshape(base_idx, [1, FLAT_STATES]) + apr_offset - 1, [N_states, N_ze_local]);
        Pol_L2idx_max = []; Pol_L2flag_max = [];
    else
        loweredge_matrix_flat = reshape(loweredge_matrix, [1, FLAT_STATES]);
        abs_fine_idx_flat = (loweredge_matrix_flat - 1) * (n2short + 1) + 1 + start_offset + apr_offset - 1;
        Pol_apr_max = floor((abs_fine_idx_flat - 1) / (n2short + 1)) + 1; Pol_apr_max = min(Pol_apr_max, N_a1 - 1);
        Pol_L2idx_max = reshape(abs_fine_idx_flat - (Pol_apr_max - 1) * (n2short + 1), [N_states, N_ze_local]);
        Pol_apr_max = reshape(Pol_apr_max, [N_states, N_ze_local]);

        % TENSOR BRIDGE FIX: Match Legacy L2flag behavior directly on Return Function
        linidx_lower = d_idx_local(:)' + (1 - 1) * max(1, N_d_safe) + (0:FLAT_STATES-1) * FLAT_CHOICES;
        linidx_upper = d_idx_local(:)' + (n2long - 1) * max(1, N_d_safe) + (0:FLAT_STATES-1) * FLAT_CHOICES;
        F_flat = reshape(F_tensor, [FLAT_CHOICES, FLAT_STATES]);
        isInfLower = (F_flat(linidx_lower) == -Inf); isInfUpper = (F_flat(linidx_upper) == -Inf);
        clear F_flat;

        inLowerStrict = (apr_offset(:)' >= 2) & (apr_offset(:)' <= n2short + 1);
        inUpperStrict = (apr_offset(:)' >= n2short + 3) & (apr_offset(:)' <= n2long - 1);
        Pol_L2flag_max = 2 * ones(1, FLAT_STATES, 'like', V_j_max);
        Pol_L2flag_max(inLowerStrict & isInfLower) = 3; Pol_L2flag_max(inUpperStrict & isInfUpper) = 1;
        Pol_L2flag_max = reshape(Pol_L2flag_max, [N_states, N_ze_local]);
    end
end


end