function [V, Policy] = ValueFnIter_VFHorz_ExpAsset(n_d1, n_d2, n_d3, n_a1, n_a2, n_z, n_semiz, N_j, d1_gridvals, d2_gridvals, d3_gridvals, a1_gridvals, a2_grid, z_gridvals_J, semiz_gridvals_J, pi_z_J, pi_semiz_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

N_d1 = prod(n_d1); N_d1_safe = max(1, N_d1);
N_d2 = prod(n_d2);
N_d3 = prod(n_d3); N_d3_safe = max(1, N_d3);
N_a1 = prod(n_a1);
N_a2 = prod(n_a2); N_a  = N_a1 * N_a2;
N_z  = prod(n_z);  N_z_safe = max(1, N_z);
N_semiz = prod(n_semiz); N_semiz_safe = max(1, N_semiz);
N_all_z_safe = N_z_safe * N_semiz_safe;

if vfoptions.parallel == 2
    d1_gridvals  = gpuArray(d1_gridvals);
    d2_gridvals  = gpuArray(d2_gridvals);
    a1_gridvals  = gpuArray(a1_gridvals);
    a2_grid      = gpuArray(a2_grid);
    if N_d3 > 0, d3_gridvals = gpuArray(d3_gridvals); end
    if N_z > 0, z_gridvals_J = gpuArray(z_gridvals_J); pi_z_J = gpuArray(pi_z_J); end
    if N_semiz > 0, semiz_gridvals_J = gpuArray(semiz_gridvals_J); pi_semiz_J = gpuArray(pi_semiz_J); end
end

aprimeFn = vfoptions.aprimeFn;
num_d2 = length(n_d2); if isempty(n_d2) || n_d2(1) == 0; num_d2 = 0; end
num_a2 = length(n_a2);
num_z = length(n_z); if isempty(n_z) || n_z(1) == 0; num_z = 0; end
num_semiz = length(n_semiz); if isempty(n_semiz) || n_semiz(1) == 0; num_semiz = 0; end

if isfield(vfoptions, 'aprimeFnParamNames')
    aprimeFnParamNames = vfoptions.aprimeFnParamNames;
else
    temp = getAnonymousFnInputNames(aprimeFn);
    num_prefix = num_d2 + num_a2 + (num_a2 >= 2) + num_z + num_semiz;
    if length(temp) > num_prefix
        aprimeFnParamNames = {temp{num_prefix + 1:end}};
    else
        aprimeFnParamNames = {};
    end
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
    apr_in = gpuArray(0); A1_cells = {};
end

a2_gridvals = CreateGridvals(n_a2, a2_grid, 1);
A2_cells = cell(1, num_a2);
for i = 1:num_a2
    A2_cells{i} = shiftdim(a2_gridvals(:, i), -3);
end

V      = zeros(N_a1, N_a2, N_all_z_safe, N_j, 'like', a2_grid);
V_next = zeros(N_a1, N_a2, N_all_z_safe, 'like', a2_grid);

gridinterplayer = (vfoptions.gridinterplayer(1) == 1);
if gridinterplayer
    n2short = vfoptions.ngridinterp;
    n2long  = n2short * 2 + 3;
    a1prime_grid = interp1(1:1:N_a1, a1_gridvals(:, 1), linspace(1, N_a1, N_a1 + (N_a1 - 1) * n2short))';
    PolicyKron = zeros(4, N_a1, N_a2, N_all_z_safe, N_j, 'like', a2_grid);
else
    % Allocate 2 explicit layers: Layer 1 for decisions, Layer 2 for asset choices
    PolicyKron = zeros(2, N_a1, N_a2, N_all_z_safe, N_j, 'like', a2_grid);
    n2short = 0; n2long = 0; a1prime_grid = [];
end

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;
    
    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(gpuArray(vfoptions.V_Jplus1), [N_a1, N_a2, N_all_z_safe]);
    end
    
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);
    aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);

    Z_cells = {}; pi_z_j = [];
    if N_z > 0
        z_work_j = z_gridvals_J(:, :, min(jj, size(z_gridvals_J, 3)));
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    end
    
    SemiZ_cells = {}; pi_semiz_j = [];
    if N_semiz > 0
        semiz_work_j = semiz_gridvals_J(:, :, min(jj, size(semiz_gridvals_J, 3)));
        pi_semiz_j = pi_semiz_J(:, :, :, min(jj, size(pi_semiz_J, 4)));
    end

    % Unify Z and SemiZ Grids into a single 5th dimension
    if N_all_z_safe > 1
        [z_idx_mesh, semiz_idx_mesh] = ndgrid(1:N_z_safe, 1:N_semiz_safe);
        z_mesh_flat = z_idx_mesh(:);
        sz_mesh_flat = semiz_idx_mesh(:);
        
        for i = 1:num_z
            z_col = z_work_j(:, i);
            Z_cells{i} = shiftdim(z_col(z_mesh_flat), -4);
        end
        for i = 1:num_semiz
            sz_col = semiz_work_j(:, i);
            SemiZ_cells{i} = shiftdim(sz_col(sz_mesh_flat), -4);
        end
        
        [d2_mesh, a2_mesh, all_z_idx] = ndgrid(d2_gridvals(:,1), a2_grid(:), 1:N_all_z_safe);
        aprime_args = {d2_mesh, a2_mesh};
        
        % Only append Z cells if the asset function explicitly requires them
        if isfield(vfoptions, 'experienceassetz') && vfoptions.experienceassetz > 0
            z_mesh_cells = cell(1, num_z);
            for iz = 1:num_z
                z_val_col = z_work_j(:, iz);
                mapped_z = z_val_col(z_mesh_flat);
                z_mesh_cells{iz} = mapped_z(all_z_idx);
            end
            aprime_args = [aprime_args, z_mesh_cells];
        end
        
        % Only append SemiZ cells if the asset function explicitly requires them
        if isfield(vfoptions, 'experienceassetsemiz') && vfoptions.experienceassetsemiz > 0
            semiz_mesh_cells = cell(1, num_semiz);
            for is = 1:num_semiz
                sz_val_col = semiz_work_j(:, is);
                mapped_sz = sz_val_col(sz_mesh_flat);
                semiz_mesh_cells{is} = mapped_sz(all_z_idx);
            end
            aprime_args = [aprime_args, semiz_mesh_cells];
        end
        
        aprime_args = [aprime_args, aprimeFnParamsCell];
        a2_prime_vals = aprimeFn(aprime_args{:});
        expected_size = [N_d2, N_a2, N_all_z_safe];
    else
        [d2_mesh, a2_mesh] = ndgrid(d2_gridvals(:,1), a2_grid(:));
        aprime_args = {d2_mesh, a2_mesh};
        
        if isfield(vfoptions, 'experienceassetz') && vfoptions.experienceassetz > 0
            z_mesh_cells = cell(1, num_z);
            for iz = 1:num_z
                z_mesh_cells{iz} = z_work_j(1, iz) + zeros(size(d2_mesh), 'like', d2_mesh);
            end
            aprime_args = [aprime_args, z_mesh_cells];
        end
        if isfield(vfoptions, 'experienceassetsemiz') && vfoptions.experienceassetsemiz > 0
            semiz_mesh_cells = cell(1, num_semiz);
            for is = 1:num_semiz
                semiz_mesh_cells{is} = semiz_work_j(1, is) + zeros(size(d2_mesh), 'like', d2_mesh);
            end
            aprime_args = [aprime_args, semiz_mesh_cells];
        end
        
        aprime_args = [aprime_args, aprimeFnParamsCell];
        a2_prime_vals = aprimeFn(aprime_args{:});
        expected_size = [N_d2, N_a2, 1];
    end

    if ~isequal(size(a2_prime_vals), expected_size)
        a2_prime_vals = a2_prime_vals + zeros(expected_size, 'like', a2_grid);
    end
    a2_prime_vals = max(a2_grid(1), min(a2_grid(end), a2_prime_vals));

    [~, a2primeIndex] = histc(a2_prime_vals(:), a2_grid);
    a2primeIndex = max(1, min(a2primeIndex, N_a2 - 1));
    a2_step = a2_grid(a2primeIndex + 1) - a2_grid(a2primeIndex);
    a2_step(a2_step == 0) = 1; 
    a2primeProbs = (a2_grid(a2primeIndex + 1) - a2_prime_vals(:)) ./ a2_step;
    a2primeProbs = max(0, min(1, a2primeProbs));
    
    a2primeIndex = reshape(a2primeIndex, expected_size);
    a2primeProbs = reshape(a2primeProbs, expected_size);

    a1_work_local = a1_gridvals(:, 1);

    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_ExpAsset_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, ...
        N_a1, N_a2, N_d1, N_d2, N_d3_safe, N_z, N_semiz, N_all_z_safe, gridinterplayer, n2short, n2long, ...
        beta_j, V_next, pi_z_j, pi_semiz_j, a1prime_grid, a2primeIndex, a2primeProbs, ...
        a1_work_local, a1_gridvals, ...
        ReturnFn, D1_cells, d2_gridvals, d3_gridvals, A1_cells, A2_cells, Z_cells, SemiZ_cells, ReturnFnParamsCell);

    if vfoptions.divideandconquer == 1
        vfoptions.level1n = vfoptions.level1n(1);
        [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = ...
            ValueFnIter_DC1_Slicer(N_a1, N_a1, N_a2, N_all_z_safe, vfoptions, EvalBlockFn);
    else
        [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = ...
            EvalBlockFn(1:N_a1, [], 0);
    end

    d_idx = Pol_d_combo;

    if gridinterplayer
        adjust = (Pol_L2idx_max < 1 + n2short + 1);
        lower_grid_pt = Pol_apr_max - adjust;
        lower_grid_pt = max(lower_grid_pt, 1);
        subgrid_step  = adjust .* Pol_L2idx_max + (1 - adjust) .* (Pol_L2idx_max - n2short - 1);
        
        G_segments = n2short + 1;
        at_top = (lower_grid_pt >= N_a1);
        lower_grid_pt(at_top) = N_a1 - 1;
        subgrid_step(at_top)  = G_segments + 1;
        
        PolicyKron(1, :, :, :, jj) = d_idx;
        PolicyKron(2, :, :, :, jj) = lower_grid_pt;
        PolicyKron(3, :, :, :, jj) = subgrid_step;
        PolicyKron(4, :, :, :, jj) = Pol_L2flag_max;
    else
        PolicyKron(1, :, :, :, jj) = d_idx;
        PolicyKron(2, :, :, :, jj) = max(Pol_apr_max, 1);
    end
    V(:, :, :, jj) = V_j_max;
    V_next = V_j_max;
end

n_a_vec = [n_a1, n_a2];
if n_d1 > 0 && n_d1(1) > 0
    n_d_vec = [n_d1, n_d2];
else
    n_d_vec = n_d2;
end
if N_d3 > 0, n_d_vec = [n_d_vec, n_d3]; end

n_a_vec = [n_a1, n_a2];
if N_z == 0 && N_semiz == 0
    V = reshape(V, [n_a_vec, N_j]);
    if gridinterplayer
        Policy = reshape(PolicyKron, [4, n_a_vec, N_j]);
    else
        Policy = reshape(PolicyKron, [2, n_a_vec, N_j]);
    end
else
    V = reshape(V, [n_a_vec, max(1, n_semiz), max(1, n_z), N_j]);
    if gridinterplayer
        Policy = reshape(PolicyKron, [4, n_a_vec, max(1, n_semiz), max(1, n_z), N_j]);
    else
        Policy = reshape(PolicyKron, [2, n_a_vec, max(1, n_semiz), max(1, n_z), N_j]);
    end
end


end



function [V_j_max, Pol_apr_max, Pol_d_combo, Pol_L2idx_max, Pol_L2flag_max] = Evaluate_ExpAsset_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, ...
    N_a1, N_a2, N_d1, N_d2, N_d3_safe, N_z, N_semiz, N_all_z_safe, gridinterplayer, n2short, n2long, ...
    beta_j, V_next, pi_z_j, pi_semiz_j, a1prime_grid, a2primeIndex, a2primeProbs, ...
    a1_work_local, a1_gridvals, ...
    ReturnFn, D1_cells, d2_gridvals, d3_gridvals, A1_cells, A2_cells, Z_cells, SemiZ_cells, ReturnFnParamsVec)

N_block = length(state_idx);
N_d1_safe = max(1, N_d1);

V_j_max     = -inf(N_block, N_a2, N_all_z_safe, 'like', V_next);
Pol_apr_max = ones(N_block, N_a2, N_all_z_safe, 'like', V_next);
Pol_d1_max  = ones(N_block, N_a2, N_all_z_safe, 'like', V_next);
Pol_d2_max  = ones(N_block, N_a2, N_all_z_safe, 'like', V_next);
Pol_d3_max  = ones(N_block, N_a2, N_all_z_safe, 'like', V_next);

if gridinterplayer
    Pol_L2idx_max  = ones(N_block, N_a2, N_all_z_safe, 'like', V_next);
    Pol_L2flag_max = 2 * ones(N_block, N_a2, N_all_z_safe, 'like', V_next);
else
    Pol_L2idx_max = []; Pol_L2flag_max = [];
end

A1_cells_block = cell(size(A1_cells));
for i = 1:length(A1_cells)
    A1_cells_block{i} = A1_cells{i}(1, 1, state_idx, :);
end

if isempty(loweredge_matrix)
    N_choice = N_a1;
    apr_idx_tensor = reshape(1:N_a1, [N_choice, 1, 1, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar);
    N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [N_choice, 1, 1, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, 1, 1, N_a2, N_all_z_safe]);
    apr_idx_tensor = base_edge + offset;
end

num_a1_vars = size(a1_gridvals, 2);
A1prime_cells = cell(1, num_a1_vars);
for i_a = 1:num_a1_vars
    a1_col = a1_gridvals(:, i_a);
    A1prime_cells{i_a} = reshape(a1_col(apr_idx_tensor(:)), size(apr_idx_tensor));
end

a2_offset = reshape(0:N_a2-1, [1, 1, 1, N_a2, 1]) .* N_a1;
z_offset  = reshape(0:N_all_z_safe-1, [1, 1, 1, 1, N_all_z_safe]) .* (N_a1 * N_a2);

for i_d3 = 1:N_d3_safe
    % Route Joint Markov Matrices
    if N_semiz > 0
        pi_sz = pi_semiz_j(:, :, i_d3);
        if N_z > 0
            pi_joint = kron(pi_sz, pi_z_j);
        else
            pi_joint = pi_sz;
        end
    else
        pi_joint = pi_z_j;
    end

    D3_cells = {};
    if ~isempty(d3_gridvals)
        D3_cells = cell(1, size(d3_gridvals, 2));
        for idx = 1:length(D3_cells)
            D3_cells{idx} = d3_gridvals(i_d3, idx);
        end
    end

    for i_d2 = 1:N_d2
        idx   = reshape(a2primeIndex(i_d2, :, :), [N_a2, N_all_z_safe]);
        probs = reshape(a2primeProbs(i_d2, :, :), [N_a2, N_all_z_safe]);

        a1_col = reshape(1:N_a1, [N_a1, 1, 1]);
        idx_lower_offset = reshape((idx - 1) * N_a1, [1, N_a2, N_all_z_safe]);
        idx_upper_offset = reshape((min(idx + 1, N_a2) - 1) * N_a1, [1, N_a2, N_all_z_safe]);
        z_offset_V = reshape((0:N_all_z_safe-1) * (N_a1 * N_a2), [1, 1, N_all_z_safe]);

        lin_lower = a1_col + idx_lower_offset + z_offset_V;
        lin_upper = a1_col + idx_upper_offset + z_offset_V;
        Vlower = V_next(lin_lower);
        Vupper = V_next(lin_upper);

        probs_full = repmat(reshape(probs, [1, N_a2, N_all_z_safe]), [N_a1, 1, 1]);
        EV_interp = probs_full .* Vlower + (1 - probs_full) .* Vupper;

        mask0 = (probs_full == 0); mask1 = (probs_full == 1);
        EV_interp(mask0) = Vupper(mask0); EV_interp(mask1) = Vlower(mask1);
        EV_interp(isnan(EV_interp)) = -Inf;

        if N_all_z_safe > 1
            EV_flat = reshape(EV_interp, [N_a1 * N_a2, N_all_z_safe]);
            inf_mask = (EV_flat == -Inf);
            EV_safe = EV_flat; EV_safe(inf_mask) = 0;
            EV_d2_full_flat = EV_safe * pi_joint';
            inf_infect = double(inf_mask) * double(pi_joint' > 0);
            EV_d2_full_flat(inf_infect > 0) = -Inf;
            EV_d2_full = reshape(EV_d2_full_flat, [N_a1, N_a2, N_all_z_safe]);
        else
            EV_d2_full = EV_interp;
        end

        D2_cells = cell(1, size(d2_gridvals, 2));
        for i = 1:length(D2_cells)
            D2_cells{i} = d2_gridvals(i_d2, i);
        end
        
        F_tensor = ReturnFn(D1_cells{:}, D2_cells{:}, D3_cells{:}, A1prime_cells{:}, A1_cells_block{:}, A2_cells{:}, Z_cells{:}, SemiZ_cells{:}, ReturnFnParamsVec{:});

        lin_idx = apr_idx_tensor + a2_offset + z_offset;
        EV_d2_bc = reshape(EV_d2_full(lin_idx(:)), size(lin_idx));
        RHS = F_tensor + beta_j .* EV_d2_bc;

        expected_sz = [N_choice, N_d1_safe, N_block, N_a2, N_all_z_safe];
        if ~isequal(size(RHS), expected_sz), RHS = RHS + zeros(expected_sz, 'like', V_next); end
        RHS_flat = reshape(RHS, [N_choice * N_d1_safe, N_block * N_a2 * N_all_z_safe]);
        
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
            loweredge_flat = repmat(reshape(loweredge_matrix, [1, N_a2 * N_all_z_safe]), [N_block, 1]);
            apr_idx_local_2d = reshape(apr_idx_local, [N_block, N_a2 * N_all_z_safe]);
            apr_idx_coarse = loweredge_flat + apr_idx_local_2d - 1;
        end
        apr_idx_coarse = reshape(apr_idx_coarse, [N_block, N_a2, N_all_z_safe]);
        d1_idx_coarse  = reshape(d1_idx_coarse,  [N_block, N_a2, N_all_z_safe]);

        if gridinterplayer
            midpoint = max(min(apr_idx_coarse, N_a1 - 1), 2);
            base_idx = midpoint + (midpoint - 1) * n2short;
            base_idx_tensor = reshape(base_idx, [1, 1, N_block, N_a2, N_all_z_safe]);
            offset_fine = reshape(-n2short-1 : n2short+1, [n2long, 1, 1, 1, 1]);
            fine_idx_tensor = base_idx_tensor + offset_fine;

            A1prime_fine_cells = cell(1, num_a1_vars);
            A1prime_fine_cells{1} = reshape(a1prime_grid(fine_idx_tensor(:)), size(fine_idx_tensor));
            for i_a = 2:num_a1_vars
                a1_col = a1_gridvals(:, i_a);
                coarse_mapping = reshape(a1_col(midpoint(:)), [1, 1, N_block, N_a2, N_all_z_safe]);
                A1prime_fine_cells{i_a} = coarse_mapping + zeros(size(fine_idx_tensor), 'like', a1_gridvals);
            end

            if N_d1 > 0
                D1_fine = cell(size(D1_cells));
                for i_d = 1:numel(D1_cells)
                    D1_val = d1_gridvals(d1_idx_coarse(:), i_d);
                    D1_fine{i_d} = reshape(D1_val, [1, 1, N_block, N_a2, N_all_z_safe]);
                end
            else
                D1_fine = {};
            end

            F_tensor_fine = ReturnFn(D1_fine{:}, D2_cells{:}, D3_cells{:}, A1prime_fine_cells{:}, A1_cells_block{:}, A2_cells{:}, Z_cells{:}, SemiZ_cells{:}, ReturnFnParamsVec{:});

            inf_mask = double(EV_d2_full == -Inf);
            EV_safe = EV_d2_full; EV_safe(EV_d2_full == -Inf) = 0;
            EV_d2_interp = interp1(a1_work_local, reshape(EV_safe, [N_a1, N_a2 * N_all_z_safe]), a1prime_grid);
            inf_interp   = interp1(a1_work_local, reshape(inf_mask, [N_a1, N_a2 * N_all_z_safe]), a1prime_grid);
            EV_d2_interp(inf_interp > 0) = -Inf;

            a2_col = reshape(1:N_a2, [1, 1, 1, N_a2, 1]);
            z_col  = reshape(0:N_all_z_safe-1, [1, 1, 1, 1, N_all_z_safe]) .* N_a2;
            fine_lin_idx = fine_idx_tensor + (a2_col + z_col - 1) .* length(a1prime_grid);
            EV_fine = reshape(EV_d2_interp(fine_lin_idx(:)), size(fine_lin_idx));

            RHS_fine = F_tensor_fine + beta_j .* EV_fine;
            expected_sz_fine = [n2long, 1, N_block, N_a2, N_all_z_safe];
            if ~isequal(size(RHS_fine), expected_sz_fine), RHS_fine = RHS_fine + zeros(expected_sz_fine, 'like', V_next); end
            RHS_fine_flat = reshape(RHS_fine, [n2long, N_block * N_a2 * N_all_z_safe]);
            [V_sub_fine, maxindexL2] = max(RHS_fine_flat, [], 1);

            isInfLower = (RHS_fine_flat(1, :) == -Inf);
            isInfUpper = (RHS_fine_flat(end, :) == -Inf);
            inLowerStrict = (maxindexL2 >= 2) & (maxindexL2 <= n2short + 1);
            inUpperStrict = (maxindexL2 >= n2short + 3) & (maxindexL2 <= n2long - 1);
            L2flag_fine = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

            V_sub   = reshape(V_sub_fine,     [N_block, N_a2, N_all_z_safe]);
            apr_idx = reshape(midpoint,       [N_block, N_a2, N_all_z_safe]);
            d1_idx  = reshape(d1_idx_coarse,  [N_block, N_a2, N_all_z_safe]);
            L2idx   = reshape(maxindexL2,     [N_block, N_a2, N_all_z_safe]);
            L2flag  = reshape(L2flag_fine,    [N_block, N_a2, N_all_z_safe]);
        else
            V_sub   = reshape(V_sub_coarse,   [N_block, N_a2, N_all_z_safe]);
            apr_idx = reshape(apr_idx_coarse, [N_block, N_a2, N_all_z_safe]);
            d1_idx  = reshape(d1_idx_coarse,  [N_block, N_a2, N_all_z_safe]);
        end

        if i_d2 == 1 && i_d3 == 1
            update_mask = true(N_block, N_a2, N_all_z_safe);
        else
            % Match standard toolkit tie-breaking: prefer lower asset choice if values are exactly equal
            update_mask = (V_sub > V_j_max) | ((V_sub == V_j_max) & (apr_idx < Pol_apr_max));
        end
        V_j_max(update_mask)     = V_sub(update_mask);
        Pol_apr_max(update_mask) = apr_idx(update_mask);
        Pol_d1_max(update_mask)  = d1_idx(update_mask);
        Pol_d2_max(update_mask)  = i_d2;
        Pol_d3_max(update_mask)  = i_d3;
        if gridinterplayer
            Pol_L2idx_max(update_mask)  = L2idx(update_mask);
            Pol_L2flag_max(update_mask) = L2flag(update_mask);
        end
    end
end
Pol_d_combo = Pol_d1_max + (Pol_d2_max - 1) * N_d1_safe + (Pol_d3_max - 1) * N_d1_safe * N_d2;
end