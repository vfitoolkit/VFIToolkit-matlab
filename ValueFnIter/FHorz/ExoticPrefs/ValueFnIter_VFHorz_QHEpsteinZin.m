function [V1, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Settings & Guardrails ---
if vfoptions.divideandconquer == 1
    error('V Universe Abort: Divide-and-Conquer assumes policy monotonicity. Quasi-Hyperbolic present-bias causes non-monotonic behavior.');
end

beta0 = Parameters.(vfoptions.QHadditionaldiscount);
isNaive = strcmp(vfoptions.quasi_hyperbolic, 'Naive');
ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;

% --- 2. Dimension and ExpAsset Slicing ---
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

N_d  = max(1, prod(n_d(n_d > 0)));
N_a1 = max(1, prod(n_a1(n_a1 > 0)));
N_a2 = max(1, prod(n_a2(n_a2 > 0)));
N_a  = N_a1 * N_a2;
N_z  = max(1, prod(n_z(n_z > 0)));
if N_z == 0; N_z = 1; end

% --- 2b. Universal Grid Packing ---
a1_grid_len = sum(n_a1);
a1_grid_vals = a_grid(1:a1_grid_len);
a2_grid_vals = a_grid(a1_grid_len+1:end);

% Pack D and A1 (Endogenous)
[TensorReturnFn, D_cells_block, A1_cells, ~, ~] = CreateTensorFnAndCells(ReturnFn, n_d, n_a1, 0, 0, d_grid, a1_grid_vals, [], []);

% Pack A2 (Experience)
[TensoraprimeFn, ~, A2_cells, ~, ~] = CreateTensorFnAndCells(vfoptions.aprimeFn, 0, n_a2, 0, 0, [], a2_grid_vals, [], []);

% Re-construct the legacy A1_mat and A2_mat formats expected by the lower TensorBlock
A1_mat = zeros(N_a1, length(n_a1), 'like', a_grid);
for i_a = 1:length(n_a1)
    A1_mat(:, i_a) = A1_cells{i_a}(:);
end

A2_mat = zeros(N_a2, length(n_a2), 'like', a_grid);
a2_grids_1d = cell(1, length(n_a2));
offset = 0;
for i_a = 1:length(n_a2)
    A2_mat(:, i_a) = A2_cells{i_a}(:);
    a2_grids_1d{i_a} = a2_grid_vals((offset + 1):(offset + n_a2(i_a)));
    offset = offset + n_a2(i_a);
end

% Ensure D_cells_block is formatted for the QHEZ 5D Tensor [N_d, 1, 1, 1, 1]
for i_d = 1:length(D_cells_block)
    D_cells_block{i_d} = reshape(D_cells_block{i_d}, [N_d, 1, 1, 1, 1]);
end

% Extract aprimeFn Params for ExpAsset
if l_a2 > 0
    aprimeFn = vfoptions.aprimeFn;
    if isfield(vfoptions, 'aprimeFnParamNames')
        aprimeFnParamNames = vfoptions.aprimeFnParamNames;
    else
        temp = getAnonymousFnInputNames(aprimeFn);
        num_prefix = length(n_d) + length(n_a2) + length(n_z);
        if length(temp) > num_prefix
            aprimeFnParamNames = {temp{num_prefix+1:end}};
        else
            aprimeFnParamNames = {};
        end
    end
    % --- TENSOR BRIDGE FIX: Filter out state variables misidentified as parameters ---
    aprimeFnParamNames = aprimeFnParamNames(isfield(Parameters, aprimeFnParamNames));
else
    aprimeFn = [];
    aprimeFnParamNames = {};
end

% Pre-allocate Flattened Output Tensors in SYSTEM RAM (CPU) to prevent VRAM overflow
V1 = zeros(N_a, N_z, N_j, vfoptions.precision);
Valt = zeros(N_a, N_z, N_j, vfoptions.precision);
has_GI = vfoptions.gridinterplayer(1) == 1;
if has_GI
    Policy = zeros(3, N_a, N_z, N_j, vfoptions.precision);
    if isNaive; Policyalt = zeros(3, N_a, N_z, N_j, vfoptions.precision); else; Policyalt = cast([],vfoptions.precision); end
else
    Policy = zeros(N_a, N_z, N_j, vfoptions.precision);
    if isNaive; Policyalt = zeros(N_a, N_z, N_j, vfoptions.precision); else; Policyalt = cast([],vfoptions.precision); end
end

% --- 3. Slicer Setup (Multi-Axis) ---
% Determine Z/E Chunking
if ismember(vfoptions.lowmemory, [0, 4])
    ze_chunks = {1:N_z}; % Keep ZE vectorized
elseif vfoptions.lowmemory == 1
    chunk_size = 300; % Safely saturate the RTX 5090!
    num_chunks = ceil(N_z / chunk_size);
    ze_chunks = cell(1, num_chunks);
    for c = 1:num_chunks
        ze_chunks{c} = (c-1)*chunk_size + 1 : min(c*chunk_size, N_z);
    end
else
    ze_chunks = num2cell(1:N_z); % Slice ZE (Chunk size 1)
end

% Determine Experience Asset (A2) Chunking
if ismember(vfoptions.lowmemory, [4, 5]) && l_a2 > 0
    a2_chunks = num2cell(1:N_a2); % Slice A2
else
    a2_chunks = {1:N_a2}; % Keep A2 vectorized
end

% --- 4. Backward Induction Loop ---
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj, vfoptions.precision);
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj, vfoptions.precision);
    beta_j = prod(DiscountFactorParamsVec);
    beta0beta_j = beta0 * beta_j;

    if l_a2 > 0
        aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);
    else
        aprimeFnParamsCell = {};
    end

    % --- TERMINAL PERIOD ---
    if jj == N_j && ~isfield(vfoptions, 'V_Jplus1')
        EV_flat = zeros(N_a * N_z, 1, 'like', a_grid);
    else
        % --- CONTINUATION PERIODS ---
        if jj == N_j
            EV_Source = reshape(gpuArray(vfoptions.V_Jplus1), [N_a, N_z]);
        else
            % Pull the slice back from CPU RAM to the GPU for this period's math
            EV_Source = gpuArray(reshape(Valt(:,:,jj+1), [N_a, N_z]));
        end

        valid_V = isfinite(EV_Source) & (EV_Source ~= 0);
        V_transformed = EV_Source;
        if ezc5(jj) == 1
            V_transformed(valid_V) = ezc4 * EV_Source(valid_V);
        else
            V_transformed(valid_V) = max(ezc4 * EV_Source(valid_V), 0).^ezc5(jj);
        end
        V_transformed(EV_Source == 0) = 0;

        % --- Decoupled, Decision-Dependent Expectations ---
        has_semiz = isfield(vfoptions, 'n_semiz') && prod(vfoptions.n_semiz) > 0;
        N_semiz_local = 1;
        N_dsemiz = 1;
        if has_semiz
            N_semiz_local = max(1, prod(vfoptions.n_semiz));
            if isfield(vfoptions, 'l_dsemiz')
                N_dsemiz = max(1, prod(n_d(end-vfoptions.l_dsemiz+1:end)));
            else
                N_dsemiz = max(1, n_d(end));
            end
        end
        N_z_exog = max(1, N_z / N_semiz_local);

        EV_Expected = zeros(N_a, N_semiz_local * N_z_exog, N_dsemiz, 'like', V_transformed);

        % 1. Exogenous Z Transition (Shared across all decisions)
        if N_z_exog > 1
            pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3))); % Raw pi_z
            V_slice = reshape(V_transformed, [N_a * N_semiz_local, N_z_exog]);
            V_z_eval = V_slice * pi_z_j';
            V_z_eval = reshape(V_z_eval, [N_a, N_semiz_local, N_z_exog]);
        else
            V_z_eval = reshape(V_transformed, [N_a, N_semiz_local, N_z_exog]);
        end

        % 2. Semi-Exogenous Transition (Decision Dependent)
        if has_semiz
            pi_semiz_j = vfoptions.pi_semiz_J(:, :, :, min(jj, size(vfoptions.pi_semiz_J, 4)));
            V_perm = reshape(permute(V_z_eval, [2, 1, 3]), [N_semiz_local, N_a * N_z_exog]);
            for idsemiz = 1:N_dsemiz
                pi_semiz_d = pi_semiz_j(:, :, idsemiz);
                EV_perm = pi_semiz_d * V_perm;
                EV_d = permute(reshape(EV_perm, [N_semiz_local, N_a, N_z_exog]), [2, 1, 3]);
                EV_Expected(:,:,idsemiz) = reshape(EV_d, [N_a, N_semiz_local * N_z_exog]);
            end
        else
            EV_Expected(:,:,1) = reshape(V_z_eval, [N_a, N_semiz_local * N_z_exog]);
        end

        valid_EV = isfinite(EV_Expected) & (EV_Expected ~= 0);
        if ezc6(jj) ~= 1; EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc6(jj); end
        if ezc8(jj) ~= 1; EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc8(jj); end

        EV_flat = reshape(EV_Expected, [N_a * N_z, N_dsemiz]);
    end

    % Allocate GPU tensors for THIS period's slices
    V1_j = zeros(N_a, N_z, 'like', a_grid);
    Valt_j = zeros(N_a, N_z, 'like', a_grid);
    Pol_j = zeros(N_a, N_z, 'like', a_grid);
    if isNaive; Polalt_j = zeros(N_a, N_z, 'like', a_grid); end

    if ~exist('N_dsemiz', 'var'); N_dsemiz = 1; end
    if N_dsemiz > 1
        if isfield(vfoptions, 'l_dsemiz')
            N_d_prefix = max(1, prod(n_d(1:end-vfoptions.l_dsemiz)));
        else
            N_d_prefix = max(1, prod(n_d(1:end-1)));
        end
        dsemiz_idx = ceil((1:N_d)' / N_d_prefix);
        dsemiz_idx_tensor = reshape(dsemiz_idx, [N_d, 1, 1, 1, 1]);
    else
        dsemiz_idx_tensor = ones(N_d, 1, 1, 1, 1);
    end

    for i_a2 = 1:length(a2_chunks)
        curr_a2 = a2_chunks{i_a2};
        N_a2_local = length(curr_a2);

        for i_ze = 1:length(ze_chunks)
            curr_ze = ze_chunks{i_ze};
            N_ze_local = length(curr_ze);

            Z_cells_local = cell(1, size(z_gridvals_J, 2));
            for iz = 1:size(z_gridvals_J, 2)
                Z_cells_local{iz} = reshape(z_gridvals_J(curr_ze, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, 1, N_ze_local]);
            end

            % Slice A2 locally for the current chunk
            A2_local = A2_mat(curr_a2, :);
            N_a2_local = size(A2_local, 1);

            start_idx = (min(curr_ze) - 1) * N_a + 1;
            end_idx   = max(curr_ze) * N_a;
            EV_local  = EV_flat(start_idx : end_idx);

            % Launch the QHEZ Bridge TensorBlock
            [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QHEZ_TensorBlock(...
                N_a1, N_a2_local, N_d, N_ze_local, Z_cells_local, D_cells_block, ...
                A1_mat, A2_local, a2_grids_1d, l_a2, beta_j, beta0beta_j, EV_local, ...
                TensorReturnFn, ReturnFnParamsCell, TensoraprimeFn, aprimeFnParamsCell, ...
                ezc2(jj), ezc3, ezc4, ezc7(jj), isNaive, jj == N_j && ~isfield(vfoptions, 'V_Jplus1'), ...
                N_dsemiz, dsemiz_idx_tensor);

            % Map the local slice back into the global V1_j structure
            start_a_idx = (min(curr_a2) - 1) * N_a1 + 1;
            end_a_idx   = max(curr_a2) * N_a1;

            V1_j( start_a_idx : end_a_idx, curr_ze ) = reshape(V_hat, [N_a1 * N_a2_local, N_ze_local]);
            Valt_j( start_a_idx : end_a_idx, curr_ze ) = reshape(V_underbar, [N_a1 * N_a2_local, N_ze_local]);
            Pol_j( start_a_idx : end_a_idx, curr_ze ) = reshape(Pol_hat, [N_a1 * N_a2_local, N_ze_local]);
            if isNaive
                Polalt_j( start_a_idx : end_a_idx, curr_ze ) = reshape(Pol_alt, [N_a1 * N_a2_local, N_ze_local]);
            end
        end
    end

    % Gather from GPU to System RAM
    V1(:,:,jj) = gather(V1_j);
    Valt(:,:,jj) = gather(Valt_j);
    if has_GI
        Policy(1,:,:,jj) = gather(Pol_j);
        Policy(2,:,:,jj) = 0;
        Policy(3,:,:,jj) = 2;
        if isNaive
            Policyalt(1,:,:,jj) = gather(Polalt_j);
            Policyalt(2,:,:,jj) = 0;
            Policyalt(3,:,:,jj) = 2;
        end
    else
        Policy(:,:,jj) = gather(Pol_j);
        if isNaive; Policyalt(:,:,jj) = gather(Polalt_j); end
    end
end

% --- 5. UnKron Policies and Final Reshape ---
out_dims = [n_a, n_z, N_j];
if isscalar(out_dims); out_dims = [out_dims, 1]; end

V1 = reshape(V1, out_dims);
Valt = reshape(Valt, out_dims);

% 1. Define the choice space that the Kron index spans
if isempty(n_d) || prod(n_d) == 0
    n_daprime = n_a1;
else
    n_daprime = [n_d, n_a1];
end

% 2. Add leading singleton so the UnKron engine recognizes it
if ~has_GI
    Policy = shiftdim(Policy, -1);
    if isNaive; Policyalt = shiftdim(Policyalt, -1); end
end

% 3. Unpack the single index into independent rows for d, a1_1, a1_2...
if N_z > 0
    Policy = UnKronPolicyIndexes1_FHorz_z(Policy, n_daprime, n_a, n_z, N_j, vfoptions);
    if isNaive; Policyalt = UnKronPolicyIndexes1_FHorz_z(Policyalt, n_daprime, n_a, n_z, N_j, vfoptions); end
else
    Policy = UnKronPolicyIndexes1_FHorz_noz(Policy, n_daprime, n_a, N_j, vfoptions);
    if isNaive; Policyalt = UnKronPolicyIndexes1_FHorz_noz(Policyalt, n_daprime, n_a, N_j, vfoptions); end
end

% 4. Final reshape for the downstream StationaryDist engine
NumPol = size(Policy, 1);
Policy = reshape(Policy, [NumPol, out_dims]);
if isNaive; Policyalt = reshape(Policyalt, [NumPol, out_dims]); end


end

% =========================================================================
% THE QHEZ TENSOR-ARRAYFUN BRIDGE
% =========================================================================
function [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QHEZ_TensorBlock(...
    N_a1, N_a2, N_d_safe, N_ze_local, Z_cells_block, D_cells_block, ...
    A1_mat, A2_mat, a2_grids_1d, l_a2, beta_j, beta0beta_j, EV_local, ...
    TensorReturnFn, ReturnFnParamsCell, TensoraprimeFn, aprimeFnParamsCell, ...
    ezc2_j, ezc3, ezc4, ezc7_j, isNaive, isTerminal, N_dsemiz, dsemiz_idx_tensor)

% 1. Build A1 and A2 Cells dynamically
num_a1 = size(A1_mat, 2);
Apr_cells = cell(1, num_a1);
A1_cells  = cell(1, num_a1);
for ia = 1:num_a1
    Apr_cells{ia} = reshape(A1_mat(:,ia), [1, N_a1, 1, 1, 1]);
    A1_cells{ia}  = reshape(A1_mat(:,ia), [1, 1, N_a1, 1, 1]);
end

% 2. Evaluate ReturnFn with raw numeric arrays for A2
if l_a2 > 0
    N_a2_local = size(A2_mat, 1);
    num_a2 = size(A2_mat, 2);
    A2_cells = cell(1, num_a2);
    for ia = 1:num_a2
        A2_cells{ia} = reshape(A2_mat(:,ia), [1, 1, 1, N_a2_local, 1]);
    end
    F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, ReturnFnParamsCell{:});
else
    F_tensor = TensorReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, ReturnFnParamsCell{:});
end

% 3. Format Expected Values (EV_bounded)
if l_a2 > 0
    % ExpAsset Transition Interpolation
    A2_prime = TensoraprimeFn(D_cells_block{:}, A2_cells{:}, Z_cells_block{:}, aprimeFnParamsCell{:});
    a2_grid_1d_vec = a2_grids_1d{1};
    a2_min = a2_grid_1d_vec(1);
    a2_max = a2_grid_1d_vec(end);
    a2_prime_clipped = max(a2_min, min(A2_prime, a2_max));
    idx = discretize(a2_prime_clipped, a2_grid_1d_vec);
    idx(isnan(idx)) = N_a2 - 1;
    idx = max(1, min(idx, N_a2 - 1));
    a2_left = reshape(a2_grid_1d_vec(idx), size(idx));
    a2_right = reshape(a2_grid_1d_vec(idx+1), size(idx));
    weight = (a2_prime_clipped - a2_left) ./ (a2_right - a2_left);
    weight(a2_right == a2_left) = 0;

    A1pr_idx = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
    ZE_idx   = reshape(1:N_ze_local, [1, 1, 1, 1, N_ze_local]);
    
    % --- CORRECTED MULTI-SHOCK INDEXING OFFSET ---
    idx_left  = A1pr_idx + (idx - 1) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);
    idx_right = A1pr_idx + (idx) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);

    max_idx_row = size(EV_local, 1); % Dynamically match EV_local dimensions
    linear_idx_left  = min(max_idx_row, max(1, idx_left  + (dsemiz_idx_tensor - 1) * max_idx_row));
    linear_idx_right = min(max_idx_row, max(1, idx_right + (dsemiz_idx_tensor - 1) * max_idx_row));

    EV_left  = EV_local(linear_idx_left);
    EV_right = EV_local(linear_idx_right);
    EV_bounded = EV_left + weight .* (EV_right - EV_left);
else
    % Standard Endogenous
    apr_idx_tensor = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
    ZE_idx = reshape(0:N_ze_local-1, [1, 1, 1, 1, N_ze_local]);
    idx_base = apr_idx_tensor + ZE_idx * N_a1;

    max_idx_row = N_a1 * N_ze_local;
    linear_idx = idx_base + (dsemiz_idx_tensor - 1) * max_idx_row;

    EV_bounded = reshape(EV_local(linear_idx(:)), size(linear_idx));
end

% 4. DUAL TRACKING
FLAT_CHOICES = max(1, N_d_safe) * N_a1;
FLAT_STATES  = N_a1 * N_a2 * N_ze_local;

if isTerminal
    [V_hat, Pol_hat] = max(reshape(F_tensor, [FLAT_CHOICES, FLAT_STATES]), [], 1);
    V_underbar = V_hat;
    Pol_alt = Pol_hat;
elseif isNaive
    RHS_alt = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_alt, Pol_alt] = max(reshape(RHS_alt, [FLAT_CHOICES, FLAT_STATES]), [], 1);

    RHS_tilde = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta0beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_hat, Pol_hat] = max(reshape(RHS_tilde, [FLAT_CHOICES, FLAT_STATES]), [], 1);
    V_underbar = V_alt;
else
    RHS_hat = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta0beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_hat, Pol_hat] = max(reshape(RHS_hat, [FLAT_CHOICES, FLAT_STATES]), [], 1);

    RHS_underbar = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    RHS_underbar_flat = reshape(RHS_underbar, [FLAT_CHOICES, FLAT_STATES]);

    maxindexfull = Pol_hat + FLAT_CHOICES * (0 : FLAT_STATES - 1);
    V_underbar = RHS_underbar_flat(maxindexfull);
    Pol_alt = [];
end

% 5. Reshape Outputs
V_hat = reshape(V_hat, [N_a1 * N_a2, N_ze_local]);
V_underbar = reshape(V_underbar, [N_a1 * N_a2, N_ze_local]);
Pol_hat = reshape(Pol_hat, [N_a1 * N_a2, N_ze_local]);
if isNaive; Pol_alt = reshape(Pol_alt, [N_a1 * N_a2, N_ze_local]); end


end