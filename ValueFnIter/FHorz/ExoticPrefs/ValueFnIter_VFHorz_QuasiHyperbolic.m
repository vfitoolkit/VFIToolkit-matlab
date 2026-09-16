function [V1, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QuasiHyperbolic(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Settings & Guardrails ---
if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
    error('V Universe Abort: Divide-and-Conquer assumes policy monotonicity. Quasi-Hyperbolic present-bias causes non-monotonic behavior.');
end
beta0 = Parameters.(vfoptions.QHadditionaldiscount);
isNaive = strcmp(vfoptions.quasi_hyperbolic, 'Naive');

% --- 2. Dimension and ExpAsset Slicing ---
l_a2 = 0;
if isfield(vfoptions, 'experienceasset') && vfoptions.experienceasset > 0
    l_a2 = vfoptions.experienceasset;
end
if isfield(vfoptions, 'experienceassetz') && vfoptions.experienceassetz > 0
    l_a2 = vfoptions.experienceassetz;
end

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
[D_cells_block, A1_cells, ~, ~] = CreateReturnFnMatrix_VFHorz(n_d, n_a1, 0, 0, d_grid, a1_grid_vals, [], []);

% Pack A2 (Experience)
[~, A2_cells, ~, ~] = CreateReturnFnMatrix_VFHorz(0, n_a2, 0, 0, [], a2_grid_vals, [], []);

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

% Ensure D_cells_block is formatted for the 5D Tensor [N_d, 1, 1, 1, 1]
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
else
    aprimeFn = [];
    aprimeFnParamNames = {};
end

% Pre-allocate Flattened Output Tensors
V1 = zeros(N_a, N_z, N_j, 'gpuArray');
Valt = zeros(N_a, N_z, N_j, 'gpuArray');

has_GI = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;
if has_GI
    Policy = zeros(3, N_a, N_z, N_j, 'gpuArray');
    if isNaive; Policyalt = zeros(3, N_a, N_z, N_j, 'gpuArray'); else; Policyalt = []; end
else
    Policy = zeros(N_a, N_z, N_j, 'gpuArray');
    if isNaive; Policyalt = zeros(N_a, N_z, N_j, 'gpuArray'); else; Policyalt = []; end
end

% --- 3. Slicer Setup (Multi-Axis) ---
% Determine Z/E Chunking
if ismember(vfoptions.lowmemory, [0, 4])
    ze_chunks = {1:N_z}; % Keep ZE vectorized
else
    ze_chunks = num2cell(1:N_z); % Slice ZE
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
    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsCell = CreateCellFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(cell2mat(DiscountFactorParamsCell));
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
            EV_Source = reshape(vfoptions.V_Jplus1, [N_a, N_z]);
        else
            EV_Source = reshape(Valt(:,:,jj+1), [N_a, N_z]);
        end

        % Standard Expected Value (No EZ Transformations)
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        EV_Expected = EV_Source * pi_z_j';
        EV_flat = reshape(EV_Expected, [N_a * N_z, 1]);
    end

    V1_j = zeros(N_a, N_z, 'like', a_grid);
    Valt_j = zeros(N_a, N_z, 'like', a_grid);
    Pol_j = zeros(N_a, N_z, 'like', a_grid);
    if isNaive; Polalt_j = zeros(N_a, N_z, 'like', a_grid); end

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

            start_idx = (min(curr_ze) - 1) * N_a + 1;
            end_idx   = max(curr_ze) * N_a;
            EV_local  = EV_flat(start_idx : end_idx);

            % Launch the QH Bridge TensorBlock
            [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QH_TensorBlock(...
                N_a1, N_a2, N_d, N_ze_local, Z_cells_local, D_cells_block, ...
                A1_mat, A2_mat, a2_grids_1d, l_a2, beta_j, beta0beta_j, EV_local, ...
                ReturnFn, ReturnFnParamsCell, aprimeFn, aprimeFnParamsCell, ...
                isNaive, jj == N_j && ~isfield(vfoptions, 'V_Jplus1'));

            V1_j(:, curr_ze) = V_hat;
            Valt_j(:, curr_ze) = V_underbar;
            Pol_j(:, curr_ze) = Pol_hat;
            if isNaive; Polalt_j(:, curr_ze) = Pol_alt; end
        end
    end

    V1(:,:,jj) = V1_j;
    Valt(:,:,jj) = Valt_j;

    if has_GI
        Policy(1,:,:,jj) = Pol_j;
        Policy(2,:,:,jj) = 0;
        Policy(3,:,:,jj) = 2;
        if isNaive
            Policyalt(1,:,:,jj) = Polalt_j;
            Policyalt(2,:,:,jj) = 0;
            Policyalt(3,:,:,jj) = 2;
        end
    else
        Policy(:,:,jj) = Pol_j;
        if isNaive; Policyalt(:,:,jj) = Polalt_j; end
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
% THE STANDARD QH TENSOR-ARRAYFUN BRIDGE
% =========================================================================
function [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QH_TensorBlock(...
    N_a1, N_a2, N_d_safe, N_ze_local, Z_cells_block, D_cells_block, ...
    A1_mat, A2_mat, a2_grids_1d, l_a2, beta_j, beta0beta_j, EV_local, ...
    ReturnFn, ReturnFnParamsCell, aprimeFn, aprimeFnParamsCell, ...
    isNaive, isTerminal)

% 1. Build A1 and A2 Cells dynamically
num_a1 = size(A1_mat, 2);
Apr_cells = cell(1, num_a1);
A1_cells  = cell(1, num_a1);
for ia = 1:num_a1
    Apr_cells{ia} = reshape(A1_mat(:,ia), [1, N_a1, 1, 1, 1]);
    A1_cells{ia}  = reshape(A1_mat(:,ia), [1, 1, N_a1, 1, 1]);
end

num_a2 = size(A2_mat, 2);
A2_cells = cell(1, num_a2);
for ia = 1:num_a2
    A2_cells{ia} = reshape(A2_mat(:,ia), [1, 1, 1, N_a2, 1]);
end

% 2. Evaluate ReturnFn (Unpacks EXACTLY what arrayfun needs)
if l_a2 > 0
    F_tensor = ReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, A2_cells{:}, Z_cells_block{:}, ReturnFnParamsCell{:});
else
    F_tensor = ReturnFn(D_cells_block{:}, Apr_cells{:}, A1_cells{:}, Z_cells_block{:}, ReturnFnParamsCell{:});
end

% 3. Format Expected Values (EV_bounded)
if l_a2 > 0
    % ExpAsset Transition Interpolation
    A2_prime = aprimeFn(D_cells_block{:}, A2_cells{:}, Z_cells_block{:}, aprimeFnParamsCell{:});
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

    A1pr_idx = reshape(1:N_a1,     [1, N_a1, 1, 1, 1]);
    ZE_idx   = reshape(1:N_ze_local, [1, 1, 1, 1, N_ze_local]);

    idx_left  = A1pr_idx + (idx - 1) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);
    idx_right = A1pr_idx + (idx) * N_a1     + (ZE_idx - 1) * (N_a1 * N_a2);

    EV_left  = EV_local(idx_left);
    EV_right = EV_local(idx_right);

    EV_bounded = EV_left + weight .* (EV_right - EV_left);
else
    % Standard Endogenous
    EV_flat = reshape(EV_local, [N_a1 * N_ze_local, 1]);
    apr_idx_tensor = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
    ZE_idx = reshape(0:N_ze_local-1, [1, 1, 1, 1, N_ze_local]);
    linear_idx = apr_idx_tensor + ZE_idx * N_a1;
    EV_bounded = reshape(EV_flat(linear_idx(:)), size(linear_idx));
end

% 4. DUAL TRACKING (Standard CRRA Bellman)
FLAT_CHOICES = max(1, N_d_safe) * N_a1;
FLAT_STATES  = N_a1 * N_a2 * N_ze_local;

if isTerminal
    [V_hat, Pol_hat] = max(reshape(F_tensor, [FLAT_CHOICES, FLAT_STATES]), [], 1);
    V_underbar = V_hat;
    Pol_alt = Pol_hat;
elseif isNaive
    RHS_alt = F_tensor + beta_j * EV_bounded;
    [V_alt, Pol_alt] = max(reshape(RHS_alt, [FLAT_CHOICES, FLAT_STATES]), [], 1);

    RHS_tilde = F_tensor + beta0beta_j * EV_bounded;
    [V_hat, Pol_hat] = max(reshape(RHS_tilde, [FLAT_CHOICES, FLAT_STATES]), [], 1);

    V_underbar = V_alt;
else
    RHS_hat = F_tensor + beta0beta_j * EV_bounded;
    [V_hat, Pol_hat] = max(reshape(RHS_hat, [FLAT_CHOICES, FLAT_STATES]), [], 1);

    RHS_underbar_flat = reshape(F_tensor + beta_j * EV_bounded, [FLAT_CHOICES, FLAT_STATES]);
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