function [V1, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Settings & Guardrails ---
if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
    error('V Universe Abort: Divide-and-Conquer assumes policy monotonicity. Quasi-Hyperbolic present-bias causes non-monotonic behavior.');
end

beta0 = Parameters.(vfoptions.QHadditionaldiscount);
isNaive = strcmp(vfoptions.quasi_hyperbolic, 'Naive');
ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;

% --- 2. Dimension and ExpAsset Slicing ---
l_a2 = 0;
if isfield(vfoptions, 'experienceasset') && vfoptions.experienceasset > 0; l_a2 = vfoptions.experienceasset; end
if isfield(vfoptions, 'experienceassetz') && vfoptions.experienceassetz > 0; l_a2 = vfoptions.experienceassetz; end

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

% A1 grids
num_a1 = length(n_a1(n_a1 > 0));
a1_grids_1d = cell(1, num_a1);
offset = 0;
for i_a = 1:num_a1
    a1_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a1(i_a)));
    offset = offset + n_a1(i_a);
end
if num_a1 > 0
    [A1_mesh{1:num_a1}] = ndgrid(a1_grids_1d{:});
    A1_mat = zeros(N_a1, num_a1, 'like', a_grid);
    for i_a = 1:num_a1; A1_mat(:, i_a) = A1_mesh{i_a}(:); end
else
    A1_mat = [];
end

% A2 grids
num_a2 = length(n_a2(n_a2 > 0));
a2_grids_1d = cell(1, num_a2);
for i_a = 1:num_a2
    a2_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a2(i_a)));
    offset = offset + n_a2(i_a);
end
if num_a2 > 0
    [A2_mesh{1:num_a2}] = ndgrid(a2_grids_1d{:});
    A2_mat = zeros(N_a2, num_a2, 'like', a_grid);
    for i_a = 1:num_a2; A2_mat(:, i_a) = A2_mesh{i_a}(:); end
else
    A2_mat = [];
end

% D grids
num_d = length(n_d(n_d > 0));
D_cells_block = cell(1, num_d);
if num_d > 0
    d_grids_1d = cell(1, num_d);
    offset = 0;
    for i_d = 1:num_d
        d_grids_1d{i_d} = d_grid((offset + 1):(offset + n_d(i_d)));
        offset = offset + n_d(i_d);
    end
    if num_d > 1
        [D_mesh{1:num_d}] = ndgrid(d_grids_1d{:});
        for i_d = 1:num_d; D_cells_block{i_d} = reshape(D_mesh{i_d}(:), [N_d, 1, 1, 1, 1]); end
    else
        D_cells_block{1} = reshape(d_grid(:), [N_d, 1, 1, 1, 1]);
    end
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

% Pre-allocate Output Tensors
state_dims = [n_a, n_z];
if isscalar(state_dims); state_dims = [state_dims, 1]; end

V1 = zeros([state_dims, N_j], 'gpuArray');
Valt = zeros([state_dims, N_j], 'gpuArray');

has_GI = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;
if has_GI
    Policy = zeros([3, state_dims, N_j], 'gpuArray');
    if isNaive; Policyalt = zeros([3, state_dims, N_j], 'gpuArray'); else; Policyalt = []; end
else
    Policy = zeros([state_dims, N_j], 'gpuArray');
    if isNaive; Policyalt = zeros([state_dims, N_j], 'gpuArray'); else; Policyalt = []; end
end

% --- 3. Slicer Setup ---
if vfoptions.lowmemory == 0
    ze_chunks = {1:N_z};
else
    ze_chunks = num2cell(1:N_z);
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

        valid_V = isfinite(EV_Source) & (EV_Source ~= 0);
        V_transformed = EV_Source;
        if ezc5(jj) == 1
            V_transformed(valid_V) = ezc4 * EV_Source(valid_V);
        else
            V_transformed(valid_V) = max(ezc4 * EV_Source(valid_V), 0).^ezc5(jj);
        end
        V_transformed(EV_Source == 0) = 0;

        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        EV_Expected = V_transformed * pi_z_j';

        valid_EV = isfinite(EV_Expected) & (EV_Expected ~= 0);
        if ezc6(jj) ~= 1; EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc6(jj); end
        if ezc8(jj) ~= 1; EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc8(jj); end

        EV_flat = reshape(EV_Expected, [N_a * N_z, 1]);
    end

    V1_j = zeros(N_a, N_z, 'like', a_grid);
    Valt_j = zeros(N_a, N_z, 'like', a_grid);
    Pol_j = zeros(N_a, N_z, 'like', a_grid);
    if isNaive; Polalt_j = zeros(N_a, N_z, 'like', a_grid); end

    for i_ze = 1:length(ze_chunks)
        curr_ze = ze_chunks{i_ze};
        N_ze_local = length(curr_ze);

        Z_cells_local = cell(1, size(z_gridvals_J, 2));
        for iz = 1:size(z_gridvals_J, 2)
            Z_cells_local{iz} = reshape(z_gridvals_J(curr_ze, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, 1, N_ze_local]);
        end

        EV_local = EV_flat( (curr_ze - 1)*N_a + 1 : curr_ze*N_a );

        % Launch the QHEZ Bridge TensorBlock
        [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QHEZ_TensorBlock(...
            N_a1, N_a2, N_d, N_ze_local, Z_cells_local, D_cells_block, ...
            A1_mat, A2_mat, a2_grids_1d, l_a2, beta_j, beta0beta_j, EV_local, ...
            ReturnFn, ReturnFnParamsCell, aprimeFn, aprimeFnParamsCell, ...
            ezc2(jj), ezc3, ezc4, ezc7(jj), isNaive, jj == N_j && ~isfield(vfoptions, 'V_Jplus1'));

        V1_j(:, curr_ze) = V_hat;
        Valt_j(:, curr_ze) = V_underbar;
        Pol_j(:, curr_ze) = Pol_hat;
        if isNaive; Polalt_j(:, curr_ze) = Pol_alt; end
    end

    V1(:,:,jj) = reshape(V1_j, state_dims);
    Valt(:,:,jj) = reshape(Valt_j, state_dims);

    if has_GI
        Policy(1,:,:,jj) = reshape(Pol_j, state_dims);
        Policy(2,:,:,jj) = 0;
        Policy(3,:,:,jj) = 2;
        if isNaive
            Policyalt(1,:,:,jj) = reshape(Polalt_j, state_dims);
            Policyalt(2,:,:,jj) = 0;
            Policyalt(3,:,:,jj) = 2;
        end
    else
        Policy(:,:,jj) = reshape(Pol_j, state_dims);
        if isNaive; Policyalt(:,:,jj) = reshape(Polalt_j, state_dims); end
    end
end
end

% =========================================================================
% THE QHEZ TENSOR-ARRAYFUN BRIDGE
% =========================================================================
function [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QHEZ_TensorBlock(...
    N_a1, N_a2, N_d_safe, N_ze_local, Z_cells_block, D_cells_block, ...
    A1_mat, A2_mat, a2_grids_1d, l_a2, beta_j, beta0beta_j, EV_local, ...
    ReturnFn, ReturnFnParamsCell, aprimeFn, aprimeFnParamsCell, ...
    ezc2_j, ezc3, ezc4, ezc7_j, isNaive, isTerminal)

% 1. Build A1 and A2 Cells dynamically
num_a1 = size(A1_mat, 2);
Apr_cells = cell(1, num_a1);
A1_cells = cell(1, num_a1);
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

    A1pr_idx = reshape(1:N_a1, [1, N_a1, 1, 1, 1]);
    ZE_idx   = reshape(1:N_ze_local, [1, 1, 1, 1, N_ze_local]);

    idx_left  = A1pr_idx + (idx - 1) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);
    idx_right = A1pr_idx + (idx) * N_a1 + (ZE_idx - 1) * (N_a1 * N_a2);

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