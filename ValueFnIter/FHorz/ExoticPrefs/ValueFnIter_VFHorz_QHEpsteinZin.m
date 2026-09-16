function [V1, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_z, N_j, d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% --- 1. Settings & Guardrails ---
if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
    error('V Universe Abort: Divide-and-Conquer assumes policy monotonicity. Quasi-Hyperbolic present-bias causes non-monotonic behavior.');
end

beta0 = Parameters.(vfoptions.QHadditionaldiscount);
isNaive = strcmp(vfoptions.quasi_hyperbolic, 'Naive');
ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4;
ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; ezc8 = vfoptions.ezc8;

% --- 2. Dimension and Grid Setup ---
N_d = max(1, prod(n_d(n_d > 0)));
N_a = max(1, prod(n_a(n_a > 0)));
N_z = max(1, prod(n_z(n_z > 0)));
if N_z == 0; N_z = 1; end

% A matrix setup
num_a = length(n_a(n_a > 0));
a_grids_1d = cell(1, num_a);
offset = 0;
for i_a = 1:num_a
    a_grids_1d{i_a} = a_grid((offset + 1):(offset + n_a(i_a)));
    offset = offset + n_a(i_a);
end
[A_mesh_raw{1:num_a}] = ndgrid(a_grids_1d{:});
A_mat = zeros(N_a, num_a, 'like', a_grid);
for i_a = 1:num_a
    A_mat(:, i_a) = A_mesh_raw{i_a}(:);
end
a_work_local = A_mat(:, 1);

% D_cells setup
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
        for i_d = 1:num_d
            D_cells_block{i_d} = reshape(D_mesh{i_d}(:), [N_d, 1, 1, 1]);
        end
    else
        D_cells_block{1} = reshape(d_grid(:), [N_d, 1, 1, 1]);
    end
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
    ze_chunks = num2cell(1:N_z); % Slice over Z safely
end

% --- 4. Backward Induction Loop ---
for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsCell = CreateCellFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(cell2mat(DiscountFactorParamsCell));
    beta0beta_j = beta0 * beta_j;

    % --- TERMINAL PERIOD ---
    if jj == N_j && ~isfield(vfoptions, 'V_Jplus1')
        % Dummy expected value to safely pass to the TensorBlock
        EV_flat = zeros(N_a * N_z, 1, 'like', a_grid);
    else
        % --- CONTINUATION PERIODS ---
        if jj == N_j
            EV_Source = reshape(vfoptions.V_Jplus1, [N_a, N_z]);
        else
            EV_Source = reshape(Valt(:,:,jj+1), [N_a, N_z]);
        end

        % A. EZ Forward Transformation (Risk Aversion)
        valid_V = isfinite(EV_Source) & (EV_Source ~= 0);
        V_transformed = EV_Source;
        if ezc5(jj) == 1
            V_transformed(valid_V) = ezc4 * EV_Source(valid_V);
        else
            V_transformed(valid_V) = max(ezc4 * EV_Source(valid_V), 0).^ezc5(jj);
        end
        V_transformed(EV_Source == 0) = 0;

        % B. Expected Value Transition
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        EV_Expected = V_transformed * pi_z_j';

        % C. EZ Reverse Transformation (Certainty Equivalent)
        valid_EV = isfinite(EV_Expected) & (EV_Expected ~= 0);
        if ezc6(jj) ~= 1
            EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc6(jj);
        end
        if ezc8(jj) ~= 1
            EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc8(jj);
        end

        EV_flat = reshape(EV_Expected, [N_a * N_z, 1]);
    end

    % --- Slicer Loop (Evaluate TensorBlocks) ---
    V1_j = zeros(N_a, N_z, 'like', a_grid);
    Valt_j = zeros(N_a, N_z, 'like', a_grid);
    Pol_j = zeros(N_a, N_z, 'like', a_grid);
    if isNaive; Polalt_j = zeros(N_a, N_z, 'like', a_grid); end

    for i_ze = 1:length(ze_chunks)
        curr_ze = ze_chunks{i_ze};
        N_ze_local = length(curr_ze);

        % Build localized Z_cells (ignoring 'e' for QEAV implementation)
        Z_cells_local = cell(1, size(z_gridvals_J, 2));
        for iz = 1:size(z_gridvals_J, 2)
            Z_cells_local{iz} = reshape(z_gridvals_J(curr_ze, iz, min(jj, size(z_gridvals_J,3))), [1, 1, 1, N_ze_local]);
        end

        % Extract EV local dependencies
        EV_local = EV_flat( (curr_ze - 1)*N_a + 1 : curr_ze*N_a );

        % Launch the QHEZ TensorBlock
        [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QHEZ_TensorBlock(...
            N_a, N_d, N_ze_local, Z_cells_local, D_cells_block, ...
            beta_j, beta0beta_j, EV_local, a_work_local, ...
            ReturnFn, ReturnFnParamsCell, ezc2(jj), ezc3, ezc4, ezc7(jj), isNaive, jj == N_j && ~isfield(vfoptions, 'V_Jplus1'));

        V1_j(:, curr_ze) = V_hat;
        Valt_j(:, curr_ze) = V_underbar;
        Pol_j(:, curr_ze) = Pol_hat;
        if isNaive; Polalt_j(:, curr_ze) = Pol_alt; end
    end

    % Reshape and assign to master tensors
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
% THE QHEZ TENSOR BLOCK
% =========================================================================
function [V_hat, Pol_hat, V_underbar, Pol_alt] = Evaluate_QHEZ_TensorBlock(...
    N_a, N_d_safe, N_ze_local, Z_cells_block, D_cells_block, ...
    beta_j, beta0beta_j, EV_local, a_work_local, ...
    ReturnFn, ReturnFnParamsCell, ezc2_j, ezc3, ezc4, ezc7_j, isNaive, isTerminal)

% 1. Implicit Dimension Setup
apr_idx_tensor = reshape(1:N_a, [1, N_a, 1, 1]);
apr_in = reshape(a_work_local(apr_idx_tensor(:)), size(apr_idx_tensor));
a_in   = reshape(a_work_local(1:N_a), [1, 1, N_a, 1]);

% 2. Evaluate the Arrayfun Kernel (No massive 5D ndgrid required)
F_tensor = ReturnFn(D_cells_block{:}, apr_in, a_in, Z_cells_block{:}, ReturnFnParamsCell{:});

% 3. Format Expected Values
EV_flat = reshape(EV_local, [N_a * N_ze_local, 1]);
linear_idx = apr_idx_tensor + reshape((0:N_ze_local-1)*N_a, [1, 1, 1, N_ze_local]);
EV_bounded = reshape(EV_flat(linear_idx(:)), size(linear_idx));

% 4. DUAL TRACKING
if isTerminal
    [V_hat, Pol_hat] = max(reshape(F_tensor, [N_d_safe * N_a, N_a * N_ze_local]), [], 1);
    V_underbar = V_hat;
    Pol_alt = Pol_hat;
elseif isNaive
    RHS_alt = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_alt, Pol_alt] = max(reshape(RHS_alt, [N_d_safe * N_a, N_a * N_ze_local]), [], 1);

    RHS_tilde = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta0beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_hat, Pol_hat] = max(reshape(RHS_tilde, [N_d_safe * N_a, N_a * N_ze_local]), [], 1);
    V_underbar = V_alt;
else
    RHS_hat = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta0beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    [V_hat, Pol_hat] = max(reshape(RHS_hat, [N_d_safe * N_a, N_a * N_ze_local]), [], 1);

    RHS_underbar = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);
    RHS_underbar_flat = reshape(RHS_underbar, [N_d_safe * N_a, N_a * N_ze_local]);

    % Extract the realized continuation using the current self's policy
    maxindexfull = Pol_hat + (N_d_safe * N_a) * (0 : N_a * N_ze_local - 1);
    V_underbar = RHS_underbar_flat(maxindexfull);
    Pol_alt = [];
end

% 5. Reshape Outputs
V_hat = reshape(V_hat, [N_a, N_ze_local]);
V_underbar = reshape(V_underbar, [N_a, N_ze_local]);
Pol_hat = reshape(Pol_hat, [N_a, N_ze_local]);
if isNaive; Pol_alt = reshape(Pol_alt, [N_a, N_ze_local]); end


end