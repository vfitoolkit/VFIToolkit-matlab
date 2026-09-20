function [V, Policy] = ValueFnIter_VFHorz_RiskyAsset_EpsteinZin(n_d, n_a1, n_a2, n_z, n_u, N_j, ...
    d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ...
    ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ...
    ReturnFnParamNames, aprimeFnParamNames, vfoptions, ...
    sj, warmglow, ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8)

% 1. Extract Dimensions and Grids
has_a1 = ~isempty(n_a1) && prod(n_a1) > 0;
N_a1 = max(prod(n_a1), 1);
N_a2 = max(prod(n_a2), 1);
N_a  = N_a1 * N_a2;
N_z  = prod(n_z);
N_u  = prod(n_u);

n_d1 = 0; N_d1 = 1;
if vfoptions.refine_d(1) > 0
    n_d1 = n_d(1:vfoptions.refine_d(1));
    N_d1 = prod(n_d1);
end
has_d1 = (N_d1 > 1) || (N_d1 == 1 && length(d_grid) >= sum(n_d1) && d_grid(1) ~= 0);

n_d2 = n_d(vfoptions.refine_d(1)+1 : vfoptions.refine_d(1)+vfoptions.refine_d(2));
N_d2 = prod(n_d2);
n_d3 = n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1 : end);
N_d3 = prod(n_d3);

d1_grid = d_grid(1 : sum(n_d1));
d2_grid = d_grid(sum(n_d1)+1 : sum(n_d1)+sum(n_d2));
d3_grid = d_grid(sum(n_d1)+sum(n_d2)+1 : end);

% 2. Push Variables to GPU
if isempty(d1_grid), d1_grid = gpuArray(0); else, d1_grid = gpuArray(d1_grid(:)); end
if isempty(a1_grid), a1_grid = gpuArray(0); else, a1_grid = gpuArray(a1_grid(:)); end
a2_grid = gpuArray(a2_grid(:));
d2_grid = gpuArray(d2_grid(:));
d3_grid = gpuArray(d3_grid(:));
u_grid  = gpuArray(u_grid(:));
pi_u    = gpuArray(pi_u(:));
pi_z_J  = gpuArray(pi_z_J);
z_gridvals = gpuArray(z_gridvals_J);

V = zeros(N_a, N_z, N_j, 'like', a2_grid);
PolicyKron = zeros(1, N_a, N_z, N_j, 'like', a2_grid);
V_next = zeros(N_a, N_z, 'like', a2_grid);

D2_3D = reshape(d2_grid, [N_d2, 1, 1]);
D3_3D = reshape(d3_grid, [1, N_d3, 1]);
U_3D  = reshape(u_grid,  [1, 1, N_u]);

ezc9 = 1;
if isfield(vfoptions, 'ezc9')
    ezc9 = vfoptions.ezc9;
end

if isempty(aprimeFnParamNames)
    if isfield(vfoptions, 'aprimeFnParamNames')
        aprimeFnParamNames = vfoptions.aprimeFnParamNames;
    else
        temp = getAnonymousFnInputNames(aprimeFn);
        num_d2_vars = length(n_d2); if isequal(n_d2, 0) || isempty(n_d2); num_d2_vars = 0; end
        num_d3_vars = length(n_d3); if isequal(n_d3, 0) || isempty(n_d3); num_d3_vars = 0; end
        num_prefix = num_d2_vars + num_d3_vars + 1; % +1 for the 'u' shock
        if length(temp) > num_prefix
            aprimeFnParamNames = {temp{num_prefix + 1 : end}};
            aprimeFnParamNames = aprimeFnParamNames(isfield(Parameters, aprimeFnParamNames));
        end
    end
end

TensorReturnFn = CreateTensorBridge(ReturnFn);
TensorAprimeFn = CreateTensorBridge(aprimeFn);

if warmglow == 1
    TensorWG_Fn = CreateTensorBridge(vfoptions.WarmGlowBequestsFn);
end

% =========================================================
% TIME LOOP
% =========================================================
for jj = N_j : -1 : 1
    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);
    aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);

    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);

    ezc1_j = 1;
    if isfield(vfoptions, 'EZoneminusbeta')
        if vfoptions.EZoneminusbeta == 1; ezc1_j = 1 - beta_j;
        elseif vfoptions.EZoneminusbeta == 2; ezc1_j = 1 - sj(jj) * beta_j;
        end
    end

    aprime_tensor = TensorAprimeFn(D2_3D, D3_3D, U_3D, aprimeFnParamsCell{:});

    % --- EXACT LEGACY INTERPOLATION PRE-COMPUTATION ---
    a2_prime_clipped = max(min(aprime_tensor, a2_grid(end)), a2_grid(1));
    idx = discretize(a2_prime_clipped, a2_grid);
    idx(isnan(idx) | idx == N_a2) = N_a2 - 1;

    a2_left = a2_grid(idx);
    a2_right = a2_grid(idx+1);

    aprimeProbs = (a2_right - a2_prime_clipped) ./ (a2_right - a2_left);
    aprimeProbs(a2_right == a2_left) = 0;

    idx_2D = reshape(idx, [N_d2*N_d3, N_u]);
    aprimeProbs_2D = reshape(aprimeProbs, [N_d2*N_d3, N_u]);
    pi_u_row = reshape(pi_u, [1, N_u]);

    % ---------------------------------------------------------
    % Warm Glow of Bequests
    % ---------------------------------------------------------
    if warmglow == 1
        WG_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
        WG_raw = TensorWG_Fn(a2_grid, WG_params{:});
        if isscalar(WG_raw)
            WG_raw = WG_raw * ones(size(a2_grid), 'like', a2_grid);
        end
        WG_raw = reshape(WG_raw, size(a2_grid));

        WG_temp = WG_raw;
        valid_wg = isfinite(WG_raw);
        WG_temp(valid_wg) = (ezc4 * WG_raw(valid_wg)) .^ ezc5(jj);
        WG_temp(WG_raw == 0) = 0;

        % Exact bit-for-bit Legacy WG Interpolation
        skipinterp = (WG_temp(idx) == WG_temp(idx+1));
        WG_probs = aprimeProbs;
        WG_probs(skipinterp) = 0;

        WG1 = WG_temp(idx) .* WG_probs;
        WG2 = WG_temp(idx+1) .* (1 - WG_probs);

        WG1(isnan(WG1)) = 0;
        WG2(isnan(WG2)) = 0;

        pi_u_3D = reshape(pi_u, [1, 1, N_u]);
        WG1_u = WG1 .* pi_u_3D;
        WG2_u = WG2 .* pi_u_3D;

        WG_u = sum(WG1_u, 3) + sum(WG2_u, 3);
        WG_u = reshape(WG_u, [N_d2*N_d3, 1]);
    else
        WG_u = 0;
    end

    % ---------------------------------------------------------
    % Multi-State Interpolation & Expectations
    % ---------------------------------------------------------
    if jj == N_j
        if warmglow == 1
            temp_WG = WG_u;
            temp_WG(isfinite(WG_u)) = ( (1 - sj(jj)) * WG_u(isfinite(WG_u)).^ezc8(jj) ) .^ ezc6(jj);
            temp_WG(WG_u == 0) = 0;
            temp4 = temp_WG;
        else
            temp4 = zeros(N_d2*N_d3, 1, 'like', a2_grid);
        end
        temp4 = repmat(reshape(temp4, [1, N_d2*N_d3, 1]), [N_a1, 1, max(N_z,1)]);
    else
        % --- Z-EXPECTATION FIRST (Bit-for-Bit match) ---
        temp_V = reshape(V_next, [N_a, max(N_z,1)]);
        temp_V(isfinite(temp_V)) = (ezc4 * temp_V(isfinite(temp_V))) .^ ezc5(jj);
        temp_V(temp_V == 0) = 0;

        if N_z > 0
            pi_z_j = pi_z_J(:,:,jj);
            EV_z_raw = temp_V .* shiftdim(pi_z_j', -1); % [N_a, N_z, N_z]
            EV_z_raw(isnan(EV_z_raw)) = 0;
            EV_z_sum = sum(EV_z_raw, 2); % [N_a, 1, N_z]
            EV_pre_z = reshape(EV_z_sum, [N_a1, N_a2, N_z]);
        else
            EV_pre_z = reshape(temp_V, [N_a1, N_a2, 1]);
        end

        % --- EXACT LEGACY INTERPOLATION AND U-EXPECTATION ---
        EV_z = zeros(N_a1, N_d2*N_d3, max(N_z,1), 'like', a2_grid);
        z_offset = reshape((0:max(N_z,1)-1) * N_a2, [1, 1, max(N_z,1)]);
        pi_u_3D = repmat(pi_u_row, [N_d2*N_d3, 1, max(N_z,1)]);

        for i_a1 = 1:N_a1
            V_slice = squeeze(EV_pre_z(i_a1, :, :));
            if max(N_z,1) == 1, V_slice = V_slice(:); end

            % Generate linear indices dynamically mapped to Z dimension
            linear_idx_left = repmat(idx_2D, [1, 1, max(N_z,1)]) + z_offset;
            linear_idx_right = repmat(idx_2D + 1, [1, 1, max(N_z,1)]) + z_offset;

            V_left = V_slice(linear_idx_left);
            V_right = V_slice(linear_idx_right);

            % Legacy applies skipinterp exact value matching to EV
            skipinterp = (V_left == V_right);
            prob_lower = repmat(aprimeProbs_2D, [1, 1, max(N_z,1)]);
            prob_lower(skipinterp) = 0;
            prob_upper = 1 - prob_lower;

            EV1 = (V_left .* prob_lower) .* pi_u_3D;
            EV2 = (V_right .* prob_upper) .* pi_u_3D;

            EV1(isnan(EV1)) = 0;
            EV2(isnan(EV2)) = 0;

            EV_z(i_a1, :, :) = sum(EV1, 2) + sum(EV2, 2);
        end

        temp4 = EV_z;
        if warmglow == 1
            WG_u_rs = reshape(WG_u, [1, N_d2*N_d3, 1]);
            WG_u_expanded = repmat(WG_u_rs, [N_a1, 1, max(N_z,1)]);
            becareful = logical(isfinite(temp4) .* isfinite(WG_u_expanded));
            temp4(becareful) = ( sj(jj)*temp4(becareful).^ezc8(jj) + (1-sj(jj))*WG_u_expanded(becareful).^ezc8(jj) ) .^ ezc6(jj);
            temp4((EV_z == 0) & (WG_u_expanded == 0)) = 0;
        else
            becareful = isfinite(temp4);
            temp4(becareful) = ( sj(jj)*temp4(becareful).^ezc8(jj) ) .^ ezc6(jj);
            temp4(EV_z == 0) = 0;
        end
    end

    % ---------------------------------------------------------
    % DIMENSIONAL COMPRESSION: Maximize out d2 (riskyshare)
    % ---------------------------------------------------------
    temp4_tensor = reshape(temp4, [N_a1, N_d2, N_d3, max(N_z,1)]);

    safe_temp4 = temp4_tensor;
    inf_mask = isinf(temp4_tensor);
    safe_temp4(inf_mask) = 0;

    masked_temp4 = (~inf_mask) .* safe_temp4;
    flipped_temp4 = ezc9 * ezc3 * masked_temp4;
    flipped_temp4(inf_mask) = -Inf;

    % Exact Math Extraction
    [EV_max_d3_raw, Pol_d2_idx] = max(flipped_temp4, [], 2);

    EV_max_d3 = reshape(EV_max_d3_raw, [N_a1, N_d3, max(N_z,1)]);
    Pol_d2_idx = reshape(Pol_d2_idx, [N_a1, N_d3, max(N_z,1)]);

    % =========================================================
    % 5D TENSOR BLOCK
    % =========================================================
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_EZ_TensorBlock(...
        state_idx, N_d1, N_d2, N_d3, N_a1, N_a2, max(N_z,1), ...
        beta_j, EV_max_d3, Pol_d2_idx, d1_grid, d3_grid, a1_grid, a2_grid, ...
        z_gridvals(:,:,jj), TensorReturnFn, ReturnFnParamsCell, ...
        ezc1_j, ezc2(jj), ezc7(jj), ezc4, ezc9, has_d1, has_a1);

    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        vfopts_dc = vfoptions;
        vfopts_dc.level1n = vfoptions.level1n(1);
        [V_j_max, Pol_d_combo] = ValueFnIter_DC1_Slicer(N_a, N_a, 1, max(N_z,1), vfopts_dc, EvalBlockFn);
    else
        [V_j_max, Pol_d_combo] = EvalBlockFn(1:N_a, [], 0);
    end

    V(:,:,jj) = V_j_max;
    PolicyKron(1, :, :, jj) = Pol_d_combo;
    V_next = V(:,:,jj);
end

% =========================================================
% UNPACK POLICY AND RESHAPE
% =========================================================
if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    Policy = PolicyKron;
    return;
end

if has_a1
    n_daprime = [n_d, n_a1];
else
    n_daprime = n_d;
end

PolicyKron_flat = reshape(PolicyKron, [size(PolicyKron,1), N_a, N_z, N_j]);
Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron_flat, n_daprime, N_a, n_z, N_j, vfoptions);

if has_a1
    n_a_full = [N_a1, N_a2];
else
    n_a_full = N_a2;
end

if isempty(n_z) || prod(n_z) == 0
    V = reshape(V, [n_a_full, N_j]);
    Policy = reshape(Policy, [size(Policy, 1), n_a_full, N_j]);
else
    V = reshape(V, [n_a_full, n_z, N_j]);
    Policy = reshape(Policy, [size(Policy, 1), n_a_full, n_z, N_j]);
end

end

% =========================================================
% UNIFIED EZ 5D TENSOR BLOCK FUNCTION
% =========================================================
function [V_sub, Pol_d_combo, L2idx, L2flag] = Evaluate_EZ_TensorBlock(...
    state_idx, N_d1, N_d2, N_d3, N_a1, N_a2, N_z_safe, ...
    beta_j, EV_max_d3, Pol_d2_idx, d1_grid, d3_grid, a1_grid, a2_grid, z_gridvals, ...
    TensorReturnFn, ReturnFnParamsCell, ezc1_j, ezc2_j, ezc7_j, ezc4, ezc9, has_d1, has_a1)

N_block = length(state_idx);

% 1. Setup 5D Choice/State Structures (Dimension swapped for Tie-Breaking)
d1_in      = reshape(d1_grid, [N_d1, 1, 1, 1, 1]);
d3_in      = reshape(d3_grid, [1, N_d3, 1, 1, 1]);
a1prime_in = reshape(a1_grid, [1, 1, N_a1, 1, 1]);

[a1_idx, a2_idx] = ind2sub([N_a1, N_a2], state_idx);
A1_cells = reshape(a1_grid(a1_idx), [1, 1, 1, N_block, 1]);
A2_cells = reshape(a2_grid(a2_idx), [1, 1, 1, N_block, 1]);
Z_cells  = reshape(z_gridvals, [1, 1, 1, 1, N_z_safe]);

% 2. Dynamically Assemble ReturnFn Signature
ReturnFn_Args = {};
if has_d1, ReturnFn_Args{end+1} = d1_in; end
ReturnFn_Args{end+1} = d3_in;
if has_a1
    ReturnFn_Args{end+1} = a1prime_in;
    ReturnFn_Args{end+1} = A1_cells;
end
ReturnFn_Args{end+1} = A2_cells;
if N_z_safe > 0, ReturnFn_Args{end+1} = Z_cells; end
ReturnFn_Args = [ReturnFn_Args, ReturnFnParamsCell];

% 3. Evaluate F (5D)
F_tensor = TensorReturnFn(ReturnFn_Args{:});
temp2 = F_tensor;
becareful = logical(isfinite(F_tensor) .* (F_tensor ~= 0));
temp2(becareful) = F_tensor(becareful) .^ ezc2_j;
temp2(F_tensor == 0) = -Inf;

% 4. Assemble RHS (EV_bc Permuted for Tie-Breaking)
EV_bc_perm = permute(EV_max_d3, [2, 1, 3]);
EV_bc = reshape(EV_bc_perm, [1, N_d3, N_a1, 1, N_z_safe]);
entireRHS = ezc1_j .* temp2 + ezc9 .* beta_j .* EV_bc;

RHS = entireRHS;
temp5 = logical(isfinite(entireRHS) .* (entireRHS ~= 0));
RHS(temp5) = entireRHS(temp5) .^ ezc7_j;

RHS_flat = reshape(RHS, [N_d1 * N_d3 * N_a1, N_block * N_z_safe]);
[V_sub_coarse, opt_idx_flat] = max(RHS_flat, [], 1);
V_sub = V_sub_coarse;

% 5. Simultaneous Compression (d3 before a1prime for Tie-Breaking)
[d1_opt, d3_opt, a1prime_opt] = ind2sub([N_d1, N_d3, N_a1], opt_idx_flat);
d1_opt = reshape(d1_opt, [N_block, N_z_safe]);
a1prime_opt = reshape(a1prime_opt, [N_block, N_z_safe]);
d3_opt = reshape(d3_opt, [N_block, N_z_safe]);

z_idx_bc = repmat(reshape(1:N_z_safe, [1, N_z_safe]), [N_block, 1]);
linear_d2_query = a1prime_opt + (d3_opt - 1)*N_a1 + (z_idx_bc - 1)*N_a1*N_d3;
d2_opt = reshape(Pol_d2_idx(linear_d2_query(:)), [N_block, N_z_safe]);

Pol_d_combo = d1_opt + (d2_opt - 1) * N_d1 + (d3_opt - 1) * N_d1 * N_d2 + (a1prime_opt - 1) * (N_d1 * N_d2 * N_d3);
V_sub  = reshape(V_sub, [N_block, N_z_safe]);

L2idx  = [];
L2flag = [];


end