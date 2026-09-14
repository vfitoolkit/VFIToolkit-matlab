function [V, Policy] = ValueFnIter_VFHorz_RiskyAssetSemiExo(n_d, n_a1, n_a2, n_semiz, n_z, n_u, N_j, ...
    d_grid, a1_grid, a2_grid, semiz_gridvals_J, z_gridvals_J, u_grid, ...
    pi_semiz_J, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, ...
    DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

% 1. Extract Dimensions and Grids
has_a1 = ~isempty(n_a1) && prod(n_a1) > 0;
N_a1 = max(prod(n_a1), 1);
N_a2 = max(prod(n_a2), 1);
N_a  = N_a1 * N_a2;
N_z  = prod(n_z);
N_u  = prod(n_u);
N_semiz = prod(n_semiz);

% Parse refine_d = [d1, d2, d3, d4]
n_d1 = 0; N_d1 = 1; d1_grid = [];
if vfoptions.refine_d(1) > 0
    n_d1 = n_d(1:vfoptions.refine_d(1));
    N_d1 = prod(n_d1);
    d1_grid = d_grid(1:sum(n_d1));
end
has_d1 = (N_d1 > 1) || (N_d1 == 1 && length(d_grid) >= sum(n_d1) && d_grid(1) ~= 0);

offset = sum(n_d1);
n_d2 = n_d(vfoptions.refine_d(1)+1 : vfoptions.refine_d(1)+vfoptions.refine_d(2));
N_d2 = prod(n_d2);
d2_grid = d_grid(offset+1 : offset+sum(n_d2));

offset = offset + sum(n_d2);
n_d3 = n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1 : vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3));
N_d3 = prod(n_d3);
d3_grid = d_grid(offset+1 : offset+sum(n_d3));

offset = offset + sum(n_d3);
n_d4 = n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+vfoptions.refine_d(3)+1 : end);
N_d4 = prod(n_d4);
d4_grid = d_grid(offset+1 : end);

% 2. Push Variables to GPU
if isempty(d1_grid), d1_grid = gpuArray(0); else, d1_grid = gpuArray(d1_grid(:)); end
if isempty(a1_grid), a1_grid = gpuArray(0); else, a1_grid = gpuArray(a1_grid(:)); end
a2_grid = gpuArray(a2_grid(:));
d2_grid = gpuArray(d2_grid(:));
d3_grid = gpuArray(d3_grid(:));
d4_grid = gpuArray(d4_grid(:));
u_grid  = gpuArray(u_grid(:));
pi_u    = gpuArray(pi_u(:));
pi_z_J  = gpuArray(pi_z_J);
pi_semiz_J = gpuArray(pi_semiz_J);
z_gridvals = gpuArray(z_gridvals_J);
semiz_gridvals = gpuArray(semiz_gridvals_J);

V = zeros(N_a1, N_a2, N_semiz, max(N_z,1), N_j, 'like', a2_grid);
PolicyKron = zeros(1, N_a * N_semiz, max(N_z,1), N_j, 'like', a2_grid);
V_next = zeros(N_a1, N_a2, N_semiz, max(N_z,1), 'like', a2_grid);

D2_3D = reshape(d2_grid, [N_d2, 1, 1]);
D3_3D = reshape(d3_grid, [1, N_d3, 1]);
U_3D  = reshape(u_grid,  [1, 1, N_u]);

% =========================================================
% UNIVERSAL MIX-IN: EPSTEIN-ZIN VS CRRA
% =========================================================
is_EZ = isfield(vfoptions, 'exoticpreferences') && strcmp(vfoptions.exoticpreferences, 'EpsteinZin');
if is_EZ
    ezc2 = vfoptions.ezc2; ezc3 = vfoptions.ezc3; ezc4 = vfoptions.ezc4; 
    ezc5 = vfoptions.ezc5; ezc6 = vfoptions.ezc6; ezc7 = vfoptions.ezc7; 
    ezc8 = vfoptions.ezc8; sj   = vfoptions.sj;   warmglow = vfoptions.warmglow;
else
    % Neutral CRRA fallbacks (collapses EZ math to standard)
    ezc2 = ones(N_j,1); ezc3 = 1; ezc4 = 1; 
    ezc5 = ones(N_j,1); ezc6 = ones(N_j,1); ezc7 = ones(N_j,1); 
    ezc8 = ones(N_j,1); sj   = ones(N_j,1); warmglow = 0;
end

% =========================================================
% TIME LOOP
% =========================================================
for jj = N_j : -1 : 1
    ReturnFnParamsCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);
    aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);
    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);

    aprime_tensor = aprimeFn(D2_3D, D3_3D, U_3D, aprimeFnParamsCell{:});
    aprime_clamped = max(min(aprime_tensor, a2_grid(end)), a2_grid(1));

    % ---------------------------------------------------------
    % Multi-State Interpolation & Expectations (u, z, semiz)
    % ---------------------------------------------------------
    if jj == N_j
        EV_max_d3 = zeros(N_a1, N_semiz, N_d4, N_d3, max(N_z,1), 'like', a2_grid);
        Pol_d2_idx = ones(N_a1, N_semiz, N_d4, N_d3, max(N_z,1), 'like', a2_grid);
    else
        pi_u_rs = reshape(pi_u, [1, N_u, 1]);
        EV_z = zeros(N_a1, N_semiz, N_d2*N_d3, max(N_z,1), 'like', a2_grid);
        pi_z_j = pi_z_J(:,:,jj);

        % =========================================================
        % STEP 2: UNIVERSAL MIX-IN (EZ VALUE TRANSFORMATION) - VECTORIZED
        % =========================================================
        valid_V = isfinite(V_next) & (V_next ~= 0);
        V_transformed = V_next;
        if ezc5(jj) == 1
            V_transformed(valid_V) = ezc4 * V_next(valid_V);
        else
            V_transformed(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj);
        end
        V_transformed(V_next == 0) = 0;

        % Interpolate aprime and compute E_u, E_z
        for i_a1 = 1:N_a1
            for i_semiz = 1:N_semiz
                % SLICE THE ALREADY-TRANSFORMED TENSOR
                V_slice = squeeze(V_transformed(i_a1, :, i_semiz, :));
                if max(N_z,1) == 1, V_slice = V_slice(:); end

                inf_mask = (V_slice == -Inf);
                V_safe = V_slice;
                V_safe(inf_mask) = 0;

                V_int_slice = interp1(a2_grid, V_safe, aprime_clamped(:), 'linear');
                inf_int_slice = interp1(a2_grid, cast(inf_mask, 'like', a2_grid), aprime_clamped(:), 'linear');
                V_int_slice(inf_int_slice > 0) = -Inf;

                V_interp = reshape(V_int_slice, [N_d2*N_d3, N_u, max(N_z,1)]);

                inf_mask_u = (V_interp == -Inf);
                V_interp_safe = V_interp;
                V_interp_safe(inf_mask_u) = 0;
                EV_u_slice = sum(V_interp_safe .* pi_u_rs, 2);
                inf_infect_u = double(inf_mask_u) .* double(pi_u_rs > 0);
                EV_u_slice(sum(inf_infect_u, 2) > 0) = -Inf;
                EV_u_slice = squeeze(EV_u_slice);

                if N_z > 0
                    inf_mask_z = (EV_u_slice == -Inf);
                    EV_u_safe = EV_u_slice;
                    EV_u_safe(inf_mask_z) = 0;
                    EV_z_slice = EV_u_safe * pi_z_j';
                    inf_infect_z = double(inf_mask_z) * double(pi_z_j' > 0);
                    EV_z_slice(inf_infect_z > 0) = -Inf;
                else
                    EV_z_slice = EV_u_slice;
                end
                EV_z(i_a1, i_semiz, :, :) = EV_z_slice;
            end
        end

        % Compute E_semiz based on d4 (buyhouse)
        EV_semiz = zeros(N_a1, N_semiz, N_d4, N_d2*N_d3, max(N_z,1), 'like', a2_grid);

        % FIXED PERMUTATION: [s_now, d4, s_next]
        pi_perm = permute(pi_semiz_J(:,:,:,jj), [1, 3, 2]);
        pi_sz_flat = reshape(pi_perm, [N_semiz * N_d4, N_semiz]);

        for i_a1 = 1:N_a1
            for i_z = 1:max(N_z,1)
                EV_z_slice = squeeze(EV_z(i_a1, :, :, i_z));
                if N_semiz == 1, EV_z_slice = EV_z_slice(:)'; end

                inf_mask_s = (EV_z_slice == -Inf);
                EV_z_safe = EV_z_slice;
                EV_z_safe(inf_mask_s) = 0;

                % Matrix mult across semiz transition
                res = pi_sz_flat * EV_z_safe;

                inf_infect_s = double(pi_sz_flat) * double(inf_mask_s);
                res(inf_infect_s > 0) = -Inf;

                EV_semiz(i_a1, :, :, :, i_z) = reshape(res, [N_semiz, N_d4, N_d2*N_d3]);
            end
        end

        valid_EV = isfinite(EV_semiz) & (EV_semiz ~= 0);
        if ezc6(jj) ~= 1
            EV_semiz(valid_EV) = max(EV_semiz(valid_EV), 0).^ezc6(jj);
        end
        if ezc8(jj) ~= 1
            EV_semiz(valid_EV) = max(EV_semiz(valid_EV), 0).^ezc8(jj);
        end

        % DIMENSIONAL COMPRESSION: Maximize out d2 (riskyshare)
        EV_tensor = reshape(EV_semiz, [N_a1, N_semiz, N_d4, N_d2, N_d3, max(N_z,1)]);
        [EV_max_d2_raw, Pol_d2_idx] = max(EV_tensor, [], 4);

        EV_max_d3 = reshape(EV_max_d2_raw, [N_a1, N_semiz, N_d4, N_d3, max(N_z,1)]);
        Pol_d2_idx = reshape(Pol_d2_idx, [N_a1, N_semiz, N_d4, N_d3, max(N_z,1)]);
    end

    % =========================================================
    % 6D TENSOR BLOCK
    % =========================================================
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_RiskyAssetSemiExo_TensorBlock(...
        state_idx, N_d1, N_d2, N_d3, N_d4, N_a1, N_a2, N_semiz, max(N_z,1), ...
        beta_j, EV_max_d3, Pol_d2_idx, d1_grid, d3_grid, d4_grid, a1_grid, a2_grid, ...
        semiz_gridvals(:,:,jj), z_gridvals(:,:,jj), ReturnFn, ReturnFnParamsCell, has_d1, has_a1, ...
        ezc2(jj), ezc3, ezc4, ezc7(jj));

    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        vfopts_dc = vfoptions; vfopts_dc.level1n = vfoptions.level1n(1);
        [V_j_max, Pol_d_combo] = ValueFnIter_DC1_Slicer(N_a * N_semiz, N_a * N_semiz, 1, max(N_z,1), vfopts_dc, EvalBlockFn);
    else
        [V_j_max, Pol_d_combo] = EvalBlockFn(1:(N_a * N_semiz), [], 0);
    end

    V(:,:,:,:,jj) = reshape(V_j_max, [N_a1, N_a2, N_semiz, max(N_z,1)]);
    PolicyKron(1, :, :, jj) = Pol_d_combo;
    V_next = V(:,:,:,:,jj);
end

% =========================================================
% UNPACK POLICY AND RESHAPE
% =========================================================
if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    Policy = PolicyKron; return;
end

n_daprime = [n_d, n_a1];
PolicyKron_flat = reshape(PolicyKron, [size(PolicyKron,1), N_a * N_semiz, max(N_z,1), N_j]);
Policy_flat = UnKronPolicyIndexes1_FHorz_z(PolicyKron_flat, n_daprime, N_a * N_semiz, n_z, N_j, vfoptions);

n_a_full = [N_a1, N_a2];
if isempty(n_z) || prod(n_z) == 0
    V = reshape(V, [n_a_full, n_semiz, N_j]);
    Policy = reshape(Policy_flat, [size(Policy_flat, 1), n_a_full, n_semiz, N_j]);
else
    V = reshape(V, [n_a_full, n_semiz, n_z, N_j]);
    Policy = reshape(Policy_flat, [size(Policy_flat, 1), n_a_full, n_semiz, n_z, N_j]);
end
end

% =========================================================
% UNIFIED 6D TENSOR BLOCK FUNCTION
% =========================================================
function [V_sub, Pol_d_combo, L2idx, L2flag] = Evaluate_RiskyAssetSemiExo_TensorBlock(...
    state_idx, N_d1, N_d2, N_d3, N_d4, N_a1, N_a2, N_semiz, N_z_safe, ...
    beta_j, EV_max_d3, Pol_d2_idx, d1_grid, d3_grid, d4_grid, a1_grid, a2_grid, Semiz_mat, z_gridvals, ...
    ReturnFn, ReturnFnParamsCell, has_d1, has_a1, ezc2_j, ezc3, ezc4, ezc7_j)

N_block = length(state_idx);

% 1. Setup 6D Choice/State Structures
d1_in      = reshape(d1_grid, [N_d1, 1, 1, 1, 1, 1]);
d3_in      = reshape(d3_grid, [1, N_d3, 1, 1, 1, 1]);
d4_in      = reshape(d4_grid, [1, 1, N_d4, 1, 1, 1]);
a1prime_in = reshape(a1_grid, [1, 1, 1, N_a1, 1, 1]);

[a1_idx, a2_idx, semiz_idx] = ind2sub([N_a1, N_a2, N_semiz], state_idx);
A1_cells = reshape(a1_grid(a1_idx), [1, 1, 1, 1, N_block, 1]);
A2_cells = reshape(a2_grid(a2_idx), [1, 1, 1, 1, N_block, 1]);
Z_cells  = reshape(z_gridvals,      [1, 1, 1, 1, 1, N_z_safe]);

num_semiz = size(Semiz_mat, 2);
Semiz_Args = cell(1, num_semiz);
for i_s = 1:num_semiz
    Semiz_Args{i_s} = reshape(Semiz_mat(semiz_idx, i_s), [1, 1, 1, 1, N_block, 1]);
end

% 2. Dynamically Assemble ReturnFn Signature
ReturnFn_Args = {};
if has_d1, ReturnFn_Args{end+1} = d1_in; end
ReturnFn_Args{end+1} = d3_in;
ReturnFn_Args{end+1} = d4_in;
if has_a1
    ReturnFn_Args{end+1} = a1prime_in;
    ReturnFn_Args{end+1} = A1_cells;
end
ReturnFn_Args{end+1} = A2_cells;
ReturnFn_Args = [ReturnFn_Args, Semiz_Args];
if N_z_safe > 0, ReturnFn_Args{end+1} = Z_cells; end
ReturnFn_Args = [ReturnFn_Args, ReturnFnParamsCell];

% 3. Evaluate F (6D) and apply ezc
F_tensor = ReturnFn(ReturnFn_Args{:});
EV_query = EV_max_d3(:, semiz_idx, :, :, :);
EV_bc = permute(EV_query, [4, 3, 1, 2, 5]);
EV_bc = reshape(EV_bc, [1, N_d3, N_d4, N_a1, N_block, N_z_safe]);
% (Assuming ezc1_j = 1 for standard models, or extract it from vfoptions if needed)
RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bc, beta_j, 1, ezc2_j, ezc3, ezc4, ezc7_j);

% 5. Simultaneous Compression (Flatten all choices)
RHS_flat = reshape(RHS, [N_d1 * N_d3 * N_d4 * N_a1, N_block * N_z_safe]);
[V_sub_coarse, opt_idx_flat] = max(RHS_flat, [], 1);

[d1_opt, d3_opt, d4_opt, a1prime_opt] = ind2sub([N_d1, N_d3, N_d4, N_a1], opt_idx_flat);

d1_opt = reshape(d1_opt, [N_block, N_z_safe]);
d3_opt = reshape(d3_opt, [N_block, N_z_safe]);
d4_opt = reshape(d4_opt, [N_block, N_z_safe]);
a1prime_opt = reshape(a1prime_opt, [N_block, N_z_safe]);

% Extract optimal d2 (riskyshare)
z_idx_bc = repmat(reshape(1:N_z_safe, [1, N_z_safe]), [N_block, 1]);
semiz_idx_bc = repmat(semiz_idx(:), [1, N_z_safe]);

linear_d2_query = a1prime_opt + ...
    (semiz_idx_bc - 1) * N_a1 + ...
    (d4_opt - 1) * (N_a1 * N_semiz) + ...
    (d3_opt - 1) * (N_a1 * N_semiz * N_d4) + ...
    (z_idx_bc - 1) * (N_a1 * N_semiz * N_d4 * N_d3);

d2_opt = reshape(Pol_d2_idx(linear_d2_query(:)), [N_block, N_z_safe]);

% Pack unified Kron Index: [d1, d2, d3, d4, a1prime]
Pol_d_combo = d1_opt + ...
    (d2_opt - 1) * N_d1 + ...
    (d3_opt - 1) * (N_d1 * N_d2) + ...
    (d4_opt - 1) * (N_d1 * N_d2 * N_d3) + ...
    (a1prime_opt - 1) * (N_d1 * N_d2 * N_d3 * N_d4);

V_sub  = reshape(V_sub_coarse, [N_block, N_z_safe]);
L2idx  = []; L2flag = [];


end