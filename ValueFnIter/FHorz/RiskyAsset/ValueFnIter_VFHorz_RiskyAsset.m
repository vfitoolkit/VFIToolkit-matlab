function [V, Policy] = ValueFnIter_VFHorz_RiskyAsset(n_d, n_a1, n_a2, n_z, n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions)

% 1. Extract Dimensions and Grids
N_a1 = prod(n_a1);
N_a2 = prod(n_a2);
N_a  = max(N_a1, 1) * max(N_a2, 1);
N_z  = prod(n_z);
N_u  = prod(n_u);

% Handle refine_d partitions (d1=0, d2=riskyshare, d3=savings)
n_d1 = 0; N_d1 = 1;
if vfoptions.refine_d(1) > 0
    n_d1 = n_d(1:vfoptions.refine_d(1));
    N_d1 = prod(n_d1);
end
n_d2 = n_d(vfoptions.refine_d(1)+1 : vfoptions.refine_d(1)+vfoptions.refine_d(2));
N_d2 = prod(n_d2);
n_d3 = n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1 : end);
N_d3 = prod(n_d3);

d2_grid = d_grid(sum(n_d1)+1 : sum(n_d1)+sum(n_d2));
d3_grid = d_grid(sum(n_d1)+sum(n_d2)+1 : end);

% 2. Push Variables to GPU
a_grid  = gpuArray(a1_grid(:));
d2_grid = gpuArray(d2_grid(:));
d3_grid = gpuArray(d3_grid(:));
u_grid  = gpuArray(u_grid(:));
pi_u    = gpuArray(pi_u(:));
pi_z_J  = gpuArray(pi_z_J);
z_gridvals = gpuArray(z_gridvals_J);

% 3. Grid Interpolation Layer Setup (Refining d3)
gridinterplayer = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;
if gridinterplayer
    n2short = vfoptions.n2short;
    n2long  = 2 * n2short + 1;
    % Create the fine grid for the continuous choice (savings)
    d3prime_grid = interp1(1:N_d3, d3_grid, 1:(1/(n2short+1)):N_d3, 'linear');
    d3prime_grid = gpuArray(d3prime_grid(:));
else
    n2short = 0;
    n2long  = 0;
    d3prime_grid = [];
end

V = zeros(N_a, N_z, N_j, 'like', a_grid);
if gridinterplayer
    PolicyKron = zeros(4, N_a, N_z, N_j, 'like', a_grid);
else
    PolicyKron = zeros(1, N_a, N_z, N_j, 'like', a_grid);
end

V_next = zeros(N_a, N_z, 'like', a_grid);

% Pre-allocate the massive aprime evaluation tensor
D2_3D = reshape(d2_grid, [N_d2, 1, 1]);
D3_3D = reshape(d3_grid, [1, N_d3, 1]);
U_3D  = reshape(u_grid,  [1, 1, N_u]);

% =========================================================
% TIME LOOP
% =========================================================
for jj = N_j : -1 : 1
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    DiscountFactor    = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactor);

    if jj == N_j
        % Final period: Future value is zero.
        EV_max_d3 = zeros(1, N_d3, N_z, 'like', a_grid);
        Pol_d2_idx = ones(1, N_d3, N_z, 'like', a_grid);
    else
        % 1. Evaluate future assets across (d2, d3, u)
        aprime_tensor = aprimeFn(D2_3D, D3_3D, U_3D, aprimeFnParamsVec{:});
        aprime_clamped = max(min(aprime_tensor, a_grid(end)), a_grid(1));

        % 2. Interpolate V_next onto aprime using Pure NaN Shield
        inf_mask = double(V_next == -Inf);
        V_safe = V_next;
        V_safe(V_next == -Inf) = 0;

        % interp1 handles the entire matrix (all z states) simultaneously
        V_interp = interp1(a_grid, V_safe, aprime_clamped(:), 'linear');
        inf_interp = interp1(a_grid, inf_mask, aprime_clamped(:), 'linear');
        V_interp(inf_interp > 0) = -Inf;

        V_interp = reshape(V_interp, [N_d2*N_d3, N_u, N_z]);

        % 3. Expectation over u (Return Shock)
        inf_mask_u = (V_interp == -Inf);
        V_interp_safe = V_interp;
        V_interp_safe(inf_mask_u) = 0;

        pi_u_rs = reshape(pi_u, [1, N_u, 1]);
        EV_u = sum(V_interp_safe .* pi_u_rs, 2);

        inf_infect_u = double(inf_mask_u) .* double(pi_u_rs > 0);
        EV_u(sum(inf_infect_u, 2) > 0) = -Inf;
        EV_u = reshape(EV_u, [N_d2*N_d3, N_z]);

        % 4. Expectation over z (Labor Shock)
        pi_z = pi_z_J(:,:,jj);
        inf_mask_z = (EV_u == -Inf);
        EV_u_safe = EV_u;
        EV_u_safe(inf_mask_z) = 0;

        EV_z = EV_u_safe * pi_z';
        inf_infect_z = double(inf_mask_z) * double(pi_z' > 0);
        EV_z(inf_infect_z > 0) = -Inf;

        EV_tensor = reshape(EV_z, [N_d2, N_d3, N_z]);

        % 5. DIMENSIONAL COMPRESSION: Maximize out d2 (riskyshare)
        [EV_max_d3, Pol_d2_idx] = max(EV_tensor, [], 1); % Size: [1, N_d3, N_z]
    end

    % =========================================================
    % TENSOR BLOCK: Optimize d3 against current state (a, z)
    % =========================================================
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_RiskyAsset_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, ...
        N_d2, N_d3, N_a, N_z, gridinterplayer, n2short, n2long, ...
        beta_j, EV_max_d3, Pol_d2_idx, d3_grid, d3prime_grid, a_grid, z_gridvals(:,:,jj), ...
        ReturnFn, ReturnFnParamsVec);

    % Slicer Dispatcher
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        vfopts_dc = vfoptions;
        vfopts_dc.level1n = vfoptions.level1n(1);
        [V_j_max, Pol_d3_max, Pol_d_combo, Pol_L2idx, Pol_L2flag] = ...
            ValueFnIter_DC1_Slicer(N_a, N_a, 1, N_z, vfopts_dc, EvalBlockFn);
    else
        [V_j_max, Pol_d3_max, Pol_d_combo, Pol_L2idx, Pol_L2flag] = EvalBlockFn(1:N_a, [], 0);
    end

    V(:,:,jj) = V_j_max;

    % Pack the policies for UnKron functions
    if gridinterplayer
        adjust = (Pol_L2idx < 1 + n2short + 1);
        lower_grid_pt = Pol_d3_max - adjust;
        lower_grid_pt = max(lower_grid_pt, 1);
        subgrid_step  = adjust .* Pol_L2idx + (1 - adjust) .* (Pol_L2idx - n2short - 1);

        G_segments = n2short + 1;
        at_top = (lower_grid_pt >= N_d3);
        lower_grid_pt(at_top) = N_d3 - 1;
        subgrid_step(at_top)  = G_segments + 1;

        PolicyKron(1, :, :, jj) = Pol_d_combo; % Packed (d2, d3)
        PolicyKron(2, :, :, jj) = lower_grid_pt;
        PolicyKron(3, :, :, jj) = subgrid_step;
        PolicyKron(4, :, :, jj) = Pol_L2flag;
    else
        PolicyKron(1, :, :, jj) = Pol_d_combo;
    end

    V_next = V(:,:,jj);
end

% =========================================================
% UNPACK POLICY
% =========================================================
if vfoptions.outputkron == 1
    Policy = PolicyKron;
    return
end

if gridinterplayer
    PolicyKron_flat = reshape(PolicyKron, [4, N_a, N_z, N_j]);
    Policy = UnKronPolicyIndexes2_FHorz_z(PolicyKron_flat, n_d2, n_d3, N_a, n_z, N_j, vfoptions);
else
    PolicyKron_flat = reshape(PolicyKron, [1, N_a, N_z, N_j]);
    Policy = UnKronPolicyIndexes2_FHorz_z(PolicyKron_flat, n_d2, n_d3, N_a, n_z, N_j, vfoptions);
end
end

% =========================================================
% UNIFIED TENSOR BLOCK FUNCTION
% =========================================================
function [V_sub, Pol_d3_idx, Pol_d_combo, L2idx, L2flag] = Evaluate_RiskyAsset_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, ...
    N_d2, N_d3, N_a, N_z, gridinterplayer, n2short, n2long, ...
    beta_j, EV_max_d3, Pol_d2_idx, d3_grid, d3prime_grid, a_grid, z_gridvals, ...
    ReturnFn, ReturnFnParamsVec)

N_block = length(state_idx);

if isempty(loweredge_matrix)
    N_choice = N_d3;
    d3_idx_tensor = reshape(1:N_d3, [N_choice, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar);
    N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [N_choice, 1, 1]);
    base_edge = reshape(loweredge_matrix, [1, N_block, N_z]);
    d3_idx_tensor = base_edge + offset;
end

d3_in = reshape(d3_grid(d3_idx_tensor(:)), size(d3_idx_tensor));
A_cells = reshape(a_grid(state_idx), [1, N_block, 1]);
Z_cells = reshape(z_gridvals, [1, 1, N_z]);

% Broadcast purely against d3 (Savings)
F_tensor = ReturnFn(d3_in, A_cells, Z_cells, ReturnFnParamsVec{:});

z_offset = reshape(0:N_z-1, [1, 1, N_z]) .* N_d3;
ev_lin_idx = d3_idx_tensor + z_offset;
EV_bc = reshape(EV_max_d3(ev_lin_idx(:)), size(ev_lin_idx));

RHS = F_tensor + beta_j .* EV_bc;
RHS_flat = reshape(RHS, [N_choice, N_block * N_z]);
[V_sub_coarse, Pol_sub_idx_coarse] = max(RHS_flat, [], 1);

if isempty(loweredge_matrix)
    d3_idx_coarse = Pol_sub_idx_coarse;
else
    loweredge_flat = loweredge_matrix(:)';
    d3_idx_coarse = loweredge_flat + Pol_sub_idx_coarse - 1;
end

% ---------------------------------------------------------
% CONTINUOUS GRID INTERPOLATION (d3 refinement)
% ---------------------------------------------------------
if gridinterplayer
    midpoint = max(min(d3_idx_coarse, N_d3 - 1), 2);
    base_idx = midpoint + (midpoint - 1) * n2short;
    base_idx_tensor = reshape(base_idx, [1, N_block, N_z]);
    offset_fine = reshape(-n2short-1 : n2short+1, [n2long, 1, 1]);
    fine_idx_tensor = base_idx_tensor + offset_fine;

    d3_in_fine = reshape(d3prime_grid(fine_idx_tensor(:)), size(fine_idx_tensor));
    F_fine = ReturnFn(d3_in_fine, A_cells, Z_cells, ReturnFnParamsVec{:});

    % Dual-Interpolation pure NaN shield for EV_max_d3
    inf_mask = double(EV_max_d3 == -Inf);
    EV_safe = EV_max_d3;
    EV_safe(inf_mask > 0) = 0;

    EV_fine_full = interp1(d3_grid, reshape(EV_safe, [N_d3, N_z]), d3prime_grid, 'linear');
    inf_fine_full = interp1(d3_grid, reshape(inf_mask, [N_d3, N_z]), d3prime_grid, 'linear');
    EV_fine_full(inf_fine_full > 0) = -Inf;

    z_col = reshape(0:N_z-1, [1, 1, N_z]) .* length(d3prime_grid);
    fine_lin_idx = fine_idx_tensor + z_col;
    EV_fine = reshape(EV_fine_full(fine_lin_idx(:)), size(fine_lin_idx));

    RHS_fine = F_fine + beta_j .* EV_fine;
    RHS_fine_flat = reshape(RHS_fine, [n2long, N_block * N_z]);
    [V_sub, maxindexL2] = max(RHS_fine_flat, [], 1);

    isInfLower = (RHS_fine_flat(1, :) == -Inf);
    isInfUpper = (RHS_fine_flat(end, :) == -Inf);
    inLowerStrict = (maxindexL2 >= 2) & (maxindexL2 <= n2short + 1);
    inUpperStrict = (maxindexL2 >= n2short + 3) & (maxindexL2 <= n2long - 1);
    L2flag = 2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper);

    V_sub      = reshape(V_sub, [N_block, N_z]);
    Pol_d3_idx = reshape(midpoint, [N_block, N_z]);
    L2idx      = reshape(maxindexL2, [N_block, N_z]);
    L2flag     = reshape(L2flag, [N_block, N_z]);
else
    V_sub      = reshape(V_sub_coarse, [N_block, N_z]);
    Pol_d3_idx = reshape(d3_idx_coarse, [N_block, N_z]);
    L2idx      = [];
    L2flag     = [];
end

% ---------------------------------------------------------
% RE-PACK COMBO INDEX (d2, d3)
% ---------------------------------------------------------
z_col = reshape(0:N_z-1, [1, N_z]) .* N_d3;
d2_query_idx = Pol_d3_idx + repmat(z_col, [N_block, 1]);
d2_opt = reshape(Pol_d2_idx(d2_query_idx(:)), [N_block, N_z]);

% Standard Toolkit Kron packing: d2 + (d3 - 1) * N_d2
Pol_d_combo = d2_opt + (Pol_d3_idx - 1) * N_d2;


end