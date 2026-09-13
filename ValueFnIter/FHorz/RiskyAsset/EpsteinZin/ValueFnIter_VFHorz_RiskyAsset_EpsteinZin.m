function [V, Policy] = ValueFnIter_VFHorz_RiskyAsset_EpsteinZin(n_d, n_a1, n_a2, n_z, n_u, N_j, d_grid, a1_grid, a2_grid, z_gridvals_J, u_grid, pi_z_J, pi_u, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions, sj, warmglow, ezc2, ezc3, ezc4, ezc5, ezc6, ezc7, ezc8)

% 1. Extract Dimensions and Grids
N_a1 = prod(n_a1); N_a2 = prod(n_a2); N_a = max(N_a1, 1) * max(N_a2, 1);
N_z = prod(n_z); N_u = prod(n_u);

n_d1 = 0; N_d1 = 1;
if vfoptions.refine_d(1) > 0; n_d1 = n_d(1:vfoptions.refine_d(1)); end
n_d2 = n_d(vfoptions.refine_d(1)+1 : vfoptions.refine_d(1)+vfoptions.refine_d(2)); N_d2 = prod(n_d2);
n_d3 = n_d(vfoptions.refine_d(1)+vfoptions.refine_d(2)+1 : end); N_d3 = prod(n_d3);

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

% 3. Grid Interpolation Layer Setup
gridinterplayer = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;
if gridinterplayer
    n2short = vfoptions.n2short; n2long  = 2 * n2short + 1;
    d3prime_grid = gpuArray(interp1(1:N_d3, d3_grid, 1:(1/(n2short+1)):N_d3, 'linear')');
else
    n2short = 0; n2long = 0; d3prime_grid = [];
end

V = zeros(N_a, N_z, N_j, 'like', a_grid);
if gridinterplayer
    PolicyKron = zeros(4, N_a, N_z, N_j, 'like', a_grid);
else
    PolicyKron = zeros(1, N_a, N_z, N_j, 'like', a_grid);
end
V_next = zeros(N_a, N_z, 'like', a_grid);

D2_3D = reshape(d2_grid, [N_d2, 1, 1]);
D3_3D = reshape(d3_grid, [1, N_d3, 1]);
U_3D  = reshape(u_grid,  [1, 1, N_u]);

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
        elseif vfoptions.EZoneminusbeta == 2; ezc1_j = 1 - sj(jj) * beta_j; end
    end

    aprime_tensor = aprimeFn(D2_3D, D3_3D, U_3D, aprimeFnParamsCell{:});
    aprime_clamped = max(min(aprime_tensor, a_grid(end)), a_grid(1));

    % ---------------------------------------------------------
    % Warm Glow of Bequests Expectation (WG_u)
    % ---------------------------------------------------------
    if warmglow == 1
        % Evaluate on a_grid and interpolate to perfectly match toolkit's numerical integration
        WG_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
        WG_raw = vfoptions.WarmGlowBequestsFn(a_grid, WG_params{:});

        WG_temp = WG_raw;
        valid_wg = isfinite(WG_raw) & (WG_raw ~= 0);
        if ezc5(jj) == 1
            WG_temp(valid_wg) = ezc4 * WG_raw(valid_wg);
        else
            WG_temp(valid_wg) = max(ezc4 * WG_raw(valid_wg), 0).^ezc5(jj);
        end
        WG_temp(WG_raw == 0) = 0;
        WG_temp(~isfinite(WG_raw)) = -Inf;

        inf_mask_wg = (WG_temp == -Inf);
        WG_safe = WG_temp; WG_safe(inf_mask_wg) = 0;

        WG_interp = interp1(a_grid, WG_safe, aprime_clamped(:), 'linear');
        inf_interp_wg = interp1(a_grid, double(inf_mask_wg), aprime_clamped(:), 'linear');
        WG_interp(inf_interp_wg > 0) = -Inf;

        WG_interp = reshape(WG_interp, [N_d2*N_d3, N_u]);
        inf_mask_wg_u = (WG_interp == -Inf);
        WG_safe_u = WG_interp; WG_safe_u(inf_mask_wg_u) = 0;

        pi_u_rs = reshape(pi_u, [1, N_u]);
        WG_u = sum(WG_safe_u .* pi_u_rs, 2);
        inf_infect_wg_u = double(inf_mask_wg_u) .* double(pi_u_rs > 0);
        WG_u(sum(inf_infect_wg_u, 2) > 0) = -Inf;
        WG_u = reshape(WG_u, [N_d2*N_d3, 1]);
    else
        WG_u = 0;
    end

    % ---------------------------------------------------------
    % EV Expectation & Fractional Exponent Combination
    % ---------------------------------------------------------
    warning('off', 'MATLAB:divideByZero');
    if jj == N_j
        temp4 = WG_u;
        valid_t4 = isfinite(temp4) & (temp4 ~= 0);
        if warmglow == 1
            temp4(valid_t4) = ((1 - sj(jj)) * max(temp4(valid_t4), 0).^ezc8(jj)).^ezc6(jj);
            temp4(WG_u == 0) = 0;
            temp4(~valid_t4 & WG_u ~= 0) = NaN;
        else
            temp4 = nan(N_d2*N_d3, 1, 'like', a_grid);
        end
        temp4 = repmat(temp4, [1, N_z]);
    else
        V_temp = V_next;
        valid_V = isfinite(V_next) & (V_next ~= 0);
        if ezc5(jj) == 1
            V_temp(valid_V) = ezc4 * V_next(valid_V);
        else
            V_temp(valid_V) = max(ezc4 * V_next(valid_V), 0).^ezc5(jj);
        end
        V_temp(V_next == 0) = 0;
        V_temp(~isfinite(V_next)) = -Inf;

        inf_mask_z = (V_temp == -Inf);
        V_safe = V_temp; V_safe(inf_mask_z) = 0;
        pi_z_j = pi_z_J(:,:,jj);
        EV_z = V_safe * pi_z_j';
        inf_infect_z = double(inf_mask_z) * double(pi_z_j' > 0);
        EV_z(inf_infect_z > 0) = -Inf;

        inf_mask = double(EV_z == -Inf);
        EV_safe = EV_z; EV_safe(inf_mask > 0) = 0;
        V_interp = interp1(a_grid, EV_safe, aprime_clamped(:), 'linear');
        inf_interp = interp1(a_grid, inf_mask, aprime_clamped(:), 'linear');
        V_interp(inf_interp > 0) = -Inf;
        V_interp = reshape(V_interp, [N_d2*N_d3, N_u, N_z]);

        inf_mask_u = (V_interp == -Inf);
        V_interp_safe = V_interp; V_interp_safe(inf_mask_u) = 0;
        pi_u_rs = reshape(pi_u, [1, N_u, 1]);
        EV_u = sum(V_interp_safe .* pi_u_rs, 2);
        inf_infect_u = double(inf_mask_u) .* double(pi_u_rs > 0);
        EV_u(sum(inf_infect_u, 2) > 0) = -Inf;
        EV_u = reshape(EV_u, [N_d2*N_d3, N_z]);

        temp4 = EV_u;
        if warmglow == 1
            valid_combined = isfinite(EV_u) & repmat(isfinite(WG_u), [1, N_z]) & ~((EV_u == 0) & repmat(WG_u == 0, [1, N_z]));
            temp4(valid_combined) = (sj(jj) * max(EV_u(valid_combined), 0).^ezc8(jj) + (1-sj(jj)) * max(WG_u(valid_combined), 0).^ezc8(jj)).^ezc6(jj);
            zero_mask = (EV_u == 0) & repmat(WG_u == 0, [1, N_z]);
            temp4(zero_mask) = 0;
            temp4(~valid_combined & ~zero_mask) = NaN;
        else
            valid_t4 = isfinite(EV_u) & (EV_u ~= 0);
            temp4(valid_t4) = (sj(jj) * max(EV_u(valid_t4), 0).^ezc8(jj)).^ezc6(jj);
            temp4(EV_u == 0) = 0;
            temp4(~valid_t4 & EV_u ~= 0) = NaN;
        end
    end
    warning('on', 'MATLAB:divideByZero');

    % DIMENSIONAL COMPRESSION: Maximize out d2 (riskyshare)
    temp4_tensor = reshape(temp4, [N_d2, N_d3, N_z]);

    % temp4 is POSITIVE. ezc3 flips it to negative so max() finds the TRUE optimum (minimum misery).
    [EV_max_d3_raw, Pol_d2_idx] = max(ezc3 * temp4_tensor, [], 1);

    % Flip it back to POSITIVE to safely feed into the Tensor Block
    EV_max_d3 = ezc3 * EV_max_d3_raw;
    EV_max_d3 = reshape(EV_max_d3, [N_d3, N_z]);
    EV_max_d3(isnan(EV_max_d3)) = -Inf;

    % =========================================================
    % TENSOR BLOCK: Optimize d3 against current state (a, z)
    % =========================================================
    EvalBlockFn = @(state_idx, loweredge_matrix, maxgap_scalar) Evaluate_EZ_TensorBlock(...
        state_idx, loweredge_matrix, maxgap_scalar, ...
        N_d2, N_d3, N_a, N_z, gridinterplayer, n2short, n2long, ...
        beta_j, EV_max_d3, Pol_d2_idx, d3_grid, d3prime_grid, a_grid, z_gridvals(:,:,jj), ...
        ReturnFn, ReturnFnParamsCell, ezc1_j, ezc2(jj), ezc7(jj), ezc4, ezc3);

    % Slicer Dispatcher
    if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
        vfopts_dc = vfoptions; vfopts_dc.level1n = vfoptions.level1n(1);
        [V_j_max, Pol_d3_max, Pol_d_combo, Pol_L2idx, Pol_L2flag] = ...
            ValueFnIter_DC1_Slicer(N_a, N_a, 1, N_z, vfopts_dc, EvalBlockFn);
    else
        [V_j_max, Pol_d3_max, Pol_d_combo, Pol_L2idx, Pol_L2flag] = EvalBlockFn(1:N_a, [], 0);
    end

    V(:,:,jj) = V_j_max;

    % Pack policies
    if gridinterplayer
        adjust = (Pol_L2idx < 1 + n2short + 1);
        lower_grid_pt = max(Pol_d3_max - adjust, 1);
        subgrid_step  = adjust .* Pol_L2idx + (1 - adjust) .* (Pol_L2idx - n2short - 1);
        at_top = (lower_grid_pt >= N_d3);
        lower_grid_pt(at_top) = N_d3 - 1;
        subgrid_step(at_top)  = (n2short + 1) + 1;

        PolicyKron(1, :, :, jj) = Pol_d_combo;
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
if isfield(vfoptions, 'outputkron') && vfoptions.outputkron == 1
    Policy = PolicyKron; return;
end
if gridinterplayer
    Policy = UnKronPolicyIndexes1_FHorz_z(reshape(PolicyKron, [4, N_a, N_z, N_j]), n_d, N_a, n_z, N_j, vfoptions);
else
    Policy = UnKronPolicyIndexes1_FHorz_z(reshape(PolicyKron, [1, N_a, N_z, N_j]), n_d, N_a, n_z, N_j, vfoptions);
end
end

% =========================================================
% UNIFIED EZ TENSOR BLOCK FUNCTION
% =========================================================
function [V_sub, Pol_d3_idx, Pol_d_combo, L2idx, L2flag] = Evaluate_EZ_TensorBlock(...
    state_idx, loweredge_matrix, maxgap_scalar, N_d2, N_d3, N_a, N_z, gridinterplayer, n2short, n2long, ...
    beta_j, EV_max_d3, Pol_d2_idx, d3_grid, d3prime_grid, a_grid, z_gridvals, ...
    ReturnFn, ReturnFnParamsCell, ezc1_j, ezc2_j, ezc7_j, ezc4, ezc3)

N_block = length(state_idx);
if isempty(loweredge_matrix)
    N_choice = N_d3;
    d3_idx_tensor = reshape(1:N_d3, [N_choice, 1, 1]);
else
    offset_vec = 0:gather(maxgap_scalar); N_choice = length(offset_vec);
    offset = reshape(gpuArray(offset_vec), [N_choice, 1, 1]);
    d3_idx_tensor = reshape(loweredge_matrix, [1, N_block, N_z]) + offset;
end

d3_in = reshape(d3_grid(d3_idx_tensor(:)), size(d3_idx_tensor));
A_cells = reshape(a_grid(state_idx), [1, N_block, 1]);
Z_cells = reshape(z_gridvals, [1, 1, N_z]);

% Evaluate F and keep it POSITIVE using ezc4
F_tensor = ReturnFn(d3_in, A_cells, Z_cells, ReturnFnParamsCell{:});
temp2 = F_tensor;
valid_F = isfinite(F_tensor) & (F_tensor ~= 0);

if ezc2_j == 1
    temp2(valid_F) = ezc4 * F_tensor(valid_F);
else
    temp2(valid_F) = max(ezc4 * F_tensor(valid_F), 0).^ezc2_j;
end
temp2(~isfinite(F_tensor)) = -Inf;

% EV_max_d3 is POSITIVE
ev_lin_idx = d3_idx_tensor + reshape(0:N_z-1, [1, 1, N_z]) .* N_d3;
EV_bc = reshape(EV_max_d3(ev_lin_idx(:)), size(ev_lin_idx));

% Assemble EZ RHS (POSITIVE + POSITIVE = POSITIVE)
entireRHS = ezc1_j .* temp2 + beta_j .* EV_bc;

% Flip back to NEGATIVE for final fractional power / evaluation using ezc3
RHS = entireRHS;
valid_RHS = isfinite(entireRHS) & (entireRHS ~= 0);

if ezc7_j == 1
    RHS(valid_RHS) = ezc3 * entireRHS(valid_RHS);
else
    RHS(valid_RHS) = ezc3 * (entireRHS(valid_RHS).^ezc7_j);
end
RHS(~isfinite(entireRHS)) = -Inf;

% RHS is now strictly NEGATIVE. max() correctly finds the highest true utility.
RHS_flat = reshape(RHS, [N_choice, N_block * N_z]);
[V_sub_coarse, Pol_sub_idx_coarse] = max(RHS_flat, [], 1);

if isempty(loweredge_matrix)
    d3_idx_coarse = Pol_sub_idx_coarse;
else
    d3_idx_coarse = loweredge_matrix(:)' + Pol_sub_idx_coarse - 1;
end

% ---------------------------------------------------------
% CONTINUOUS GRID INTERPOLATION (d3 refinement)
% ---------------------------------------------------------
if gridinterplayer
    midpoint = max(min(d3_idx_coarse, N_d3 - 1), 2);
    base_idx_tensor = reshape(midpoint + (midpoint - 1) * n2short, [1, N_block, N_z]);
    fine_idx_tensor = base_idx_tensor + reshape(-n2short-1 : n2short+1, [n2long, 1, 1]);

    d3_in_fine = reshape(d3prime_grid(fine_idx_tensor(:)), size(fine_idx_tensor));
    F_fine = ReturnFn(d3_in_fine, A_cells, Z_cells, ReturnFnParamsCell{:});

    temp2_fine = F_fine;
    valid_F_f = isfinite(F_fine) & (F_fine ~= 0);
    if ezc2_j == 1
        temp2_fine(valid_F_f) = ezc4 * F_fine(valid_F_f);
    else
        temp2_fine(valid_F_f) = max(ezc4 * F_fine(valid_F_f), 0).^ezc2_j;
    end
    temp2_fine(~isfinite(F_fine)) = -Inf;

    inf_mask = double(EV_max_d3 == -Inf);
    EV_safe = EV_max_d3; EV_safe(inf_mask > 0) = 0;
    EV_fine_full = interp1(d3_grid, EV_safe, d3prime_grid, 'linear');
    inf_fine_full = interp1(d3_grid, inf_mask, d3prime_grid, 'linear');
    EV_fine_full(inf_fine_full > 0) = -Inf;

    fine_lin_idx = fine_idx_tensor + reshape(0:N_z-1, [1, 1, N_z]) .* length(d3prime_grid);
    EV_bc_fine = reshape(EV_fine_full(fine_lin_idx(:)), size(fine_lin_idx));

    entireRHS_fine = ezc1_j .* temp2_fine + beta_j .* EV_bc_fine;
    RHS_fine = entireRHS_fine;
    valid_RHS_f = isfinite(entireRHS_fine) & (entireRHS_fine ~= 0);

    if ezc7_j == 1
        RHS_fine(valid_RHS_f) = ezc3 * entireRHS_fine(valid_RHS_f);
    else
        RHS_fine(valid_RHS_f) = ezc3 * (entireRHS_fine(valid_RHS_f).^ezc7_j);
    end
    RHS_fine(~isfinite(entireRHS_fine)) = -Inf;

    RHS_fine_flat = reshape(RHS_fine, [n2long, N_block * N_z]);
    [V_sub, maxindexL2] = max(RHS_fine_flat, [], 1);

    isInfLower = (RHS_fine_flat(1, :) == -Inf); isInfUpper = (RHS_fine_flat(end, :) == -Inf);
    inLowerStrict = (maxindexL2 >= 2) & (maxindexL2 <= n2short + 1);
    inUpperStrict = (maxindexL2 >= n2short + 3) & (maxindexL2 <= n2long - 1);

    L2flag = reshape(2 + (inLowerStrict & isInfLower) - (inUpperStrict & isInfUpper), [N_block, N_z]);
    V_sub = reshape(V_sub, [N_block, N_z]);
    Pol_d3_idx = reshape(midpoint, [N_block, N_z]);
    L2idx = reshape(maxindexL2, [N_block, N_z]);
else
    V_sub      = reshape(V_sub_coarse, [N_block, N_z]);
    Pol_d3_idx = reshape(d3_idx_coarse, [N_block, N_z]);
    L2idx = []; L2flag = [];
end

% RE-PACK COMBO INDEX (d2, d3)
d2_query_idx = Pol_d3_idx + repmat(reshape(0:N_z-1, [1, N_z]) .* N_d3, [N_block, 1]);
d2_opt = reshape(Pol_d2_idx(d2_query_idx(:)), [N_block, N_z]);
Pol_d_combo = d2_opt + (Pol_d3_idx - 1) * N_d2;


end