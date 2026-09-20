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

% --- DYNAMIC EXTRACTION OF EZC9 ---
ezc9 = 1;
if isfield(vfoptions, 'ezc9')
    ezc9 = vfoptions.ezc9;
end

% --- SMART nargin PARSER FOR RISKY ASSET aprimeFn ---
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
    aprime_clamped = max(min(aprime_tensor, a2_grid(end)), a2_grid(1));

    % ---------------------------------------------------------
    % Warm Glow of Bequests
    % ---------------------------------------------------------
    if warmglow == 1
        WG_params = CreateCellFromParams(Parameters, vfoptions.WarmGlowBequestsFnParamsNames, jj);
        WG_raw = vfoptions.WarmGlowBequestsFn(a2_grid, WG_params{:});
        if isscalar(WG_raw)
            WG_raw = WG_raw * ones(size(a2_grid), 'like', a2_grid);
        end
        WG_raw = reshape(WG_raw, size(a2_grid));

        WG_temp = WG_raw;
        valid_wg = isfinite(WG_raw);
        WG_temp(valid_wg) = (ezc4 * WG_raw(valid_wg)) .^ ezc5(jj);
        WG_temp(WG_raw == 0) = 0;

        inf_mask_wg = (WG_temp == -Inf);
        WG_safe = WG_temp;
        WG_safe(inf_mask_wg) = 0;

        WG_interp = interp1(a2_grid, WG_safe, aprime_clamped(:), 'linear');
        inf_interp_wg = interp1(a2_grid, double(inf_mask_wg), aprime_clamped(:), 'linear');
        WG_interp(inf_interp_wg > 0) = -Inf;
        WG_interp = reshape(WG_interp, [N_d2*N_d3, N_u]);

        inf_mask_wg_u = (WG_interp == -Inf);
        WG_safe_u = WG_interp;
        WG_safe_u(inf_mask_wg_u) = 0;

        pi_u_rs = reshape(pi_u, [1, N_u]);
        WG_u = sum(WG_safe_u .* pi_u_rs, 2);
        inf_infect_wg_u = double(inf_mask_wg_u) .* double(pi_u_rs > 0);
        WG_u(sum(inf_infect_wg_u, 2) > 0) = -Inf;
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
            % FIX: Removed the inverted becareful logic!
            temp_WG(isfinite(WG_u)) = ( (1 - sj(jj)) * WG_u(isfinite(WG_u)).^ezc8(jj) ) .^ ezc6(jj);
            temp_WG(WG_u == 0) = 0;
            temp4 = temp_WG;
        else
            temp4 = zeros(N_d2*N_d3, 1, 'like', a2_grid);
        end
        temp4 = repmat(reshape(temp4, [1, N_d2*N_d3, 1]), [N_a1, 1, max(N_z,1)]);
    else
        V_next_3D = reshape(V_next, [N_a1, N_a2, max(N_z,1)]);
        V_interp = zeros(N_a1, N_d2*N_d3, N_u, max(N_z,1), 'like', a2_grid);

        for i_a1 = 1:N_a1
            V_slice = squeeze(V_next_3D(i_a1, :, :));
            if max(N_z,1) == 1, V_slice = V_slice(:); end

            temp_V = V_slice;
            % FIX: Match legacy exactly, no becareful variable here!
            temp_V(isfinite(V_slice)) = (ezc4 * V_slice(isfinite(V_slice))) .^ ezc5(jj);
            temp_V(V_slice == 0) = 0;

            inf_mask = double(temp_V == -Inf);
            V_safe = temp_V;
            V_safe(inf_mask > 0) = 0;

            V_int_slice = interp1(a2_grid, V_safe, aprime_clamped(:), 'linear');
            inf_int_slice = interp1(a2_grid, inf_mask, aprime_clamped(:), 'linear');
            V_int_slice(inf_int_slice > 0) = -Inf;
            V_interp(i_a1, :, :, :) = reshape(V_int_slice, [1, N_d2*N_d3, N_u, max(N_z,1)]);
        end

        pi_u_rs = reshape(pi_u, [1, 1, N_u, 1]);
        inf_mask_u = (V_interp == -Inf);
        V_interp_safe = V_interp;
        V_interp_safe(inf_mask_u) = 0;
        EV_u = sum(V_interp_safe .* pi_u_rs, 3);
        inf_infect_u = double(inf_mask_u) .* double(pi_u_rs > 0);
        EV_u(sum(inf_infect_u, 3) > 0) = -Inf;
        EV_u = reshape(EV_u, [N_a1, N_d2*N_d3, max(N_z,1)]);

        if N_z > 0
            EV_z = zeros(N_a1, N_d2*N_d3, N_z, 'like', a2_grid);
            pi_z_j = pi_z_J(:,:,jj);
            for i_a1 = 1:N_a1
                EV_u_slice = squeeze(EV_u(i_a1, :, :));
                if N_z == 1, EV_u_slice = EV_u_slice(:)'; end
                inf_mask_z = (EV_u_slice == -Inf);
                EV_u_safe = EV_u_slice;
                EV_u_safe(inf_mask_z) = 0;

                EV_z_slice = EV_u_safe * pi_z_j';
                inf_infect_z = double(inf_mask_z) * double(pi_z_j' > 0);
                EV_z_slice(inf_infect_z > 0) = -Inf;
                EV_z(i_a1, :, :) = EV_z_slice;
            end
        else
            EV_z = EV_u;
        end

        temp4 = EV_z;
        if warmglow == 1
            WG_u_rs = reshape(WG_u, [1, N_d2*N_d3, 1]);
            becareful = logical(isfinite(temp4) .* isfinite(WG_u_rs));
            temp4(becareful) = ( sj(jj)*temp4(becareful).^ezc8(jj) + (1-sj(jj))*WG_u_rs(becareful).^ezc8(jj) ) .^ ezc6(jj);
            temp4((EV_z == 0) & (WG_u_rs == 0)) = 0;
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

    % The Legacy Toolkit Parity Block: NaN masking and ezc9 * ezc3 flip
    masked_temp4 = (~isinf(temp4_tensor)) .* temp4_tensor;
    flipped_temp4 = ezc9 * ezc3 * masked_temp4;

    [EV_max_d3_raw, Pol_d2_idx] = max(flipped_temp4, [], 2);

    % Keep it as the raw positive output to match legacy RHS assembly
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

n_daprime = n_d;
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

% 1. Setup 5D Choice/State Structures
d1_in      = reshape(d1_grid, [N_d1, 1, 1, 1, 1]);
a1prime_in = reshape(a1_grid, [1, N_a1, 1, 1, 1]);
d3_in      = reshape(d3_grid, [1, 1, N_d3, 1, 1]);

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

% 3. Evaluate F (5D) matching legacy becareful logic
F_tensor = TensorReturnFn(ReturnFn_Args{:});
temp2 = F_tensor;
becareful = logical(isfinite(F_tensor) .* (F_tensor ~= 0));
temp2(becareful) = F_tensor(becareful) .^ ezc2_j;
temp2(F_tensor == 0) = -Inf;

% 4. Assemble RHS matching legacy summation bugs
EV_bc = reshape(EV_max_d3, [1, N_a1, N_d3, 1, N_z_safe]);
entireRHS = ezc1_j .* temp2 + ezc9 .* beta_j .* EV_bc;

RHS = entireRHS;
temp5 = logical(isfinite(entireRHS) .* (entireRHS ~= 0));
RHS(temp5) = entireRHS(temp5) .^ ezc7_j;
% FIX: Removed the RHS(~isfinite) = -Inf override. Let MATLAB handle NaNs natively!

RHS_flat = reshape(RHS, [N_d1 * N_a1 * N_d3, N_block * N_z_safe]);
[V_sub_coarse, opt_idx_flat] = max(RHS_flat, [], 1);
V_sub = V_sub_coarse;

% 5. Simultaneous Compression (Flatten all choices)
[d1_opt, a1prime_opt, d3_opt] = ind2sub([N_d1, N_a1, N_d3], opt_idx_flat);
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
