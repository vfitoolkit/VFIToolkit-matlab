function [V, Policy] = ValueFnIter_VFHorz_ExpAsset(n_d1, n_d2, n_a1, n_a2, n_z, N_j, d1_gridvals, d2_gridvals, a1_gridvals, a2_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)
N_d1 = prod(n_d1);
N_d2 = prod(n_d2);
N_a1 = prod(n_a1);
N_a2 = prod(n_a2);
N_a = N_a1 * N_a2;
N_z = prod(n_z);
N_d1_safe = max(1, N_d1);
N_z_safe = max(1, N_z);

if vfoptions.parallel == 2
    d1_gridvals = gpuArray(d1_gridvals);
    d2_gridvals = gpuArray(d2_gridvals);
    a1_gridvals = gpuArray(a1_gridvals);
    a2_grid = gpuArray(a2_grid);
    z_gridvals_J = gpuArray(z_gridvals_J);
    pi_z_J = gpuArray(pi_z_J);
end

aprimeFn = vfoptions.aprimeFn;
num_d2 = length(n_d2);
num_a2 = length(n_a2);
temp = getAnonymousFnInputNames(aprimeFn);
if length(temp) > (num_d2 + num_a2 + (num_a2 >= 2))
    aprimeFnParamNames = {temp{num_d2 + num_a2 + (num_a2 >= 2) + 1:end}};
else
    aprimeFnParamNames = {};
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
    apr_in = gpuArray(0);
    A1_cells = {};
end

a2_gridvals = CreateGridvals(n_a2, a2_grid, 1);
A2_cells = cell(1, num_a2);
for i = 1:num_a2
    A2_cells{i} = shiftdim(a2_gridvals(:, i), -3);
end

V = zeros(N_a1, N_a2, N_z_safe, N_j, 'like', a2_grid);
PolicyKron = zeros(N_a1, N_a2, N_z_safe, N_j, 'like', a2_grid);
V_next = zeros(N_a1, N_a2, N_z_safe, 'like', a2_grid);

for reverse_j = 0:N_j-1
    jj = N_j - reverse_j;

    if jj == N_j && isfield(vfoptions, 'V_Jplus1') && ~isempty(vfoptions.V_Jplus1)
        V_next = reshape(gpuArray(vfoptions.V_Jplus1), [N_a1, N_a2, N_z_safe]);
    end

    DiscountFactorParamsVec = CreateVectorFromParams(Parameters, DiscountFactorParamNames, jj);
    beta_j = prod(DiscountFactorParamsVec);
    ReturnFnParamsVec = CreateVectorFromParams(Parameters, ReturnFnParamNames, jj);
    if ~iscell(ReturnFnParamsVec)
        ReturnFnParamsVec = num2cell(ReturnFnParamsVec);
    end
    aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);

    num_z = length(n_z);
    if N_z > 0
        if size(z_gridvals_J, 3) > 1
            z_work_j = z_gridvals_J(:, :, jj);
        else
            z_work_j = z_gridvals_J(:, :, 1);
        end
        Z_cells = cell(1, num_z);
        for i = 1:num_z
            Z_cells{i} = shiftdim(z_work_j(:, i), -4);
        end
        pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    else
        Z_cells = {};
        pi_z_j = [];
    end

    [a2primeIndex, a2primeProbs] = CreateExperienceAssetFnMatrix(aprimeFn, n_d2, n_a2, d2_gridvals, a2_grid, aprimeFnParamsVec, 2);

    V_j_max = -inf(N_a1, N_a2, N_z_safe, 'like', a2_grid);
    Pol_apr_max = ones(N_a1, N_a2, N_z_safe, 'like', a2_grid);
    Pol_d1_max = ones(N_a1, N_a2, N_z_safe, 'like', a2_grid);
    Pol_d2_max = ones(N_a1, N_a2, N_z_safe, 'like', a2_grid);

    for i_d2 = 1:N_d2
        idx = a2primeIndex(i_d2, :);
        probs = a2primeProbs(i_d2, :);

        idx_rs = reshape(idx, [1, N_a2, 1]);
        probs_rs = reshape(probs, [1, N_a2, 1]);

        idx_lower = idx;
        idx_upper = min(idx + 1, N_a2);

        Vlower = V_next(:, idx_lower, :);
        Vupper = V_next(:, idx_upper, :);

        EV_interp = probs_rs .* Vlower + (1 - probs_rs) .* Vupper;

        if N_z > 0
            EV_flat = reshape(EV_interp, [N_a1 * N_a2, N_z]);
            EV_d2 = reshape(EV_flat * pi_z_j', [N_a1, N_a2, N_z]);
        else
            EV_d2 = EV_interp;
        end

        D2_cells = cell(1, num_d2);
        for i = 1:num_d2
            D2_cells{i} = d2_gridvals(i_d2, i);
        end

        F_tensor = ReturnFn(D1_cells{:}, D2_cells{:}, apr_in, A1_cells{:}, A2_cells{:}, Z_cells{:}, ReturnFnParamsVec{:});

        EV_d2_bc = reshape(EV_d2, [N_a1, 1, 1, N_a2, N_z_safe]);
        RHS = F_tensor + beta_j .* EV_d2_bc;

        RHS_flat = reshape(RHS, [N_a1 * N_d1_safe, N_a1 * N_a2 * N_z_safe]);
        [V_sub, Pol_sub_idx] = max(RHS_flat, [], 1);
        V_sub = reshape(V_sub, [N_a1, N_a2, N_z_safe]);

        if N_d1 > 0
            apr_idx = mod(Pol_sub_idx - 1, N_a1) + 1;
            d1_idx = ceil(Pol_sub_idx / N_a1);
        else
            apr_idx = Pol_sub_idx;
            d1_idx = ones(size(Pol_sub_idx), 'like', Pol_sub_idx);
        end

        apr_idx = reshape(apr_idx, [N_a1, N_a2, N_z_safe]);
        d1_idx = reshape(d1_idx, [N_a1, N_a2, N_z_safe]);

        if i_d2 == 1
            update_mask = true(N_a1, N_a2, N_z_safe);
        else
            update_mask = V_sub > V_j_max;
        end

        V_j_max(update_mask) = V_sub(update_mask);
        Pol_apr_max(update_mask) = apr_idx(update_mask);
        Pol_d1_max(update_mask) = d1_idx(update_mask);
        Pol_d2_max(update_mask) = i_d2;
    end

    d_idx = Pol_d1_max + (Pol_d2_max - 1) * N_d1_safe;
    PolicyKron_j = d_idx + (Pol_apr_max - 1) * (N_d1_safe * N_d2);
    
    V(:, :, :, jj) = V_j_max;
    PolicyKron(:, :, :, jj) = PolicyKron_j;
    V_next = V_j_max;
end

V = reshape(V, [N_a, N_z_safe, N_j]);
PolicyKron = reshape(PolicyKron, [N_a, N_z_safe, N_j]);

if vfoptions.outputkron == 1
    Policy = PolicyKron;
else
    PolicyKron = shiftdim(PolicyKron, -1);
    
    if n_d1 > 0
        n_d_vec = [n_d1, n_d2];
    else
        n_d_vec = n_d2;
    end
    
    if n_a1 > 0 && n_a1(1) > 0
        n_d_vec = [n_d_vec, n_a1];
        n_a_vec = [n_a1, n_a2];
    else
        n_a_vec = n_a2;
    end
    
    if N_z == 0
        V = reshape(V, [N_a, N_j]);
        Policy = UnKronPolicyIndexes1_FHorz_noz(PolicyKron, n_d_vec, n_a_vec, N_j, vfoptions);
    else
        Policy = UnKronPolicyIndexes1_FHorz_z(PolicyKron, n_d_vec, n_a_vec, n_z, N_j, vfoptions);
    end
end


end
