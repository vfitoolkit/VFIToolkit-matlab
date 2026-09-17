function StationaryDist = StationaryDist_VFHorz_ExpAssetsemiz(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, N_j, pi_semiz_J, pi_z_J, Parameters, simoptions)
% STATIONARYDIST_VFHORZ_EXPASSETSEMIZ
% V-World Universal Forward Simulator for Experience Asset + Semi-Exogenous OLG Models

if ~isfield(simoptions, 'optimize_nProbs')
    simoptions.optimize_nProbs = 0;
end

% --- 1. Dimension Extraction ---
l_dsemiz = 1;
if isfield(simoptions, 'l_dsemiz'), l_dsemiz = simoptions.l_dsemiz; end
l_dexperienceasset = 1;
if isfield(simoptions, 'l_dexperienceasset'), l_dexperienceasset = simoptions.l_dexperienceasset; end

% Extract decision partitions (d3=semiz, d2=expasset, d1=other)
n_d3 = n_d(end - l_dsemiz + 1 : end);
n_d2 = n_d(end - l_dexperienceasset - l_dsemiz + 1 : end - l_dsemiz);
if length(n_d) > l_dexperienceasset + l_dsemiz
    n_d1 = n_d(1 : end - l_dexperienceasset - l_dsemiz);
    l_d1 = length(n_d1);
else
    n_d1 = []; l_d1 = 0;
end

if isscalar(n_a)
    n_a1 = 0; n_a2 = n_a; l_a1 = 0;
else
    n_a1 = n_a(1:end-1); n_a2 = n_a(end); l_a1 = length(n_a1);
end

N_a1 = max(1, prod(n_a1));
N_a2 = max(1, prod(n_a2));
N_semiz = max(1, prod(n_semiz));
N_z_safe = max(1, prod(n_z));
N_d2 = max(1, prod(n_d2));
N_d3 = max(1, prod(n_d3));
N_states = N_a1 * N_a2 * N_semiz * N_z_safe;

% --- 2. Setup Grids and Functions ---
aprimeFn = simoptions.aprimeFn;
a2_grid = simoptions.a_grid(sum(n_a1)+1:end);
d2_grid = simoptions.d_grid(sum(n_d1)+1 : sum(n_d1)+sum(n_d2));
d2_gridvals = CreateGridvals(n_d2, d2_grid, 1);

% Introspect the original, un-bridged function so we get the real variable names
input_names = getAnonymousFnInputNames(aprimeFn);
aprimeFnParamNames = input_names(isfield(Parameters, input_names));

% Build the tensor bridge
TensoraprimeFn = CreateTensorBridge(aprimeFn);

% --- 3. Pre-Process pi_semiz_J into a Dense Vectorized Map ---
N_sz = size(pi_semiz_J, 1);
N_dsz = size(pi_semiz_J, 3);
N_j_pi = size(pi_semiz_J, 4);

max_trans = max(sum(pi_semiz_J > 0, 2), [], 'all'); % Max possible transitions
sz_to_idx = ones(N_sz, N_dsz, N_j_pi, max_trans);
sz_prob   = zeros(N_sz, N_dsz, N_j_pi, max_trans, 'like', pi_semiz_J);

for j_pi = 1:N_j_pi
    for d_sz = 1:N_dsz
        for sz_from = 1:N_sz
            trans_probs = pi_semiz_J(sz_from, :, d_sz, j_pi);
            valid_idx = find(trans_probs > 0);
            n_v = length(valid_idx);
            if n_v > 0
                sz_to_idx(sz_from, d_sz, j_pi, 1:n_v) = valid_idx;
                sz_prob(sz_from, d_sz, j_pi, 1:n_v) = trans_probs(valid_idx);
            end
        end
    end
end
sz_to_idx = cast(sz_to_idx, 'like', pi_semiz_J); % Push to GPU if necessary

% --- 4. Output Allocation ---
if isscalar(n_a)
    n_a_out = n_a2;
else
    n_a_out = [n_a1, n_a2];
end
StationaryDist = zeros([N_a1, N_a2, N_semiz, N_z_safe, N_j], 'like', jequaloneDist);
Dist_curr = reshape(jequaloneDist, [N_states, 1]);

% Construct full-size state coordinate vectors
[~, A2_idx_grid, SZ_idx_grid, Z_idx_grid] = ndgrid(1:N_a1, 1:N_a2, 1:N_semiz, 1:N_z_safe);
A2_grid_idx = A2_idx_grid(:);
SZ_grid_idx = SZ_idx_grid(:);
Z_grid_idx  = Z_idx_grid(:);

NumPolicies = size(Policy, 1);
Policy_reshaped = reshape(Policy, [NumPolicies, N_a1, N_a2, N_semiz, N_z_safe, N_j]);

% --- BYPASS EXOG SHOCK SETUP ---
if isfield(simoptions, 'n_z')
    z_gridvals_J = simoptions.z_grid;
else
    z_gridvals_J = [];
end

% =========================================================
% TIME LOOP (FORWARD SIMULATION)
% =========================================================
total_zeros_created = 0;
jj_at_max_a2 = 0;
l_d = length(n_d);

for jj = 1:N_j

    % 1. Extract, Optimize, and Store the current cohort distribution
    StationaryDist_jj = reshape(Dist_curr, [N_a1, N_a2, N_semiz * N_z_safe]);
    if simoptions.optimize_nProbs == 1
        [StationaryDist_jj, total_zeros_created, jj_at_max_a2] = StationaryDist_FHorz_Optimize_nProbs_raw(...
            StationaryDist_jj, n_a1, n_a2, N_semiz * N_z_safe, jj, 10, total_zeros_created, jj_at_max_a2, simoptions);
        Dist_curr = reshape(StationaryDist_jj, [N_states, 1]);
    end
    StationaryDist(:, :, :, :, jj) = reshape(StationaryDist_jj, [N_a1, N_a2, N_semiz, N_z_safe]);
    if jj == N_j; break; end

    % 2. Extract Exact Policy Indexes for Current Age
    d3_layer = reshape(Policy_reshaped(l_d, :,:,:,:, jj), [N_states, 1]);
    d3_linear_idx = max(1, min(d3_layer, N_d3));

    d2_layer = reshape(Policy_reshaped(l_d - l_dsemiz, :,:,:,:, jj), [N_states, 1]);
    d2_linear_idx = max(1, min(d2_layer, N_d2));

    a1_linear_idx = zeros(N_states, 1) + 1;
    cum_n_a1 = 1;
    for ia = 1:length(n_a1)
        pol_idx = reshape(Policy_reshaped(l_d + ia, :,:,:,:, jj), [N_states, 1]);
        a1_linear_idx = a1_linear_idx + cum_n_a1 * (pol_idx(:) - 1);
        cum_n_a1 = cum_n_a1 * n_a1(ia);
    end
    a1_linear_idx = max(1, min(a1_linear_idx, N_a1));

    % 3. Calculate Experience Asset Transition (a2)
    aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);
    if N_z_safe > 1
        z_work_j = z_gridvals_J(:, :, min(jj, size(z_gridvals_J, 3)));
        [d2_mesh, a2_mesh, z_idx_mesh] = ndgrid(d2_gridvals(:), a2_grid(:), 1:N_z_safe);
        z_mesh_cells = cell(1, length(n_z));
        for iz = 1:length(n_z)
            z_mesh_cells{iz} = z_work_j(z_idx_mesh, iz);
        end
        
        % Dynamically calculate dummy variables needed
        num_expected_args = nargin(aprimeFn);
        num_provided_args = 2 + length(z_mesh_cells) + length(aprimeFnParamsCell);
        num_dummy_args    = max(0, num_expected_args - num_provided_args);
        dummy_padding     = num2cell(zeros(1, num_dummy_args));
        
        % Evaluate the bridged tensor function
        a2_prime_vals = TensoraprimeFn(d2_mesh, a2_mesh, z_mesh_cells{:}, dummy_padding{:}, aprimeFnParamsCell{:});
    else
        [d2_mesh, a2_mesh] = ndgrid(d2_gridvals(:), a2_grid(:));

        % Dynamically calculate how many dummy variables are needed to satisfy the wrapper
        num_expected_args = nargin(aprimeFn);
        num_provided_args = 2 + length(aprimeFnParamsCell);
        num_dummy_args    = max(0, num_expected_args - num_provided_args);

        % Generate a general cell array of zeros to pad the function call
        dummy_padding = num2cell(zeros(1, num_dummy_args));

        % Evaluate the bridged tensor function universally
        a2_prime_vals = TensoraprimeFn(d2_mesh, a2_mesh, dummy_padding{:}, aprimeFnParamsCell{:});
    end

    expected_size = [N_d2, N_a2, N_z_safe];
    if ~isequal(size(a2_prime_vals), expected_size)
        a2_prime_vals = reshape(a2_prime_vals(1:prod(expected_size)), expected_size);
    end
    a2_prime_vals = max(a2_grid(1), min(a2_grid(end), a2_prime_vals));
    [~, a2primeIndex] = histc(a2_prime_vals(:), a2_grid);
    a2primeIndex = max(1, min(a2primeIndex, N_a2 - 1));
    a2_step = a2_grid(a2primeIndex + 1) - a2_grid(a2primeIndex);
    a2_step(a2_step == 0) = 1;
    a2primeProbs = max(0, min(1, (a2_grid(a2primeIndex + 1) - a2_prime_vals(:)) ./ a2_step));

    lookup_idx = d2_linear_idx(:) + N_d2 * (A2_grid_idx(:) - 1) + N_d2 * N_a2 * (Z_grid_idx(:) - 1);
    a2_p_lower = a2primeIndex(lookup_idx);
    a2_p_upper = min(a2_p_lower + 1, N_a2);
    a2_prob_lower = a2primeProbs(lookup_idx);

    % 4. Pre-Fetch Semi-Exogenous Transitions for all states simultaneously
    j_pi_sz = min(jj, N_j_pi);
    sz_to_local = zeros(N_states, max_trans, 'like', sz_to_idx);
    sz_pr_local = zeros(N_states, max_trans, 'like', sz_prob);
    for tr = 1:max_trans
        % Linear index into the [N_sz, N_dsz, N_j_pi, max_trans] lookup table
        lin_idx_sz = SZ_grid_idx + N_sz*(d3_linear_idx - 1) + N_sz*N_dsz*(j_pi_sz - 1) + N_sz*N_dsz*N_j_pi*(tr - 1);
        sz_to_local(:, tr) = sz_to_idx(lin_idx_sz);
        sz_pr_local(:, tr) = sz_prob(lin_idx_sz);
    end

    % 5. Map Mass Forward (Tensor-Product of A1, A2, and SemiZ dispersion)
    is_gridinterp = isfield(simoptions, 'gridinterplayer') && simoptions.gridinterplayer == 1;
    if is_gridinterp
        l2_layer = reshape(Policy_reshaped(end-1, :,:,:,:, jj), [N_states, 1]);
        a1_prob_upper = (l2_layer(:) - 1) / (simoptions.ngridinterp + 1);
        idx_L_base  = a1_linear_idx + N_a1 * (a2_p_lower - 1);
        idx_U_base  = min(a1_linear_idx + 1, N_a1) + N_a1 * (a2_p_lower - 1);
        idx_LU_base = a1_linear_idx + N_a1 * (a2_p_upper - 1);
        idx_UU_base = min(a1_linear_idx + 1, N_a1) + N_a1 * (a2_p_upper - 1);
        mass_L  = Dist_curr .* a2_prob_lower .* (1 - a1_prob_upper);
        mass_U  = Dist_curr .* a2_prob_lower .* a1_prob_upper;
        mass_LU = Dist_curr .* (1 - a2_prob_lower) .* (1 - a1_prob_upper);
        mass_UU = Dist_curr .* (1 - a2_prob_lower) .* a1_prob_upper;
        alloc_mult = 4;
    else
        idx_L_base = a1_linear_idx + N_a1 * (a2_p_lower - 1);
        idx_U_base = a1_linear_idx + N_a1 * (a2_p_upper - 1);
        mass_L = Dist_curr .* a2_prob_lower;
        mass_U = Dist_curr .* (1 - a2_prob_lower);
        alloc_mult = 2;
    end

    % Preallocate Accumulation Arrays to prevent dynamic reshaping
    idx_acc = zeros(N_states * max_trans * alloc_mult, 1, 'double');
    mass_acc = zeros(N_states * max_trans * alloc_mult, 1, 'like', Dist_curr);
    counter = 0;

    for tr = 1:max_trans
        valid = (sz_pr_local(:, tr) > 0);
        if ~any(valid); continue; end

        sz_dest = sz_to_local(valid, tr);
        sz_p    = sz_pr_local(valid, tr);

        sz_z_offset = N_a1*N_a2 * (sz_dest - 1) + N_a1*N_a2*N_semiz * (Z_grid_idx(valid) - 1);

        if is_gridinterp
            v_idx = [idx_L_base(valid) + sz_z_offset; idx_U_base(valid) + sz_z_offset; ...
                idx_LU_base(valid) + sz_z_offset; idx_UU_base(valid) + sz_z_offset];
            v_mass = [mass_L(valid).*sz_p; mass_U(valid).*sz_p; mass_LU(valid).*sz_p; mass_UU(valid).*sz_p];
        else
            v_idx = [idx_L_base(valid) + sz_z_offset; idx_U_base(valid) + sz_z_offset];
            v_mass = [mass_L(valid).*sz_p; mass_U(valid).*sz_p];
        end

        len = length(v_idx);
        idx_acc(counter+1 : counter+len) = v_idx;
        mass_acc(counter+1 : counter+len) = v_mass;
        counter = counter + len;
    end

    Dist_mid_flat = accumarray(idx_acc(1:counter), double(mass_acc(1:counter)), [N_states, 1]);
    Dist_mid_flat = cast(Dist_mid_flat, 'like', Dist_curr);

    % 6. Apply Exogenous Markov Shocks (z)
    if N_z_safe > 1
        pi_z = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        Dist_next_z = reshape(Dist_mid_flat, [N_a1 * N_a2 * N_semiz, N_z_safe]) * pi_z;
        Dist_curr = reshape(Dist_next_z, [N_states, 1]);
    else
        Dist_curr = Dist_mid_flat;
    end

    % 7. Apply Age Weights (The Grim Reaper)
    for aw = 1:length(AgeWeightParamNames)
        weight_name = AgeWeightParamNames{aw};
        if isfield(Parameters, weight_name)
            weight_vals = Parameters.(weight_name);
            if jj < length(weight_vals)
                Dist_curr = Dist_curr * (weight_vals(jj+1) / weight_vals(jj));
            else
                Dist_curr = Dist_curr * 0;
            end
        end
    end
end

% --- nProbs Total Summary Report ---
if simoptions.optimize_nProbs == 1 && isfield(simoptions, 'verbose') && simoptions.verbose >= 1
    fprintf('nProbs Optimization: Total zeros created across all ages = %d\n', total_zeros_created);
end

% =========================================================
% OUTPUT UNPACKING
% =========================================================
if N_z_safe > 1
    StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, n_z, N_j]);
else
    StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, N_j]);
end

end