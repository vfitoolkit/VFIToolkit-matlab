function StationaryDist = StationaryDist_VFHorz_ExpAsset(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_z, N_j, z_gridvals_J, pi_z_J, Parameters, simoptions)
% STATIONARYDIST_VFHORZ_EXPASSET
% V-World Universal Forward Simulator for Experience Asset OLG Models (ExpAsset & ExpAssetz)

% --- 1. Dimension Extraction ---
l_dexperienceasset = 1;

n_d2 = n_d(end - l_dexperienceasset + 1 : end);
if length(n_d) > l_dexperienceasset
    n_d1 = n_d(1 : end - l_dexperienceasset);
    l_d1 = length(n_d1);
else
    n_d1 = [];
    l_d1 = 0;
end
l_d2 = length(n_d2);

if isscalar(n_a)
    n_a1 = 0;
    n_a2 = n_a;
    l_a1 = 0;
else
    n_a1 = n_a(1:end-1);
    n_a2 = n_a(end);
    l_a1 = length(n_a1);
end

N_a1 = max(1, prod(n_a1));
N_a2 = max(1, prod(n_a2));
N_z_safe = max(1, prod(n_z));
N_d1 = max(1, prod(n_d1));
N_d2 = max(1, prod(n_d2));

% --- 2. Setup Grids and Functions ---
aprimeFn = simoptions.aprimeFn;
d_grid = simoptions.d_grid;
a2_grid = simoptions.a_grid(sum(n_a1)+1:end);
d2_grid = d_grid(sum(n_d1)+1 : sum(n_d1)+sum(n_d2));
d2_gridvals = CreateGridvals(n_d2, d2_grid, 1);

input_names = getAnonymousFnInputNames(aprimeFn);
aprimeFnParamNames = input_names(isfield(Parameters, input_names));

% --- 3. Output Allocation ---
if isscalar(n_a)
    n_a_out = n_a2;
else
    n_a_out = [n_a1, n_a2];
end

StationaryDist = zeros([N_a1, N_a2, N_z_safe, N_j], 'like', jequaloneDist);
Dist_curr = reshape(jequaloneDist, [N_a1 * N_a2 * N_z_safe, 1]);

% Construct full-size state coordinate vectors
[~, A2_idx_grid, Z_idx_grid] = ndgrid(1:N_a1, 1:N_a2, 1:N_z_safe);
A2_grid_idx = A2_idx_grid(:);
Z_grid_idx  = Z_idx_grid(:);

% Policy tensor comes in as [NumPolicies, N_a1, N_a2, N_z_safe, N_j]
NumPolicies = size(Policy, 1);
Policy_reshaped = reshape(Policy, [NumPolicies, N_a1, N_a2, N_z_safe, N_j]);

% =========================================================
% TIME LOOP (FORWARD SIMULATION)
% =========================================================
for jj = 1:N_j
    % 1. Store the current cohort distribution
    StationaryDist(:, :, :, jj) = reshape(Dist_curr, [N_a1, N_a2, N_z_safe]);
    if jj == N_j
        break;
    end

    % 2. Extract Exact Policy Indexes for Current Age
    l_d = length(n_d);
    % Layer 1: d2 (experience asset decision is always the last 'd')
    d2_layer = reshape(Policy_reshaped(l_d, :, :, :, jj), [N_a1, N_a2, N_z_safe]);
    d2_linear_idx = max(1, min(d2_layer(:), N_d2)); % Clamp to safe bounds

    % Layer 2+: aprime (endogenous asset indexes)
    a1_linear_idx = zeros(N_a1 * N_a2 * N_z_safe, 1) + 1; % Base 1 index
    cum_n_a1 = 1;
    for ia = 1:length(n_a1)
        % Endogenous asset policies start immediately after 'd'
        pol_idx = reshape(Policy_reshaped(l_d + ia, :, :, :, jj), [N_a1, N_a2, N_z_safe]);
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
            z_val_col = z_work_j(:, iz);
            z_mesh_cells{iz} = z_val_col(z_idx_mesh);
        end
        a2_prime_vals = aprimeFn(d2_mesh, a2_mesh, z_mesh_cells{:}, aprimeFnParamsCell{:});
    else
        [d2_mesh, a2_mesh] = ndgrid(d2_gridvals(:), a2_grid(:));
        a2_prime_vals = aprimeFn(d2_mesh, a2_mesh, aprimeFnParamsCell{:});
    end

    expected_size = [N_d2, N_a2, N_z_safe];
    if numel(a2_prime_vals) > prod(expected_size)
        a2_prime_vals = reshape(a2_prime_vals(1:prod(expected_size)), expected_size);
    elseif ~isequal(size(a2_prime_vals), expected_size)
        a2_prime_vals = a2_prime_vals + zeros(expected_size, 'like', a2_grid);
    end
    a2_prime_vals = max(a2_grid(1), min(a2_grid(end), a2_prime_vals));

    [~, a2primeIndex] = histc(a2_prime_vals(:), a2_grid);
    a2primeIndex = max(1, min(a2primeIndex, N_a2 - 1));

    a2_step = a2_grid(a2primeIndex + 1) - a2_grid(a2primeIndex);
    a2_step(a2_step == 0) = 1;
    a2primeProbs = (a2_grid(a2primeIndex + 1) - a2_prime_vals(:)) ./ a2_step;
    a2primeProbs = max(0, min(1, a2primeProbs));

    lookup_idx = d2_linear_idx(:) + N_d2 * (A2_grid_idx - 1) + N_d2 * N_a2 * (Z_grid_idx(:) - 1);

    a2_p_lower = a2primeIndex(lookup_idx);
    a2_p_upper = min(a2_p_lower + 1, N_a2);
    a2_prob_lower = a2primeProbs(lookup_idx);

    % 4 & 5. Map Mass forward (with Grid Interpolation Support)
    sz_mid = N_a1 * N_a2 * N_z_safe;

    if isfield(simoptions, 'gridinterplayer') && simoptions.gridinterplayer == 1
        % Extract L2 index to preserve fractional wealth
        l2_layer = reshape(Policy_reshaped(end-1, :, :, :, jj), [N_a1, N_a2, N_z_safe]);
        a1_prob_upper = (l2_layer(:) - 1) / (simoptions.ngridinterp + 1);

        % Four-way target mapping (splitting between a1 and a2 bounds)
        idx_LL = a1_linear_idx(:) + N_a1 * (a2_p_lower(:) - 1) + N_a1 * N_a2 * (Z_grid_idx(:) - 1);
        idx_UL = min(a1_linear_idx(:) + 1, N_a1) + N_a1 * (a2_p_lower(:) - 1) + N_a1 * N_a2 * (Z_grid_idx(:) - 1);
        idx_LU = a1_linear_idx(:) + N_a1 * (a2_p_upper(:) - 1) + N_a1 * N_a2 * (Z_grid_idx(:) - 1);
        idx_UU = min(a1_linear_idx(:) + 1, N_a1) + N_a1 * (a2_p_upper(:) - 1) + N_a1 * N_a2 * (Z_grid_idx(:) - 1);

        mass_LL = Dist_curr(:) .* a2_prob_lower(:) .* (1 - a1_prob_upper(:));
        mass_UL = Dist_curr(:) .* a2_prob_lower(:) .* a1_prob_upper(:);
        mass_LU = Dist_curr(:) .* (1 - a2_prob_lower(:)) .* (1 - a1_prob_upper(:));
        mass_UU = Dist_curr(:) .* (1 - a2_prob_lower(:)) .* a1_prob_upper(:);

        Dist_mid_flat = accumarray([idx_LL; idx_UL; idx_LU; idx_UU], double([mass_LL; mass_UL; mass_LU; mass_UU]), [sz_mid, 1]);
    else
        % Standard discrete 2-way split (a2 only)
        idx_L = a1_linear_idx(:) + N_a1 * (a2_p_lower(:) - 1) + N_a1 * N_a2 * (Z_grid_idx(:) - 1);
        idx_U = a1_linear_idx(:) + N_a1 * (a2_p_upper(:) - 1) + N_a1 * N_a2 * (Z_grid_idx(:) - 1);

        mass_L = Dist_curr(:) .* a2_prob_lower(:);
        mass_U = Dist_curr(:) .* (1 - a2_prob_lower(:));

        Dist_mid_flat = accumarray([idx_L; idx_U], double([mass_L; mass_U]), [sz_mid, 1]);
    end

    Dist_mid = reshape(cast(Dist_mid_flat, 'like', Dist_curr), [N_a1 * N_a2, N_z_safe]);

    % 6. Apply Exogenous Markov Shocks (z)
    if N_z_safe > 1
        pi_z = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
        Dist_next_z = reshape(Dist_mid, [N_a1 * N_a2, N_z_safe]) * pi_z;
        Dist_curr = reshape(Dist_next_z, [N_a1 * N_a2 * N_z_safe, 1]);
    else
        Dist_curr = reshape(Dist_mid, [N_a1 * N_a2 * N_z_safe, 1]);
    end

    % 7. Apply Age Weights (The Grim Reaper) ---
    for aw = 1:length(AgeWeightParamNames)
        weight_name = AgeWeightParamNames{aw};
        if isfield(Parameters, weight_name)
            weight_vals = Parameters.(weight_name);
            % The actual survival rate is the ratio of the next cohort's mass to the current one
            if jj < length(weight_vals)
                survival_rate = weight_vals(jj+1) / weight_vals(jj);
                Dist_curr = Dist_curr * survival_rate;
            else
                Dist_curr = Dist_curr * 0; % Terminal period
            end
        end
    end

end

% =========================================================
% OUTPUT UNPACKING
% =========================================================
if N_z_safe > 1
    StationaryDist = reshape(StationaryDist, [n_a_out, n_z, N_j]);
else
    StationaryDist = reshape(StationaryDist, [n_a_out, N_j]);
end

end