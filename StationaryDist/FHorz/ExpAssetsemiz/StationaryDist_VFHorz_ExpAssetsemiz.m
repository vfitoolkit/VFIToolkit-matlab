function StationaryDist = StationaryDist_VFHorz_ExpAssetsemiz(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, N_j, pi_semiz_J, pi_z_J, Parameters, simoptions)

% STATIONARYDIST_VFHORZ_EXPASSETSEMIZ
% V-World Universal Forward Simulator for Experience Asset + Semi-Exogenous OLG Models

% --- 1. Dimension Extraction ---
l_dsemiz = simoptions.l_dsemiz;
if ~isfield(simoptions,'l_dexperienceasset')
    l_dexperienceasset=1;
else
    l_dexperienceasset = simoptions.l_dexperienceasset;
end

n_d3 = n_d(end - l_dsemiz + 1 : end);
n_d2 = n_d(end - l_dexperienceasset - l_dsemiz + 1 : end - l_dsemiz);
if length(n_d) > l_dexperienceasset + l_dsemiz
    n_d1 = n_d(1 : end - l_dexperienceasset - l_dsemiz);
    l_d1 = length(n_d1);
else
    n_d1 = []; l_d1 = 0;
end
l_d2 = length(n_d2); l_d3 = length(n_d3); l_d = length(n_d);

if isscalar(n_a)
    n_a1 = 0; n_a2 = n_a; l_a1 = 0;
else
    n_a1 = n_a(1:end-1); n_a2 = n_a(end); l_a1 = length(n_a1);
end

N_a1 = max(1, prod(n_a1)); N_a2 = max(1, prod(n_a2));
N_semiz_safe = max(1, prod(n_semiz)); N_z_safe = max(1, prod(n_z));
N_d1 = max(1, prod(n_d1)); N_d2 = max(1, prod(n_d2)); N_d3 = max(1, prod(n_d3));

if isfield(simoptions, 'gridinterplayer') && simoptions.gridinterplayer(1) == 1
    error('Grid interpolation not yet supported in V-World StationaryDist_ExpAssetsemiz.');
end

% --- 2. Setup Grids and Functions ---
aprimeFn = simoptions.aprimeFn;
d_grid = simoptions.d_grid;
a2_grid = simoptions.a_grid(sum(n_a1)+1:end);
d2_gridvals = CreateGridvals(n_d2, d_grid(l_d1+1 : l_d1+l_d2), 1);

input_names = getAnonymousFnInputNames(aprimeFn);
aprimeFnParamNames = input_names(isfield(Parameters, input_names));

% --- 3. Output Allocation ---
if isscalar(n_a)
    n_a_out = n_a2;
else
    n_a_out = [n_a1, n_a2];
end
StationaryDist = zeros([N_a1, N_a2, N_semiz_safe, N_z_safe, N_j], 'like', jequaloneDist);

% Initialize Cohort
Dist_curr = reshape(jequaloneDist, [N_a1 * N_a2 * N_semiz_safe * N_z_safe, 1]);

% Explicitly construct full-size state coordinate vectors (Length = 511,875)
[~, A2_idx_grid, Semiz_idx_grid, Z_idx_grid] = ndgrid(1:N_a1, 1:N_a2, 1:N_semiz_safe, 1:N_z_safe);

A2_grid_idx    = A2_idx_grid(:);
Semiz_grid_idx = Semiz_idx_grid(:);
Z_grid_idx     = Z_idx_grid(:);

% Policy comes in with un-flattened state grids (10 dimensions).
% We reshape it down to a clean 6D tensor matching our flattened state counts:
% [NumPolicies, N_a1, N_a2, N_semiz_safe, N_z_safe, N_j]
NumPolicies = size(Policy, 1);
Policy_reshaped = reshape(Policy, [NumPolicies, N_a1, N_a2, N_semiz_safe, N_z_safe, N_j]);

% =========================================================
% TIME LOOP (FORWARD SIMULATION)
% =========================================================
for jj = 1:N_j
    % 1. Store the current cohort distribution
    StationaryDist(:, :, :, :, jj) = reshape(Dist_curr, [N_a1, N_a2, N_semiz_safe, N_z_safe]);

    if jj == N_j
        break; % Terminal period, no forward transition needed.
    end

    % 2. Extract Exact Policy Indexes for Current Age
    % From ElectrifyHousingsemizV_ReturnFn, the first 4 inputs are:
    % 1. installpv (d2 -> drives Experience Asset)
    % 2. buyhouse  (d3 -> drives Semi-Exogenous states)
    % 3. aprime    (Endogenous asset 1)
    % 4. hprime    (Endogenous asset 2)
    
    % Layer 1: d2 (installpv)
    d2_layer = reshape(Policy_reshaped(1, :, :, :, :, jj), [N_a1, N_a2, N_semiz_safe, N_z_safe]);
    d2_linear_idx = d2_layer(:);
    
    % Layer 2: d3 (buyhouse)
    d3_layer = reshape(Policy_reshaped(2, :, :, :, :, jj), [N_a1, N_a2, N_semiz_safe, N_z_safe]);
    d3_linear_idx = d3_layer(:);
    
    % Layer 3 & 4: aprime and hprime
    aprime_idx = reshape(Policy_reshaped(3, :, :, :, :, jj), [N_a1, N_a2, N_semiz_safe, N_z_safe]);
    hprime_idx = reshape(Policy_reshaped(4, :, :, :, :, jj), [N_a1, N_a2, N_semiz_safe, N_z_safe]);
    
    % Combine aprime and hprime into the flattened a1 linear index
    % n_a1(1) is the size of the 'a' grid (13). This maps them into the 39-element N_a1 space.
    a1_linear_idx = aprime_idx(:) + n_a1(1) * (hprime_idx(:) - 1);
    
    % Clamp indices to safe bounds
    d2_linear_idx = max(1, min(d2_linear_idx, N_d2));
    d3_linear_idx = max(1, min(d3_linear_idx, N_d3));
    a1_linear_idx = max(1, min(a1_linear_idx, N_a1));

    % 3. Calculate Experience Asset Transition (a2)
    aprimeFnParamsCell = CreateCellFromParams(Parameters, aprimeFnParamNames, jj);

    % Build full 3D mesh for d2, a2, and semiz
    % This guarantees the output is strictly [N_d2, N_a2, N_semiz_safe]
    [d2_mesh, a2_mesh, semiz_mesh_1] = ndgrid(d2_gridvals(:), a2_grid(:), simoptions.semiz_gridvals_J(:,1,jj));
    [~,       ~,       semiz_mesh_2] = ndgrid(d2_gridvals(:), a2_grid(:), simoptions.semiz_gridvals_J(:,2,jj));
    [~,       ~,       semiz_mesh_3] = ndgrid(d2_gridvals(:), a2_grid(:), simoptions.semiz_gridvals_J(:,3,jj));
    [~,       ~,       semiz_mesh_4] = ndgrid(d2_gridvals(:), a2_grid(:), simoptions.semiz_gridvals_J(:,4,jj));

    % Evaluate aprimeFn directly across the completely explicit 3D space
    a2_prime_vals = aprimeFn(d2_mesh, a2_mesh, semiz_mesh_1, semiz_mesh_2, semiz_mesh_3, semiz_mesh_4, aprimeFnParamsCell{:});

    % Failsafe: Force array to expand to the full 3D size in case aprimeFn dropped dimensions
    if numel(a2_prime_vals) < N_d2 * N_a2 * N_semiz_safe
        a2_prime_vals = a2_prime_vals + zeros(N_d2, N_a2, N_semiz_safe);
    end

    % --- NEW LINE: Clamp floating-point overshoots strictly to the grid boundaries ---
    a2_prime_vals = max(a2_grid(1), min(a2_grid(end), a2_prime_vals));

    % Flat evaluate histc using (:) so it returns a flat vector of exactly N_d2*N_a2*N_semiz_safe elements
    [~, a2primeIndex] = histc(a2_prime_vals(:), a2_grid);
    a2primeIndex = max(1, min(a2primeIndex, N_a2 - 1));

    a2_step = a2_grid(a2primeIndex + 1) - a2_grid(a2primeIndex);
    a2_step(a2_step == 0) = 1; % Prevent division by zero

    % Ensure a2primeProbs is also explicitly flattened
    a2primeProbs = (a2_grid(a2primeIndex + 1) - a2_prime_vals(:)) ./ a2_step;
    a2primeProbs = max(0, min(1, a2primeProbs));

    % Map to the full flattened state space
    lookup_idx = d2_linear_idx(:) + N_d2 * (A2_grid_idx - 1) + N_d2 * N_a2 * (Semiz_grid_idx - 1);
    
    a2_p_lower = a2primeIndex(lookup_idx);
    a2_p_upper = min(a2_p_lower + 1, N_a2);
    a2_prob_lower = a2primeProbs(lookup_idx);

    % 4. Build Target Linear Indices (for AccumArray)
    % We push the mass into an intermediate tensor: [a1', a2', semiz, z, d3]
    target_idx_lower = a1_linear_idx(:) + N_a1 * (a2_p_lower(:) - 1) ...
        + N_a1*N_a2 * (Semiz_grid_idx(:) - 1) ...
        + N_a1*N_a2*N_semiz_safe * (Z_grid_idx(:) - 1) ...
        + N_a1*N_a2*N_semiz_safe*N_z_safe * (d3_linear_idx(:) - 1);

    target_idx_upper = a1_linear_idx(:) + N_a1 * (a2_p_upper(:) - 1) ...
        + N_a1*N_a2 * (Semiz_grid_idx(:) - 1) ...
        + N_a1*N_a2*N_semiz_safe * (Z_grid_idx(:) - 1) ...
        + N_a1*N_a2*N_semiz_safe*N_z_safe * (d3_linear_idx(:) - 1);

    % 5. Map the Mass forward using `accumarray`
    mass_lower = Dist_curr(:) .* a2_prob_lower(:);
    mass_upper = Dist_curr(:) .* (1 - a2_prob_lower(:));

    sz_mid = N_a1 * N_a2 * N_semiz_safe * N_z_safe * N_d3;
    % Convert to double for safe GPU accumulation, then cast back
    Dist_mid_flat = accumarray([target_idx_lower; target_idx_upper], double([mass_lower; mass_upper]), [sz_mid, 1]);
    Dist_mid = reshape(cast(Dist_mid_flat, 'like', Dist_curr), [N_a1 * N_a2, N_semiz_safe, N_z_safe, N_d3]);

    % 6. Apply Semi-Exogenous Markov Shocks (semiz)
    Dist_next_semiz = zeros(N_a1 * N_a2, N_semiz_safe, N_z_safe, 'like', Dist_curr);
    if N_semiz_safe > 1
        for i_d3 = 1:N_d3
            Dist_d3 = Dist_mid(:, :, :, i_d3); % [N_a_flat, N_semiz, N_z]
            pi_sz = pi_semiz_J(:,:, i_d3, min(jj, size(pi_semiz_J,4))); % [N_semiz, N_semiz_next]

            % Vectorized matrix mult: Dist(z, semiz) * pi(semiz, next_semiz)
            Dist_d3_perm = permute(Dist_d3, [1, 3, 2]); % [N_a_flat, N_z, N_semiz]
            Dist_trans = reshape(Dist_d3_perm, [N_a1*N_a2*N_z_safe, N_semiz_safe]) * pi_sz;
            Dist_trans = permute(reshape(Dist_trans, [N_a1*N_a2, N_z_safe, N_semiz_safe]), [1, 3, 2]);

            Dist_next_semiz = Dist_next_semiz + Dist_trans;
        end
    else
        Dist_next_semiz = sum(Dist_mid, 4);
    end

    % 7. Apply Purely Exogenous Markov Shocks (z)
    if N_z_safe > 1
        pi_z = pi_z_J(:,:, min(jj, size(pi_z_J,3))); % [N_z, N_z_next]
        % Dist_next_semiz is [N_a_flat * N_semiz, N_z]
        Dist_next_z = reshape(Dist_next_semiz, [N_a1*N_a2*N_semiz_safe, N_z_safe]) * pi_z;
        Dist_curr = reshape(Dist_next_z, [N_a1 * N_a2 * N_semiz_safe * N_z_safe, 1]);
    else
        Dist_curr = reshape(Dist_next_semiz, [N_a1 * N_a2 * N_semiz_safe * N_z_safe, 1]);
    end
end

% =========================================================
% OUTPUT UNPACKING
% =========================================================
if ~isfield(simoptions, 'outputkron') || simoptions.outputkron == 0
    if N_z_safe > 1
        StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, n_z, N_j]);
    else
        StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, N_j]);
    end
else
    StationaryDist = reshape(StationaryDist, [N_a1 * N_a2, N_semiz_safe * N_z_safe, N_j]);
end


end