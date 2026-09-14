function StationaryDist = StationaryDist_VFHorz_ExpAssetsemiz(jequaloneDist, AgeWeightParamNames, Policy, n_d, n_a, n_semiz, n_z, N_j, pi_semiz_J, pi_z_J, Parameters, simoptions)

% STATIONARYDIST_VFHORZ_EXPASSETSEMIZ
% V-World Universal Forward Simulator for Experience Asset + Semi-Exogenous OLG Models

% --- 1. Dimension Extraction ---
l_dsemiz = simoptions.l_dsemiz;
l_dexperienceasset = simoptions.l_dexperienceasset;

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
N_d2 = max(1, prod(n_d2)); N_d3 = max(1, prod(n_d3));

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

% Prepare explicit coordinate grids for linear index lookups
[~, A2_grid_idx, Semiz_grid_idx, Z_grid_idx] = ndgrid(1:N_a1, 1:N_a2, 1:N_semiz_safe, 1:N_z_safe);

% =========================================================
% TIME LOOP (FORWARD SIMULATION)
% =========================================================
for jj = 1:N_j
    % 1. Store the current cohort distribution
    StationaryDist(:, :, :, :, jj) = reshape(Dist_curr, [N_a1, N_a2, N_semiz_safe, N_z_safe]);

    if jj == N_j
        break; % Terminal period, no forward transition needed.
    end

    % 2. Extract Exact Sub-Policy Indexes for Current Age
    % Build linear indexing over n_a1 (endogenous states)
    a1_linear_idx = zeros(N_a1 * N_a2 * N_semiz_safe * N_z_safe, 1, 'like', Dist_curr);
    cum_n_a1 = [1, cumprod(n_a1)];
    for i = 1:l_a1
        Pol_a1_i = shiftdim(Policy(l_d+i, :, :, :, :, jj), 1);
        a1_linear_idx = a1_linear_idx + cum_n_a1(i) * (Pol_a1_i(:) - 1);
    end
    a1_linear_idx = a1_linear_idx + 1;

    % Build linear indexing over n_d3 (semi-exogenous decision)
    d3_linear_idx = zeros(N_a1 * N_a2 * N_semiz_safe * N_z_safe, 1, 'like', Dist_curr);
    cum_n_d3 = [1, cumprod(n_d3)];
    for i = 1:l_d3
        Pol_d3_i = shiftdim(Policy(l_d1+l_d2+i, :, :, :, :, jj), 1);
        d3_linear_idx = d3_linear_idx + cum_n_d3(i) * (Pol_d3_i(:) - 1);
    end
    d3_linear_idx = d3_linear_idx + 1;

    % Build linear indexing over n_d2 (experience asset decision)
    d2_linear_idx = zeros(N_a1 * N_a2 * N_semiz_safe * N_z_safe, 1, 'like', Dist_curr);
    cum_n_d2 = [1, cumprod(n_d2)];
    for i = 1:l_d2
        Pol_d2_i = shiftdim(Policy(l_d1+i, :, :, :, :, jj), 1);
        d2_linear_idx = d2_linear_idx + cum_n_d2(i) * (Pol_d2_i(:) - 1);
    end
    d2_linear_idx = d2_linear_idx + 1;

    % 3. Calculate Experience Asset Transition (a2)
    aprimeFnParamsVec = CreateVectorFromParams(Parameters, aprimeFnParamNames, jj);
    [a2primeIndex_matrix, a2primeProbs_matrix] = CreateExperienceAssetsemizFnMatrix(...
        aprimeFn, n_d2, n_a2, n_semiz, d2_gridvals, a2_grid, ...
        simoptions.semiz_grid(:,:,jj), aprimeFnParamsVec, 2);

    lookup_idx = d2_linear_idx(:) + N_d2 * (A2_grid_idx(:) - 1) + N_d2 * N_a2 * (Semiz_grid_idx(:) - 1);
    a2_p_lower = a2primeIndex_matrix(lookup_idx);
    a2_p_upper = min(a2_p_lower + 1, N_a2);
    a2_prob_lower = a2primeProbs_matrix(lookup_idx);

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
if simoptions.outputkron == 0
    if N_z_safe > 1
        StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, n_z, N_j]);
    else
        StationaryDist = reshape(StationaryDist, [n_a_out, n_semiz, N_j]);
    end
else
    StationaryDist = reshape(StationaryDist, [N_a1 * N_a2, N_semiz_safe * N_z_safe, N_j]);
end


end