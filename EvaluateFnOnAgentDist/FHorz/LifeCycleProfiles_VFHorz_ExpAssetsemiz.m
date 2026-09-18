function AgeConditionalStats = LifeCycleProfiles_VFHorz_ExpAssetsemiz(StationaryDist, Policy, FnsToEvaluate, Parameters, ~, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, simoptions)

% --- 1. Unpack Semi-Exogenous Definitions ---
n_semiz = simoptions.n_semiz;
semiz_grid = simoptions.semiz_grid;

% --- 2. Extract Grids ---
installpv_grid = d_grid(1:n_d(1));
buyhouse_grid  = d_grid(n_d(1)+1 : sum(n_d(1:2)));

asset_grid   = a_grid(1:n_a(1));
house_grid   = a_grid(n_a(1)+1 : sum(n_a(1:2)));
solarpv_grid = a_grid(sum(n_a(1:2))+1 : end);

pbefore_grid = semiz_grid(1:n_semiz(1));
pafter_grid  = semiz_grid(n_semiz(1)+1 : sum(n_semiz(1:2)));
years_grid   = semiz_grid(sum(n_semiz(1:2))+1 : sum(n_semiz(1:3)));
down_grid    = semiz_grid(sum(n_semiz(1:3))+1 : end);

% --- 3. Define the Dynamic State Tensor Shape ---
% Handle multi-dimensional n_z safely
n_z_dims = n_z(:)';
dim_shape = [n_a(1), n_a(2), n_a(3), n_semiz(1), n_semiz(2), n_semiz(3), n_semiz(4), n_z_dims];

% --- 4. Create the Universal State Mesh ---
% Construct grid inputs dynamically to support arbitrary z dimensions
grid_inputs = [{asset_grid, house_grid, solarpv_grid, ...
    pbefore_grid, pafter_grid, years_grid, down_grid}, ...
    cellfun(@(v) reshape(v, [], 1), {z_grid}, 'UniformOutput', false)];

% If z_grid is multi-column, we need to extract individual z vectors for ndgrid
if size(z_grid, 2) > 1
    z_cell_args = cell(1, size(z_grid, 2));
    for iz = 1:size(z_grid, 2)
        % Reconstruct individual AR(1) grids for the mesh
        unique_z_vals = unique(z_grid(:, iz));
        z_cell_args{iz} = unique_z_vals(:);
    end
    [A_mesh, H_mesh, Solar_mesh, PB_mesh, PA_mesh, Y_mesh, D_mesh, Z_meshes{1:length(z_cell_args)}] = ndgrid(...
        asset_grid, house_grid, solarpv_grid, ...
        pbefore_grid, pafter_grid, years_grid, down_grid, z_cell_args{:});
    % Combine multi-z into the evaluation function or pass individually
    Z_mesh = Z_meshes{1}; % Primary reference
else
    [A_mesh, H_mesh, Solar_mesh, PB_mesh, PA_mesh, Y_mesh, D_mesh, Z_mesh] = ndgrid(...
        asset_grid, house_grid, solarpv_grid, ...
        pbefore_grid, pafter_grid, years_grid, down_grid, z_grid(:,1));
end

% --- 5. Prepare Output Structure ---
fn_names = fieldnames(FnsToEvaluate);
for i = 1:length(fn_names)
    AgeConditionalStats.(fn_names{i}).Mean = zeros(N_j, 1, 'like', asset_grid);
    AgeConditionalStats.(fn_names{i}).Var  = zeros(N_j, 1, 'like', asset_grid);
end

NumPolicies = size(Policy, 1);
Policy_reshaped = reshape(Policy, [NumPolicies, dim_shape, N_j]);
Dist_reshaped   = reshape(StationaryDist, [dim_shape, N_j]);
decision_idx = repmat({':'}, 1, ndims(Policy_reshaped)-2);

% --- 6. The Cross-Sectional Age Loop ---
for jj = 1:N_j
    Dist_j = Dist_reshaped(:,:,:,:,:,:,:,:, jj);
    mass_j = sum(Dist_j, 'all');

    if mass_j > 0
        Dist_pdf = Dist_j ./ mass_j; % Condition on survival
    else
        continue;
    end

    % Extract decisions for this age (Using trailing expansion for multi-D safety)
    pol_install  = reshape(Policy_reshaped(1, decision_idx{:}, jj), dim_shape);
    pol_buyhouse = reshape(Policy_reshaped(2, decision_idx{:}, jj), dim_shape);
    pol_aprime   = reshape(Policy_reshaped(3, decision_idx{:}, jj), dim_shape);
    pol_hprime   = reshape(Policy_reshaped(4, decision_idx{:}, jj), dim_shape);

    % Map indexes to actual choice values
    val_install  = installpv_grid(pol_install);
    val_buyhouse = buyhouse_grid(pol_buyhouse);
    val_aprime   = asset_grid(pol_aprime);
    val_hprime   = house_grid(pol_hprime);

    % Evaluate all Anonymous Functions across all states simultaneously
    for i = 1:length(fn_names)
        fn = FnsToEvaluate.(fn_names{i});

        try
            % Attempt to pass age-dependent and exogenous params (w, kappa_j)
            kappa = Parameters.kappa_j(jj);
            w = Parameters.w;
            val_tensor = fn(val_install, val_buyhouse, val_aprime, val_hprime, ...
                A_mesh, H_mesh, Solar_mesh, PB_mesh, PA_mesh, Y_mesh, D_mesh, Z_mesh, w, kappa);
        catch
            % Fallback for simpler functions (e.g., pure assets)
            val_tensor = fn(val_install, val_buyhouse, val_aprime, val_hprime, ...
                A_mesh, H_mesh, Solar_mesh, PB_mesh, PA_mesh, Y_mesh, D_mesh, Z_mesh);
        end

        % Ensure val_tensor is fully expanded in case a function returned a scalar constraint
        val_tensor = val_tensor + zeros(dim_shape, 'like', val_tensor);

        % Calculate Moments
        mean_val = sum(val_tensor .* Dist_pdf, 'all');
        AgeConditionalStats.(fn_names{i}).Mean(jj) = gather(mean_val);

        var_val = sum(((val_tensor - mean_val).^2) .* Dist_pdf, 'all');
        AgeConditionalStats.(fn_names{i}).Var(jj)  = gather(var_val);
    end
end


end