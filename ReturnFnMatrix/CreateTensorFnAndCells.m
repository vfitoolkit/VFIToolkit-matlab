function [TensorFn, D_cells, A_cells, Z_cells, E_cells] = CreateTensorFnAndCells(InputFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_gridvals, e_grid)
% THE UNIVERSAL PACKER (tensor-bridge)
% Dynamically unpacks stacked 1D grids into isolated ND-cells for the Fused CUDA Kernel

% 1. Pack Decision States (D)
% Shifted to dim 4 to match the Slicer tensor shape [1, 1, 1, N_d]
D_cells = PackGrid(n_d, d_grid, 3);

% 2. Pack Endogenous States (A)
A_cells = PackGrid(n_a, a_grid, 0);

% 3. Pack Semi-Exogenous States (E)
E_cells = PackGrid(n_e, e_grid, 0);

% 4. Pack Exogenous States (Z)
% Z grids come pre-meshed as [N_z, num_vars] from ExogShockSetup, so we just cell-split them
if ~isempty(z_gridvals) && size(z_gridvals, 1) > 0
    num_z = size(z_gridvals, 2);
    Z_cells = cell(1, num_z);
    for i = 1:num_z
        Z_cells{i} = z_gridvals(:, i);
    end
else
    Z_cells = {};
end
end

function cells_out = PackGrid(n_dims, grid_in, shift_amount)
% Helper to extract 1D grids, run ndgrid, and optionally shiftdim
num_vars = length(n_dims);

if isempty(grid_in) || num_vars == 0 || (num_vars == 1 && n_dims(1) == 0)
    cells_out = {};
    return;
end

if num_vars > 1
    grids_1d = cell(1, num_vars);
    offset = 0;
    for i = 1:num_vars
        grids_1d{i} = grid_in((offset + 1):(offset + n_dims(i)));
        offset = offset + n_dims(i);
    end
    [mesh_out{1:num_vars}] = ndgrid(grids_1d{:});

    cells_out = cell(1, num_vars);
    for i = 1:num_vars
        if shift_amount > 0
            cells_out{i} = shiftdim(mesh_out{i}(:), -shift_amount);
        else
            cells_out{i} = mesh_out{i}(:);
        end
    end
else
    if shift_amount > 0
        cells_out = { shiftdim(grid_in(:), -shift_amount) };
    else
        cells_out = { grid_in(:) };
    end
end


end