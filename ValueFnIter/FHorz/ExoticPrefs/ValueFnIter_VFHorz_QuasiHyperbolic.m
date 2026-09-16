function [V1, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QuasiHyperbolic(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Extract Settings
beta0 = Parameters.(vfoptions.QHadditionaldiscount);
isNaive = strcmp(vfoptions.quasi_hyperbolic, 'Naive');

% 2. Pre-allocate Output Tensors (using native V Universe dimensions)
N_choices = prod(n_d) * prod(n_a);
state_dims = [n_a, n_z]; % V Universe native state shape
if isscalar(state_dims); state_dims = [state_dims, 1]; end

V1 = zeros([state_dims, N_j], 'gpuArray');
Policy = zeros([state_dims, N_j], 'gpuArray');
Valt = zeros([state_dims, N_j], 'gpuArray');

if isNaive
    Policyalt = zeros([state_dims, N_j], 'gpuArray');
else
    Policyalt = [];
end

% 3. Backward Induction Loop
for reverse_j = 1:N_j
    jj = N_j - reverse_j + 1;

    % Generate Parameters as Cell Arrays (V Universe Implicit Expansion)
    ReturnFnParamCell = CreateCellFromParams(Parameters, ReturnFnParamNames, jj);
    DiscountFactorParamsCell = CreateCellFromParams(Parameters, DiscountFactorParamNames, jj);
    beta = prod(cell2mat(DiscountFactorParamsCell));
    beta0beta = beta0 * beta;

    % Expand ReturnMatrix (Flattened to: [Choices x States])
    ReturnMatrix = CreateReturnFnMatrix_VFHorz(ReturnFn, n_d, n_a, n_z, d_gridvals, a_grid, z_gridvals_J(:,:,jj), ReturnFnParamCell);
    N_states = numel(ReturnMatrix) / N_choices;
    ReturnMatrix_Flat = reshape(ReturnMatrix, [N_choices, N_states]);

    % --- TERMINAL PERIOD ---
    if jj == N_j && ~isfield(vfoptions, 'V_Jplus1')
        [Vtemp, maxindex] = max(ReturnMatrix_Flat, [], 1);
        V1_jj = reshape(Vtemp, state_dims);
        Pol_jj = reshape(maxindex, state_dims);

        V1(:,:,jj) = V1_jj;
        Policy(:,:,jj) = Pol_jj;
        Valt(:,:,jj) = V1_jj; % Valt = Vtilde or Vhat in terminal
        if isNaive
            Policyalt(:,:,jj) = Pol_jj;
        end
        continue;
    end

    % --- CONTINUATION PERIODS ---
    if jj == N_j
        EV = reshape(vfoptions.V_Jplus1, state_dims); % V_Jplus1 is Valt for Naive, Vunderbar for Soph.
    else
        EV = Valt(:,:,jj+1);
    end

    % Vectorized Expected Value Calculation
    % (Assuming pi_z_J transition matrix multiplication is handled here via tensor contraction or standard V Universe EV prep)
    EV_expanded = CalculateEV_VFHorz(EV, pi_z_J(:,:,jj)); % <-- Replace with your exact V Universe EV expansion function
    EV_Flat = reshape(EV_expanded, [1, N_states]);

    if isNaive
        % 1. Exponential Discounter (The imagined future self)
        RHS_alt = ReturnMatrix_Flat + beta * EV_Flat;
        [Vtemp_alt, maxindex_alt] = max(RHS_alt, [], 1);
        Valt(:,:,jj) = reshape(Vtemp_alt, state_dims);
        Policyalt(:,:,jj) = reshape(maxindex_alt, state_dims);

        % 2. Naive QH Discounter (The current self)
        RHS_tilde = ReturnMatrix_Flat + beta0beta * EV_Flat;
        [Vtemp_tilde, maxindex_tilde] = max(RHS_tilde, [], 1);
        V1(:,:,jj) = reshape(Vtemp_tilde, state_dims);
        Policy(:,:,jj) = reshape(maxindex_tilde, state_dims);

    else % Sophisticated
        % 1. Sophisticated QH Discounter (The current self's choice)
        RHS_hat = ReturnMatrix_Flat + beta0beta * EV_Flat;
        [Vtemp_hat, maxindex_hat] = max(RHS_hat, [], 1);
        V1(:,:,jj) = reshape(Vtemp_hat, state_dims);
        Policy(:,:,jj) = reshape(maxindex_hat, state_dims);

        % 2. The Realized Continuation Value (Vunderbar)
        RHS_underbar = ReturnMatrix_Flat + beta * EV_Flat;

        % V Universe Vectorized Linear Indexing
        maxindexfull = maxindex_hat(:)' + N_choices * (0 : N_states - 1);
        Vunderbar_Flat = RHS_underbar(maxindexfull);
        Valt(:,:,jj) = reshape(Vunderbar_Flat, state_dims);
    end
end


end