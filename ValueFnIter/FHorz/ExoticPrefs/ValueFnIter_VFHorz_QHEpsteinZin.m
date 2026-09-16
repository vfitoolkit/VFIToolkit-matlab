function [V1, Policy, Valt, Policyalt] = ValueFnIter_VFHorz_QHEpsteinZin(n_d, n_a, n_z, N_j, d_gridvals, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions)

% 1. Extract Settings & Guardrails
if isfield(vfoptions, 'divideandconquer') && vfoptions.divideandconquer == 1
    error('V Universe Abort: Divide-and-Conquer assumes policy monotonicity. Quasi-Hyperbolic present-bias frequently causes non-monotonic savings behavior. Please set vfoptions.divideandconquer = 0.');
end

beta0 = Parameters.(vfoptions.QHadditionaldiscount);
isNaive = strcmp(vfoptions.quasi_hyperbolic, 'Naive');

% (For QHEpsteinZin.m, keep your ezc parameter extractions here)

% 2. Pre-allocate Output Tensors (using native V Universe dimensions)
N_choices = prod(n_d) * prod(n_a);
state_dims = [n_a, n_z];
if isscalar(state_dims); state_dims = [state_dims, 1]; end

V1 = zeros([state_dims, N_j], 'gpuArray');
Valt = zeros([state_dims, N_j], 'gpuArray');

% GI requires a 3-tier Policy tensor: [Coarse Idx, Subgrid Step, Edge Flag]
has_GI = isfield(vfoptions, 'gridinterplayer') && vfoptions.gridinterplayer == 1;
if has_GI
    Policy = zeros([3, state_dims, N_j], 'gpuArray');
    if isNaive; Policyalt = zeros([3, state_dims, N_j], 'gpuArray'); else; Policyalt = []; end
else
    Policy = zeros([state_dims, N_j], 'gpuArray');
    if isNaive; Policyalt = zeros([state_dims, N_j], 'gpuArray'); else; Policyalt = []; end
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
        Valt(:,:,jj) = V1_jj; % Valt = Vtilde or Vhat in terminal

        if has_GI
            Policy(1,:,:,jj) = Pol_jj;
            Policy(2,:,:,jj) = 0; % Dummy subgrid step (snapped to coarse)
            Policy(3,:,:,jj) = 2; % Interior flag
            if isNaive
                Policyalt(1,:,:,jj) = Pol_jj;
                Policyalt(2,:,:,jj) = 0;
                Policyalt(3,:,:,jj) = 2;
            end
        else
            Policy(:,:,jj) = Pol_jj;
            if isNaive
                Policyalt(:,:,jj) = Pol_jj;
            end
        end
        continue;
    end

    % --- CONTINUATION PERIODS ---
    if jj == N_j
        EV_Source = reshape(vfoptions.V_Jplus1, state_dims);
    else
        EV_Source = Valt(:,:,jj+1);
    end

    % --- A. EZ Forward Transformation (Risk Aversion) ---
    valid_V = isfinite(EV_Source) & (EV_Source ~= 0);
    V_transformed = EV_Source;
    if ezc5(jj) == 1
        V_transformed(valid_V) = ezc4 * EV_Source(valid_V);
    else
        V_transformed(valid_V) = max(ezc4 * EV_Source(valid_V), 0).^ezc5(jj);
    end
    V_transformed(EV_Source == 0) = 0;

    % --- B. Expected Value Transition ---
    pi_z_j = pi_z_J(:, :, min(jj, size(pi_z_J, 3)));
    EV_Expected = V_transformed * pi_z_j';

    % --- C. EZ Reverse Transformation (Certainty Equivalent) ---
    valid_EV = isfinite(EV_Expected) & (EV_Expected ~= 0);
    if ezc6(jj) ~= 1
        EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc6(jj);
    end
    if ezc8(jj) ~= 1
        EV_Expected(valid_EV) = max(EV_Expected(valid_EV), 0).^ezc8(jj);
    end

    EV_Flat = reshape(EV_Expected, [1, N_states]);

    % --- D. QUASI-HYPERBOLIC DUAL TRACKING ---
    if isNaive
        % 1. Exponential EZ Discounter (The imagined future self)
        RHS_alt = Evaluate_Universal_RHS_VFHorz(ReturnMatrix_Flat, EV_Flat, beta, 1, ezc2(jj), ezc3, ezc4, ezc7(jj));
        [Vtemp_alt, maxindex_alt] = max(RHS_alt, [], 1);
        Valt(:,:,jj) = reshape(Vtemp_alt, state_dims);

        % 2. Naive QH-EZ Discounter (The current self)
        RHS_tilde = Evaluate_Universal_RHS_VFHorz(ReturnMatrix_Flat, EV_Flat, beta0beta, 1, ezc2(jj), ezc3, ezc4, ezc7(jj));
        [Vtemp_tilde, maxindex_tilde] = max(RHS_tilde, [], 1);
        V1(:,:,jj) = reshape(Vtemp_tilde, state_dims);

        % Policy GI Routing
        if has_GI
            Policy(1,:,:,jj) = reshape(maxindex_tilde, state_dims);
            Policy(2,:,:,jj) = 0;
            Policy(3,:,:,jj) = 2;
            Policyalt(1,:,:,jj) = reshape(maxindex_alt, state_dims);
            Policyalt(2,:,:,jj) = 0;
            Policyalt(3,:,:,jj) = 2;
        else
            Policy(:,:,jj) = reshape(maxindex_tilde, state_dims);
            Policyalt(:,:,jj) = reshape(maxindex_alt, state_dims);
        end

    else % Sophisticated
        % 1. Sophisticated QH-EZ Discounter (The current self's choice)
        RHS_hat = Evaluate_Universal_RHS_VFHorz(ReturnMatrix_Flat, EV_Flat, beta0beta, 1, ezc2(jj), ezc3, ezc4, ezc7(jj));
        [Vtemp_hat, maxindex_hat] = max(RHS_hat, [], 1);
        V1(:,:,jj) = reshape(Vtemp_hat, state_dims);

        % 2. The Realized Continuation Value (Standard EZ Aggregation)
        RHS_underbar = Evaluate_Universal_RHS_VFHorz(ReturnMatrix_Flat, EV_Flat, beta, 1, ezc2(jj), ezc3, ezc4, ezc7(jj));

        % Extract the realized continuation using the current self's policy
        maxindexfull = maxindex_hat(:)' + N_choices * (0 : N_states - 1);
        Vunderbar_Flat = RHS_underbar(maxindexfull);
        Valt(:,:,jj) = reshape(Vunderbar_Flat, state_dims);

        % Policy GI Routing
        if has_GI
            Policy(1,:,:,jj) = reshape(maxindex_hat, state_dims);
            Policy(2,:,:,jj) = 0;
            Policy(3,:,:,jj) = 2;
        else
            Policy(:,:,jj) = reshape(maxindex_hat, state_dims);
        end
    end
end


end