function RHS = Evaluate_Universal_RHS_VFHorz(F_tensor, EV_bounded, beta_j, ezc1_j, ezc2_j, ezc3, ezc4, ezc7_j)
% EVALUATE_UNIVERSAL_RHS_VFHORZ: Universally assembles the RHS of the Bellman equation
% seamlessly handling both standard CRRA and Epstein-Zin non-linear curvature.

% --- 1. Apply ezc2 & ezc4 to Return Function ---
temp2 = F_tensor;

if ezc2_j == 1 && ezc4 == 1
    % Relax
else
    if ezc2_j == 1
        temp2 = ezc4 * F_tensor;
    else
        valid_F = isfinite(F_tensor) & (F_tensor ~= 0);
        temp2(valid_F) = max(ezc4 * F_tensor(valid_F), 0).^ezc2_j;
    end
end

% --- 2. Assemble Coarse RHS ---
if ezc1_j ~= 1 || beta_j ~= 0
    entireRHS = ezc1_j .* temp2 + beta_j .* EV_bounded;
else
    entireRHS = temp2;
end

% --- 3. Apply ezc3 & ezc7 to the combined RHS ---
RHS = entireRHS;

if ezc7_j == 1
    if ezc3 ~= 1
        RHS = ezc3 * entireRHS;
    end
else
    valid_RHS = isfinite(entireRHS) & (entireRHS ~= 0);
    RHS(valid_RHS) = ezc3 * (entireRHS(valid_RHS).^ezc7_j);
end


end