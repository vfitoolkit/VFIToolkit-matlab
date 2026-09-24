function RHS = Compute_EZ_RHS(F, EV_CE, ezc1, ezc2, ezc3, ezc7, beta_j)
% 1. F^ezc2
F_term = zeros(size(F), 'like', F);
valid_F = isfinite(F) & (F ~= 0);
F_term(valid_F) = F(valid_F).^ezc2;
F_term(F == 0) = -Inf;

% 2. Combine
RHS = ezc1 .* F_term + ezc3 .* beta_j .* EV_CE;

% 3. RHS^ezc7
valid_RHS = isfinite(RHS) & (RHS ~= 0);
RHS(valid_RHS) = RHS(valid_RHS).^ezc7;
RHS(RHS == 0) = -Inf;


end