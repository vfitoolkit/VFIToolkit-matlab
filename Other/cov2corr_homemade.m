function [StandardDeviationVec, CorrMatrix] = cov2corr_homemade(CovarMatrix)
% Convert covariance matrix to vector of standard deviations and correlation matrix
%
%   [StandardDeviationVec, CorrMatrix] = cov2corr(CovarMatrix)
%
% Input:
%   CovarMatrix   : covariance matrix (n x n)
%
% Outputs:
%   StandardDeviationVec : vector of standard deviations (n x 1)
%   CorrMatrix   : correlation matrix (n x n)

    % Basic dimension check
    [n, m] = size(CovarMatrix);
    if n ~= m
        error('Covariance matrix must be square.');
    end

    % Standard deviations from diagonal of covariance matrix (column vector)
    StandardDeviationVec = sqrt(diag(CovarMatrix));

    % Construct correlation matrix
    Dinv = diag(1 ./ StandardDeviationVec);
    CorrMatrix = Dinv * CovarMatrix * Dinv;
end
