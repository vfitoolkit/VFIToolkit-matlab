function CovarMatrix = corr2cov_homemade(StandardDeviationVec, CorrMatrix)
% Convert correlation matrix (plus vector of standard deviations) to covariance matrix
%
%   CovarMatrix = corr2cov(StandardDeviationVec, CorrMatrix)
%
% Inputs:
%   StandardDeviationVec : vector of standard deviations (n x 1 or 1 x n)
%   CorrMatrix   : correlation matrix (n x n)
%
% Output:
%   CovarMatrix    : covariance matrix (n x n)

    % Ensure StandardDeviationVec is a column vector
    StandardDeviationVec = StandardDeviationVec(:);

    % Basic dimension check
    n = length(StandardDeviationVec);
    if ~isequal(size(CorrMatrix), [n n])
        error('Dimensions of standard deviations and correlation matrix do not match.');
    end

    % Construct covariance matrix
    D = diag(StandardDeviationVec);
    CovarMatrix = D * CorrMatrix * D;
end
