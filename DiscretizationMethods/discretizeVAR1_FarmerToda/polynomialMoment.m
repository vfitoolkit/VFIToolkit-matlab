function T = polynomialMoment(X,mu,scalingFactor,nMoments)

% Purpose:
%       Compute the moment defining function used in discreteApproximation
%
% Usage:
%       T = FT_polynomialMoment(X,mu,scalingFactor,nMoment)
%
% Inputs:
% X     - (1 x N) vector of grid points
% mu    - location parameter (conditional mean)
% scalingFactor    - scaling factor for numerical stability (typically largest grid point)
% nMoments   - number of polynomial moments

% Check that scaling factor is positive
if isnan(scalingFactor) || (scalingFactor <= 0)
    error('sF must be a positive number')
end

Y = (X-mu)/scalingFactor; % standardized grid
T = bsxfun(@power,Y,[1:nMoments]');

end

