%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% discreteApproximation
% (c) 2016 Leland E. Farmer and Alexis Akira Toda
% 
% Purpose: 
%       Compute a discrete state approximation to a distribution with known
%       moments, using the maximum entropy procedure proposed in Tanaka and
%       Toda (2013)
%
% Usage:
%       [p,lambdaBar,momentError] = discreteApproximation(D,T,TBar,q,lambda0)
%
% Inputs:
% D         - (K x N) matrix of grid points. K is the dimension of the
%             domain. N is the number of points at which an approximation
%             is to be constructed.
% T         - A function handle which should accept arguments of dimension
%             (K x N) and return an (L x N) matrix of moments evaluated at
%             each grid point, where L is the number of moments to be
%             matched.
% TBar      - (L x 1) vector of moments of the underlying distribution
%             which should be matched
% Optional:
% q         - (1 X N) vector of prior weights for each point in D. The
%             default is for each point to have an equal weight.
% lambda0   - (L x 1) vector of initial guesses for the dual problem
%             variables. The default is a vector of zeros.
%
% Outputs:
% p         - (1 x N) vector of probabilties assigned to each grid point in
%             D.
% lambdaBar - (L x 1) vector of dual problem variables which solve the
%             maximum entropy problem
% momentError - (L x 1) vector of errors in moments (defined by moments of
%               discretization minus actual moments)
%
% Version 1.2: June 7, 2016
%
% Version 1.3: May 26, 2019
%
% Changed algorithm to 'trust-region' to use Hessian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [p,lambdaBar,momentError] = discreteApproximation(D,T,TBar,q,lambda0)

% Input error checking

if nargin < 3
    error('You must provide at least 3 arguments to discreteApproximation.')
end

N = size(D,2);

Tx = T(D);
L = size(Tx,1);

if size(Tx,2) ~= N || length(TBar) ~= L
    error('Dimension mismatch')
end

% Default prior weights
if nargin == 3
    q = ones(1,N)./N;
end

% Compute maximum entropy discrete distribution

%options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','off','GradObj','on','Hessian','on');
options = optimset('TolFun',1e-10,'TolX',1e-10,'Display','off','Algorithm','trust-region');

% Sometimes the algorithm fails to converge if the initial guess is too far
% away from the truth. If this occurs, the program tries an initial guess
% of all zeros.
try
    lambdaBar = fminunc(@(lambda) entropyObjective(lambda,Tx,TBar,q),lambda0,options);
catch
    warning('Failed to find a solution from provided initial guess. Trying new initial guess.')
    lambdaBar = fminunc(@(lambda) entropyObjective(lambda,Tx,TBar,q),zeros(size(lambda0)),options);
end

% Compute final probability weights and moment errors
[obj,gradObj] = entropyObjective(lambdaBar,Tx,TBar,q);
Tdiff = Tx-repmat(TBar,1,N);
p = (q.*exp(lambdaBar'*Tdiff))./obj;
momentError = gradObj./obj;

end