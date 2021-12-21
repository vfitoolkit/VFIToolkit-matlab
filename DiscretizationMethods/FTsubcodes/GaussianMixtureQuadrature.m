%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GaussianMixtureQuadrature
% (c) 2019 Alexis Akira Toda
%
% Purpose:
%       compute the nodes and weights of Gaussian quadrature when the
%       weighting function is a Gaussian mixture
% Usage:
%       [x,w] = GaussianMixtureQuadrature(Coeff, Mu, Sigma, N)
%
% Inputs:
% Coeff     - coefficients (proportions) of Gaussian mixture components
% Mu        - vector of means
% Sigma     - vector of standard deviations
% N         - number of nodes for Gaussian quadrature
%
% Outputs:
% x         - nodes of Gaussian quadrature
% w         - weights of Gaussian quadrature
%
% Version 1.2: May 24, 2019
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

function [x,w] = GaussianMixtureQuadrature(Coeff, Mu, Sigma, N)

%% Some error checking
if length(Coeff) ~= length(Mu) || length(Coeff) ~= length(Sigma) || length(Mu) ~= length(Sigma)
    error('Coeff, Mu, Sigma must be of same length');
end

if size(Coeff,1) > size(Coeff,2)
    Coeff = Coeff'; % convert to row vector
end
if size(Mu,1) > size(Mu,2)
    Mu = Mu'; % convert to row vector
end
if size(Sigma,1) > size(Sigma,2)
    Sigma = Sigma'; % convert to row vector
end

%% Precompute polynomial moments of Gaussian mixture
K = length(Coeff); % number of mixture components
temp = zeros(2*N+1,K); % matrix that stores moments of each mixture component
Sigma2 = Sigma.^2; % vector of variances
temp(1,:) = 1;
temp(2,:) = Mu;
for n=2:2*N % n is the order of moments
    temp(n+1,:) = Mu.*temp(n,:) + (n-1)*Sigma2.*temp(n-1,:);
end
PolyMoments = temp*(Coeff'); % column vector of polynomial moments

%% Implement Golub-Welsch algorithm
M = zeros(N+1); % matrix of moments
for n=1:N+1
    M(n,:) = PolyMoments(n:N+n);
end
R = chol(M); % Cholesky factorization
temp0 = diag(R);
temp0(end) = [];
beta = temp0(2:N)./temp0(1:N-1);
temp1 = diag(R,1);
temp2 = temp1./temp0;
alpha = temp2 - [0;temp2(1:N-1)];

T = diag(alpha) + diag(beta,-1) + diag(beta,1);
[V,D] = eig(T);

%% Compute nodes and weights of Gaussian quadrature
x = diag(D)';
[x,ind] = sort(x);

w = zeros(1,N);
for n = 1 : N
    v = V(:,n);
    w(n) = sum(Coeff) * v(1)^2 / dot(v,v);
end
w = w(ind);

end
