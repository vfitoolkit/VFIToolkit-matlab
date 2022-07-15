function [z_grid,P] = discretizeAR1_TauchenHussey(mew,rho,sigma,znum,tauchenhusseyoptions)
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation 
%    of AR(1) process z'=mew+rho*z+e, e~N(0,sigma^2), by Tauchen-Hussey method
%
% Input:      
%   N         scalar, number of nodes for Z
%   mew       scalar, unconditional mean of process
%   rho       scalar
%   sigma     scalar, std. dev. of epsilons
% Optional inputs (tauchenhusseyoptions)
%   baseSigma scalar, std. dev. used to calculate Gaussian quadrature weights and nodes, 
%                 i.e. to build the grid. I recommend that you use baseSigma = w*sigma +
%                 (1-w)*sigmaZ where sigmaZ = sigma/sqrt(1-rho^2), and w = 0.5 + rho/4. 
%                 Tauchen & Hussey recommend baseSigma = sigma, and also mention baseSigma = sigmaZ.
%
% Output:     
%   z_grid    N*1 vector, nodes for Z
%   P         N*N matrix, transition probabilities
%
% Martin Floden, Stockholm School of Economics
% January 2007 (updated August 2007)
% This version was lightly modified by Robert Kirkby
%%%%%%%%
% Original paper:
% Tauchen and Hussey (1991) - Quadrature-Based Methods for Obtaining Approximate Solutions to Nonlinear Asset Pricing Models

%%
fprintf('COMMENT: The Tauchen-Hussey method is inferior to the Farmer-Toda method for discretizing AR(1) processes. \n') 
fprintf('         It is strongly recommended you use Farmer-Toda instead. \n')

%% Set default for baseSigma following Floden's suggestion (see above)
if ~exist('tauchenhusseyoptions','var')
    w=0.5+rho/4;
    sigmaZ=sigma/sqrt(1-rho^2);
    baseSigma=w*sigma+(1-w)*sigma*sigmaZ;
else
    if ~isfield(tauchenhusseyoptions,'baseSigma')
        w=0.5+rho/4;
        sigmaZ=sigma/sqrt(1-rho^2);
        baseSigma=w*sigma+(1-w)*sigma*sigmaZ;
    else
        baseSigma=tauchenhusseyoptions.baseSigma;
    end
end

% z_grid=zeros(znum,1);
P = zeros(znum,znum);

[z_grid,w] = gaussnorm(znum,mew,baseSigma^2);   % See note 1 below


for i = 1:znum
    for j = 1:znum
        EZprime    = (1-rho)*mew + rho*z_grid(i);
        P(i,j) = w(j) * norm_pdf(z_grid(j),EZprime,sigma^2) / norm_pdf(z_grid(j),mew,baseSigma^2);
    end
end

for i = 1:znum
    P(i,:) = P(i,:) / sum(P(i,:),2);
end

a = 1;

function c = norm_pdf(x,mu,s2)
    c = 1/sqrt(2*pi*s2) * exp(-(x-mu)^2/2/s2);

function [x,w] = gaussnorm(n,mu,s2)
% Find Gaussian nodes and weights for the normal distribution
% n  = # nodes
% mu = mean
% s2 = variance
[x0,w0] = gausshermite(n);
x = x0*sqrt(2*s2) + mu;
w = w0 / sqrt(pi);


function [x,w] = gausshermite(n)
% Gauss Hermite nodes and weights following "Numerical Recipes for C"

MAXIT = 10;
EPS   = 3e-14;
PIM4  = 0.7511255444649425;

x = zeros(n,1);
w = zeros(n,1);

m = floor(n+1)/2;
for i=1:m
    if i == 1
        z = sqrt((2*n+1)-1.85575*(2*n+1)^(-0.16667));
    elseif i == 2
        z = z - 1.14*(n^0.426)/z;
    elseif i == 3
        z = 1.86*z - 0.86*x(1);
    elseif i == 4
        z = 1.91*z - 0.91*x(2);
    else
        z = 2*z - x(i-2);
    end
    
    for iter = 1:MAXIT
        p1 = PIM4;
        p2 = 0;
        for j=1:n
            p3 = p2;
            p2 = p1;
            p1 = z*sqrt(2/j)*p2 - sqrt((j-1)/j)*p3;
        end
        pp = sqrt(2*n)*p2;
        z1 = z;
        z = z1 - p1/pp;
        if abs(z-z1) <= EPS
            break
        end
    end
    if iter>MAXIT
        error('too many iterations')
    end
    x(i)     = z;
    x(n+1-i) = -z;
    w(i)     = 2/pp/pp;
    w(n+1-i) = w(i);
end
x(:) = x(end:-1:1);


% Note 1: If you have Miranda and Fackler's CompEcon toolbox you can use
% their qnwnorm function to obtain quadrature nodes and weights for the
% normal function: [Z,w] = qnwnorm(N,mu,baseSigma^2);
% Compecon is available at http://www4.ncsu.edu/~pfackler/compecon/
% Otherwise, use gaussnorm as here.