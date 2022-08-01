function [Z_grid,P] = discretizeVAR1_FarmerToda(Mew,Rho,SigmaSq,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments
% 
% Purpose: 
%       Compute a finite-state Markov chain approximation to a VAR(1)
%       process of the form 
%
%           Z_(t+1) = Mew + Rho*Z_(t) + SigmaSq^(1/2)*epsilon_(t+1)
%
%       where epsilon_(t+1) is an (M x 1) vector of independent standard
%       normal innovations. Notice that SigmaSq is the variance-covariance matrix.
%
% Usage:
%       [P,Z_grid] = discretizeVAR1_FarmerToda(Mew,Rho,Sigma,znum,farmertodaoptions)
%
% Inputs:
%   Mew       - (M x 1) constant vector
%   Rho       - (M x M) matrix of impact coefficients
%   Sigma     - (M x M) variance-covariance matrix of the innovations
%   znum      - Desired number of discrete points in each dimension
%               (must be same for every dimension; actual grids are jointly determined as znum^M points per variable)
% Optional inputs (farmertodaoptions):
%   parallel: - set equal to 2 to use GPU, 0 to use CPU
%   nMoments  - Desired number of moments to match. The default is 2.
%   method    - String specifying the method used to determine the grid points. 
%               Accepted inputs are 'even,' 'quantile,' and 'quadrature.' The 
%               default option is 'even.' Please see the paper for more details.
%   nSigmas   - If method='even' option is specified, nSigmas is used to
%               determine the number of unconditional standard deviations
%               used to set the endpoints of the grid (mew+-nSigmas*standarddeviation)
%
% Outputs:
%   P         - (znum^M x znum^M) probability transition matrix. Each row
%               corresponds to a discrete conditional probability 
%               distribution over the state M-tuples in X
%   Z_grid    - (M x znum^M) matrix of states. Each column corresponds to an
%               M-tuple of values which correspond to the state associated 
%               with each row of P. (Puts znum^M points on each variable,
%               the grids for the variables are codetermined.)
%               Note: Z_grid are jointly determined.
%
% NOTES:
% - discretizeVAR1_FarmerToda only accepts non-singular variance-covariance matrices.
% - discretizeVAR1_FarmerToda only constructs tensor product grids where each dimension
%     contains the same number of points. For this reason it is recommended
%     that this code not be used for problems of more than about 4 or 5
%     dimensions due to curse of dimensionality issues.
%
% Future updates will allow for singular variance-covariance matrices and sparse grid specifications (This comment is by Farmer & Toda).
%
% (c) 2015 Leland E. Farmer and Alexis Akira Toda (v1.3, 2019)
% This version was lightly modified by Robert Kirkby
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
% Original paper:
% Farmer & Toda (2017) - Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments

%% Input error checks

warning off MATLAB:singularMatrix % surpress inversion warnings

%% Set defaults
if ~exist('farmertodaoptions','var')
    farmertodaoptions.nMoments = 2;
    if any(eig(Rho) > 0.8)
        farmertodaoptions.method='even';
    else
        farmertodaoptions.method='gauss-hermite';
    end
    farmertodaoptions.parallel=1+(gpuDeviceCount>0);
else
    if ~isfield(farmertodaoptions,'nMoments')
        farmertodaoptions.nMoments = 2; % Default number of moments to match is 2        
    end
    % define grid spacing parameter if not provided (only used for 'even' method)
    if ~isfield(farmertodaoptions,'nSigmas') % This is just direct from Farmer-Toda code. I am not aware of any results showing it performs 'better'
        if abs(eig(Rho)) <= 1-2/(znum-1)
            farmertodaoptions.nSigmas = sqrt(2*(znum-1)); % This was in Farmer-Toda AR(1) code, but not their VAR(1) code. I have put it here as well.
        else
            farmertodaoptions.nSigmas = sqrt(znum-1); 
        end
    end
    % Set method based on findings of paper of Farmer & Toda (2017): last para on pg 678
    %   method='even' for rho>0.8, 'gauss-hermite' for rho<=0.8
    if ~isfield(farmertodaoptions,'method')
        if any(eig(Rho) > 0.8)
            farmertodaoptions.method='even';
        else
            farmertodaoptions.method='gauss-hermite';
        end
    end
    if ~isfield(farmertodaoptions,'parallel')
        farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    end
end

% Make sure method is set appropraitely
if ~strcmp(farmertodaoptions.method,'quantile') && ~strcmp(farmertodaoptions.method,'even') && ~strcmp(farmertodaoptions.method,'gauss-hermite')
    error('Method must be one of quantile, even, or gauss-hermite')
end
if strcmp(farmertodaoptions.method,'quantile')
    warning('quantile method is poor and not recommended')
end
% Warning about persistence for quadrature method
if strcmp(farmertodaoptions.method,'gauss-hermite') && any(eig(Rho) > 0.8)
    warning('The gauss-hermite quadrature method may perform poorly for persistent processes.')
end


%% Check that inputs are correctly formatted
RhoSize = size(Rho);
M = RhoSize(1);

% Check size restrictions on matrices
if RhoSize(1) ~= RhoSize(2)
    error('Rho must be a square matrix')
end
if size(Mew,2) ~= 1
    error('Mew must be a column vector')
end
if size(Mew,1) ~= RhoSize(1)
    error('Mew must have the same number of rows as Rho')
end
% Check that Sigma is a valid covariance matrix
[~,posDefCheck] = chol(SigmaSq);
if posDefCheck
    error('Sigma must be a positive definite matrix')
end
% Check that znum is a valid number of grid points
if ~isnumeric(znum) || znum < 3 || rem(znum,1) ~= 0
    error('znum must be a positive integer greater than 3')
end
% Check that nMoments is a valid number
if ~isnumeric(farmertodaoptions.nMoments) || farmertodaoptions.nMoments < 1 || ~((rem(farmertodaoptions.nMoments,2) == 0) || (farmertodaoptions.nMoments == 1))
    error('farmertodaoptions.nMoments must be either 1 or a positive even integer')
end


%% Compute polynomial moments of standard normal distribution
gaussianMoment = zeros(farmertodaoptions.nMoments,1);
c = 1;
for k=1:floor(farmertodaoptions.nMoments/2)
    c = (2*k-1)*c;
    gaussianMoment(2*k) = c;
end

%% Compute standardized VAR(1) representation (zero mean and diagonal covariance matrix)

if M == 1
    
    C = sqrt(SigmaSq);
    A = Rho;
    mu = Mew/(1-Rho);
    SigmaSq = 1/(1-Rho^2);
    
else
    
    C1 = chol(SigmaSq,'lower');
    mu = ((eye(M)-Rho)\eye(M))*Mew;
    A1 = C1\(Rho*C1);
    Sigma1 = reshape(((eye(M^2)-kron(A1,A1))\eye(M^2))*reshape(eye(M),M^2,1),M,M); % unconditional variance
    U = minVarTrace(Sigma1);
    A = U'*A1*U;
    SigmaSq = U'*Sigma1*U;
    C = C1*U;
    
end

%% Construct 1-D grids

sigmas = sqrt(diag(SigmaSq));
y1D = zeros(M,znum);

switch farmertodaoptions.method
    case 'even'
        for ii = 1:M
            minSigmas = sqrt(min(eigs(SigmaSq)));
            y1D(ii,:) = linspace(-minSigmas*farmertodaoptions.nSigmas,minSigmas*farmertodaoptions.nSigmas,znum);
        end
    case 'quantile'
        y1DBounds = zeros(M,znum+1);
        for ii = 1:M
            y1D(ii,:) = norminv((2*(1:znum)-1)./(2*znum),0,sigmas(ii));
            y1DBounds(ii,:) = [-Inf, norminv((1:znum-1)./znum,0,sigmas(ii)), Inf];
        end
    case 'gauss-hermite'
        [nodes,weights] = GaussHermite(znum);
        for ii = 1:M
            y1D(ii,:) = sqrt(2)*nodes;
        end
end

% Construct all possible combinations of elements of the 1-D grids
D = allcomb2(y1D)';

%% Construct finite-state Markov chain approximation

condMean = A*D; % conditional mean of the VAR process at each grid point
P = ones(znum^M); % probability transition matrix
scalingFactor = y1D(:,end); % normalizing constant for maximum entropy computations
temp = zeros(M,znum); % used to store some intermediate calculations
lambdaBar = zeros(2*M,znum^M); % store optimized values of lambda (2 moments) to improve initial guesses
kappa = 1e-8; % small positive constant for numerical stability

for ii = 1:(znum^M)
   
    % Construct prior guesses for maximum entropy optimizations
    switch farmertodaoptions.method
        case 'even'
            q = normpdf(y1D,repmat(condMean(:,ii),1,znum),1);
        case 'quantile'
            q = normcdf(y1DBounds(:,2:end),repmat(condMean(:,ii),1,znum),1)...
                - normcdf(y1DBounds(:,1:end-1),repmat(condMean(:,ii),1,znum),1);
        case 'gauss-hermite'
            q = bsxfun(@times,(normpdf(y1D,repmat(condMean(:,ii),1,znum),1)./normpdf(y1D,0,1)),...
                (weights'./sqrt(pi)));
    end
    
    % Make sure all elements of the prior are stricly positive
    q(q<kappa) = kappa;
    
    for jj = 1:M
        
        % Try to use intelligent initial guesses
        if ii == 1
            lambdaGuess = zeros(2,1);
        else
            lambdaGuess = lambdaBar((jj-1)*2+1:jj*2,ii-1);
        end
        
        % Maximum entropy optimization
        if farmertodaoptions.nMoments == 1 % match only 1 moment
            temp(jj,:) = discreteApproximation(y1D(jj,:),...
                @(X)(X-condMean(jj,ii))/scalingFactor(jj),0,q(jj,:),0);
        else % match 2 moments first
            [p,lambda,momentError] = discreteApproximation(y1D(jj,:),...
                @(X) polynomialMoment(X,condMean(jj,ii),scalingFactor(jj),2),...
                [0; 1]./(scalingFactor(jj).^(1:2)'),q(jj,:),lambdaGuess);
            if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                warning('Failed to match first 2 moments. Just matching 1.')
                temp(jj,:) = discreteApproximation(y1D(jj,:),...
                    @(X)(X-condMean(jj,ii))/scalingFactor(jj),0,q(jj,:),0);
                lambdaBar((jj-1)*2+1:jj*2,ii) = zeros(2,1);
            elseif farmertodaoptions.nMoments == 2
                lambdaBar((jj-1)*2+1:jj*2,ii) = lambda;
                temp(jj,:) = p;
            else % solve maximum entropy problem sequentially from low order moments
                lambdaBar((jj-1)*2+1:jj*2,ii) = lambda;
                for mm = 4:2:farmertodaoptions.nMoments
                    lambdaGuess = [lambda;0;0]; % add zero to previous lambda
                    [pnew,lambda,momentError] = discreteApproximation(y1D(jj,:),...
                        @(X) polynomialMoment(X,condMean(jj,ii),scalingFactor(jj),mm),...
                        gaussianMoment(1:mm)./(scalingFactor(jj).^(1:mm)'),q(jj,:),lambdaGuess);
                    if norm(momentError) > 1e-5
                        warning('Failed to match first %d moments.  Just matching %d.',mm,mm-2)
                        break;
                    else
                        p = pnew;
                    end
                end
                temp(jj,:) = p;
            end
        end
    end
    
    P(ii,:) = prod(allcomb2(temp),2)';
    
end

Z_grid = C*D + repmat(mu,1,znum^M); % map grids back to original space
% Z_grid is M-by-(znum^M)
% It is NOT a kronecker-product grid.

warning on MATLAB:singularMatrix

% HAVE DONE THE LAZY OPTION. THIS SHOULD REALLY BE REWRITTEN SO THAT JUST
% CREATE ON GPU OR CPU AS APPROPRIATE. (AVOID THE OVERHEAD OF MOVING TO GPU)
if farmertodaoptions.parallel==2 
    Z_grid=gpuArray(Z_grid);
    P=gpuArray(P); %(z,zprime)  
end

end
