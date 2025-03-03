function [z_grid,pi_z,otheroutputs] = discretizeAR1wGM_FarmerToda(mew,rho,mixprobs_i,mu_i,sigma_i,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments"
%
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation
%    of AR(1) process with gaussian mixture innovations:
%       z'=(1-rho)*mew+rho*z+e, e~F
%          where F=sum_{i=1}^nmix mixprobs_i*N(mu_i,sigma_i^2) is a gaussian mixture
%    by Farmer-Toda method
%
% We use "nmix" to denote the number of normal distributions being mixed in the gaussian mixture innovations
%
% Inputs
%   mew            - 'constant' in above formula (note the (1-rho)*mew)
%   rho            - autocorrelation coefficient
%   mixprobs_i     - (nmix-by-1) mixture probabilities of the gaussian mixture innovations (must sum to 1)
%   mus_i          - (nmix-by-1) means of the gaussian mixture innovations
%   sigma_i        - (nmix-by-1) standard deviations of the gaussian mixture innovations
%   znum           - number of states in discretization of z (scalar)
% Optional Inputs (farmertodaoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre', 'clenshaw-curtis','gauss-hermite')
%   nMoments       - Number of conditional moments to match (default=2)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigmaz (default depends on znum)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   pi_z           - transition matrix of the discrete approximation of z;
%                    pi_z(i,j) is the probability of transitioning from state i to state j
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - optional output that shows how many moments were  matched from each grid point
%
% Comment: mew is what would be the unconditional mean of z if the gaussian mixture were mean zero (but we do not require the gaussian mixture to be mean zero).
%
%
% This code is modified from that of Toda & Farmer (v: https://github.com/alexisakira/discretization
% Please cite them if you use this.
% This version was lightly modified by Robert Kirkby
%
%%%%%%%%%%%%%%%
% Original paper:
% Farmer & Toda (2017) - Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments

%% Set default options
if ~exist('farmertodaoptions','var')
    farmertodaoptions.method='even'; % Informally I have the impression even is more robust
    farmertodaoptions.nMoments=4; % Toda had default set at 2, I have used 4 as the point of gaussian mixtures is typically to get higher order moments
    if rho <= 1-2/(znum-1)  % This is just what Toda used.
        farmertodaoptions.nSigmas = sqrt(2*(znum-1));
    else
        farmertodaoptions.nSigmas = sqrt(znum-1);
    end
    farmertodaoptions.parallel=1+(gpuDeviceCount>0);
else
    if ~isfield(farmertodaoptions,'method')
        farmertodaoptions.method='even'; % Informally I have the impression even is more robust
    end
    if ~isfield(farmertodaoptions,'nMoments')
        farmertodaoptions.nMoments = 4;  % Toda had default set at 2, I have used 4 as the point of gaussian mixtures is typically to get higher order moments
    end
    if ~isfield(farmertodaoptions,'nSigmas')
        if rho <= 1-2/(znum-1) % This is just what Toda used.
            farmertodaoptions.nSigmas = sqrt(2*(znum-1));
        else
            farmertodaoptions.nSigmas = sqrt(znum-1);
        end
    end
    if ~isfield(farmertodaoptions,'parallel')
        farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    end
end

%% some error checking
if any(mixprobs_i < 0)
    error('mixture proportions must be positive')
end
if any(sigma_i < 0)
    error('standard deviations must be positive')
end
if sum(mixprobs_i) ~= 1
    error('mixture proportions must add up to 1')
end

if size(mixprobs_i,1) < size(mixprobs_i,2)
    mixprobs_i = mixprobs_i'; % convert to column vector
end
if size(mu_i,1) < size(mu_i,2)
    mu_i = mu_i'; % convert to column vector
end
if size(sigma_i,1) < size(sigma_i,2)
    sigma_i = sigma_i'; % convert to column vector
end

if rho >= 1
    error('autocorrelation coefficient (spectral radius) must be less than one')
end

% Check that Nm is a valid number of grid points
if ~isnumeric(znum) || znum < 3 || rem(znum,1) ~= 0
    error('Nm must be a positive integer greater than 3')
end
% Check that nMoments is a valid number
if ~isnumeric(farmertodaoptions.nMoments) || farmertodaoptions.nMoments < 1 || farmertodaoptions.nMoments > 4 || ~((rem(farmertodaoptions.nMoments,1) == 0) || (farmertodaoptions.nMoments == 1))
    error('farmertodaoptions.nMoments must be either 1, 2, 3, 4')
end

if farmertodaoptions.nSigmas<1.2
    warning('Trying to hit the 2nd moment with farmertodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for farmertodaoptions.nSigmas<1.2).')
end


%% compute conditional moments
sigmaC2 = sigma_i.^2;
T1 = mixprobs_i'*mu_i; % mean
T2 = mixprobs_i'*(mu_i.^2+sigmaC2); % uncentered second moment
T3 = mixprobs_i'*(mu_i.^3+3*mu_i.*sigmaC2); % uncentered third moment
T4 = mixprobs_i'*(mu_i.^4+6*(mu_i.^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment

TBar = [T1 T2 T3 T4]';

nComp = length(mixprobs_i); % number of mixture components
temp = zeros(1,1,nComp);
temp(1,1,:) = sigmaC2;
gmObj = gmdistribution(mu_i,temp,mixprobs_i); % define the Gaussian mixture object

sigma = sqrt(T2-T1^2); % conditional standard deviation
temp = (eye(1^2)-kron(rho,rho))\eye(1^2);
sigmaX = sigma*sqrt(temp(1,1)); % unconditional standard deviation

% construct the one dimensional grid
switch farmertodaoptions.method
    case 'even' % evenly-spaced grid
        X1 = linspace(mew-farmertodaoptions.nSigmas*sigmaX,mew+farmertodaoptions.nSigmas*sigmaX,znum);
        W = ones(1,znum);
    case 'gauss-legendre' % Gauss-Legendre quadrature
        [X1,W] = legpts(znum,[mew-farmertodaoptions.nSigmas*sigmaX,mew+farmertodaoptions.nSigmas*sigmaX]);
        X1 = X1';
    case 'clenshaw-curtis' % Clenshaw-Curtis quadrature
        [X1,W] = fclencurt(znum,mew-farmertodaoptions.nSigmas*sigmaX,mew+farmertodaoptions.nSigmas*sigmaX);
        X1 = fliplr(X1');
        W = fliplr(W');
    case 'gauss-hermite' % Gauss-Hermite quadrature
        if rho > 0.8
            warning('Model is persistent; even-spaced grid is recommended')
        end
        [X1,W] = GaussHermite(znum);
        X1 = mew+sqrt(2)*sigma*X1';
        W = W'./sqrt(pi);
    case 'GMQ' % Gaussian Mixture Quadrature
        if rho > 0.8
            warning('Model is persistent; even-spaced grid is recommended')
        end
        [X1,W] = GaussianMixtureQuadrature(mixprobs_i,mu_i,sigma_i,znum);
        X1 = X1 + mew;
end

z_grid = allcomb2(X1)'; % 1*Nm matrix of grid points

pi_z = NaN(znum,znum); % transition probability matrix
P1 = NaN(znum,znum); % matrix to store transition probability
P2 = ones(znum,1); % znum*1 matrix used to construct P
scalingFactor = max(abs(X1));
kappa = 1e-8;

nMoments_grid=zeros(znum,1);

for z_c = 1:znum

    % First, calculate what Farmer & Toda (2017) call qnn', which are essentially an inital guess for pnn'
    condMean = mew*(1-sum(rho))+rho*z_grid(z_c); % z_grid(z_c) here is the lag grid point
    xPDF = (X1-condMean)';
    switch farmertodaoptions.method
        case 'gauss-hermite'
            q = W.*(pdf(gmObj,xPDF)./normpdf(xPDF,0,sigma))';
        case 'GMQ'
            q = W.*(pdf(gmObj,xPDF)./pdf(gmObj,X1'))';
        otherwise
            q = W.*(pdf(gmObj,xPDF))';
    end
    
    if any(q < kappa)
        q(q < kappa) = kappa;
    end    
            
    if farmertodaoptions.nMoments == 1 % match only 1 moment
        P1(z_c,:) = discreteApproximation(X1,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
        nMoments_grid(z_c)=1;
    else % match 2 moments first
        [p,lambda,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
            ((x-condMean)./scalingFactor).^2],...
            TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
        if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
            warning('Failed to match first 2 moments. Just matching 1.')
            P1(z_c,:) = discreteApproximation(X1,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
            nMoments_grid(z_c)=1;
        elseif farmertodaoptions.nMoments == 2
            P1(z_c,:) = p;
            nMoments_grid(z_c)=2;
        elseif farmertodaoptions.nMoments == 3
            [pnew,~,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
            if norm(momentError) > 1e-5
                warning('Failed to match first 3 moments.  Just matching 2.')
                P1(z_c,:) = p;
                nMoments_grid(z_c)=2;
            else
                P1(z_c,:) = pnew;
                nMoments_grid(z_c)=3;
            end
        else % 4 moments
            [pnew,~,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
                ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
            if norm(momentError) > 1e-5
                %warning('Failed to match first 4 moments.  Just matching 3.')
                [pnew,~,momentError] = discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError) > 1e-5
                    warning('Failed to match first 3 moments.  Just matching 2.')
                    P1(z_c,:) = p;
                    nMoments_grid(z_c)=2;
                else
                    warning('Failed to match first 4 moments.  Just matching 3.')
                    P1(z_c,:) = pnew;
                    nMoments_grid(z_c)=3;
                end
            else
                P1(z_c,:) = pnew;
                nMoments_grid(z_c)=4;
            end
        end
        pi_z(z_c,:) = kron(P1(z_c,:),P2(z_c,:));
    end
    
end


if farmertodaoptions.parallel==2
    pi_z=gpuArray(pi_z);
    z_grid=gpuArray(z_grid);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid;

z_grid=z_grid'; % Column vector for output

end
