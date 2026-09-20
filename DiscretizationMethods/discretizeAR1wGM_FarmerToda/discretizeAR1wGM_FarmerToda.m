function [z_grid,pi_z,otheroutputs] = discretizeAR1wGM_FarmerToda(mew,rho,mixprobs_i,mu_i,sigma_i,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments"
%
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation
%    of AR(1) process with gaussian mixture innovations:
%       z'=mew+rho*z+e, e~F
%          where F=sum_{i=1}^nmix mixprobs_i*N(mu_i,sigma_i^2) is a gaussian mixture
%    by Farmer-Toda method
%
% We use "nmix" to denote the number of normal distributions being mixed in the gaussian mixture innovations
%
% Inputs
%   mew            - constant term coefficient (the INTERCEPT of the AR(1), not the mean of z)
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
%   verbose        - set to zero to suppress the report of how many grid points matched fewer moments
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   pi_z           - transition matrix of the discrete approximation of z;
%                    pi_z(i,j) is the probability of transitioning from state i to state j
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - optional output that shows how many moments were  matched from each grid point
%
% Comment: mew is the intercept, so E(z)=mew/(1-rho) when the gaussian mixture is mean zero.
%          (We do not require the gaussian mixture to be mean zero. When it is not, the
%          unconditional mean of z is (mew+E(e))/(1-rho), and the grid is still centred on
%          mew/(1-rho); see discretizeLifeCycleAR1wGM_KFTT, which treats this the same way.)
%
% Note on the convention: this command used to read mew as the unconditional mean, i.e. it
% implemented z'=(1-rho)*mew+rho*z+e. It was the only command in the AR(1)/VAR(1) family to do so;
% the other ten, including its own life-cycle counterpart discretizeLifeCycleAR1wGM_KFTT, all read
% mew as the intercept. Changed to match them. This alters results for any call with mew~=0 and
% rho~=0.
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
    farmertodaoptions.verbose=1;
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
    if ~isfield(farmertodaoptions,'verbose')
        farmertodaoptions.verbose=1;
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

% Make sure method is set appropriately
% Without this the switch on method below simply falls through, leaving the grid variable
% unassigned, and the failure surfaces a line later as an undefined-variable error naming a
% variable the caller has never heard of. Checking here names the option instead.
if ~strcmp(farmertodaoptions.method,'even') && ~strcmp(farmertodaoptions.method,'gauss-legendre') && ~strcmp(farmertodaoptions.method,'clenshaw-curtis') && ~strcmp(farmertodaoptions.method,'gauss-hermite') && ~strcmp(farmertodaoptions.method,'GMQ')
    error('farmertodaoptions.method must be one of even, gauss-legendre, clenshaw-curtis, gauss-hermite, or GMQ')
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

% gmdistribution and pdf() need the Statistics Toolbox, and a gaussian mixture density is
% one line without them: sum_i p_i*normpdf(x,mu_i,s_i). The component parameters are kept
% as plain vectors and the density is written out at each use below.
gmmu = mu_i; gmsd = sqrt(sigmaC2); gmp = mixprobs_i; % the mixture, as plain vectors

sigma = sqrt(T2-T1^2); % conditional standard deviation
temp = (eye(1^2)-kron(rho,rho))\eye(1^2);
sigmaX = sigma*sqrt(temp(1,1)); % unconditional standard deviation
zstar = mew/(1-rho); % expected value of z (note: mew is the intercept of the AR(1), not the mean of z)

% construct the one dimensional grid
switch farmertodaoptions.method
    case 'even' % evenly-spaced grid
        X1 = linspace(zstar-farmertodaoptions.nSigmas*sigmaX,zstar+farmertodaoptions.nSigmas*sigmaX,znum);
        W = ones(1,znum);
    case 'gauss-legendre' % Gauss-Legendre quadrature
        [X1,W] = legpts(znum,[zstar-farmertodaoptions.nSigmas*sigmaX,zstar+farmertodaoptions.nSigmas*sigmaX]);
        X1 = X1';
    case 'clenshaw-curtis' % Clenshaw-Curtis quadrature
        [X1,W] = fclencurt(znum,zstar-farmertodaoptions.nSigmas*sigmaX,zstar+farmertodaoptions.nSigmas*sigmaX);
        X1 = fliplr(X1');
        W = fliplr(W');
    case 'gauss-hermite' % Gauss-Hermite quadrature
        if rho > 0.8
            if farmertodaoptions.verbose==1
                warning('Model is persistent; even-spaced grid is recommended')
            end
        end
        [X1,W] = GaussHermite(znum);
        X1 = zstar+sqrt(2)*sigma*X1';
        W = W'./sqrt(pi);
    case 'GMQ' % Gaussian Mixture Quadrature
        if rho > 0.8
            if farmertodaoptions.verbose==1
                warning('Model is persistent; even-spaced grid is recommended')
            end
        end
        [X1,W] = GaussianMixtureQuadrature(mixprobs_i,mu_i,sigma_i,znum);
        X1 = X1 + zstar;
end

z_grid = allcomb2(X1)'; % 1*Nm matrix of grid points

pi_z = NaN(znum,znum); % transition probability matrix
P1 = NaN(znum,znum); % matrix to store transition probability
P2 = ones(znum,1); % znum*1 matrix used to construct P
scalingFactor = max(abs(X1));
kappa = 1e-8;

nMoments_grid=zeros(znum,1);

for z_c = 1:znum

    % First, calculate what Farmer & Toda (2017) call qnn', which are essentially an initial guess for pnn'
    condMean = mew+rho*z_grid(z_c); % z_grid(z_c) here is the lag grid point
    xPDF = (X1-condMean)';
    switch farmertodaoptions.method
        case 'gauss-hermite'
            q = W.*(sum(gmp'.*exp(-0.5*((xPDF-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2)./(exp(-0.5*(xPDF./sigma).^2)./(sigma*sqrt(2*pi))))';
        case 'GMQ'
            q = W.*(sum(gmp'.*exp(-0.5*((xPDF-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2)./sum(gmp'.*exp(-0.5*((X1'-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2))';
        otherwise
            q = W.*(sum(gmp'.*exp(-0.5*((xPDF-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2))';
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
                    P1(z_c,:) = p;
                    nMoments_grid(z_c)=2;
                else
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


% Report the maximum entropy fallbacks once for the whole call, rather than once per grid point.
% The count is what matters, and nMoments_grid says exactly where.
if farmertodaoptions.verbose==1 && sum(nMoments_grid<farmertodaoptions.nMoments)>0
    warning('Matched fewer than the requested %i moments from %i of the %i grid points, as few as %i. See otheroutputs.nMoments_grid for which.',farmertodaoptions.nMoments,sum(nMoments_grid<farmertodaoptions.nMoments),znum,min(nMoments_grid))
end

if farmertodaoptions.parallel==2
    pi_z=gpuArray(pi_z);
    z_grid=gpuArray(z_grid);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid;

z_grid=z_grid'; % Column vector for output

end
