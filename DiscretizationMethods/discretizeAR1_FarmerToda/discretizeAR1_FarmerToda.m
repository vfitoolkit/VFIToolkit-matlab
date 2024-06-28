function [z_grid,pi_z] = discretizeAR1_FarmerToda(mew,rho,sigma,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments
% [If you use this to discretize and iid normal (rho=0) then instead please cite
% Tanaka & Toda (2013) - "Discrete approximations of continuous distributions by maximum entropy" instead.]
%
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation 
%    of AR(1) process z'=mew+rho*z+e, e~N(0,sigma^2), by Farmer-Toda method
%
% Inputs
%   mew            - constant term coefficient
%   rho            - autocorrelation coefficient
%   sigma          - standard deviation of (gaussian) innovations
%   znum           - number of states in discretization of z (minumum of 3)
% Optional Inputs (farmertodaoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre', 'clenshaw-curtis','gauss-hermite')
%   nMoments       - Number of conditional moments to match (default=2)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigmaz (default depends on znum)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   pi_z           - transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
% Helpful info:
%   Var(z)=(sigma^2)/(1-rho^2). So sigmaz=sigma/sqrt(1-rho^2);   sigma=sigmaz*sqrt(1-rho^2)
%                                  where sigmaz= standard deviation of z
%     E(z)=mew/(1-rho)
%
% This code is modified from that of Toda & Farmer (v: https://github.com/alexisakira/discretization
% Please cite them if you use this.
% This version was lightly modified by Robert Kirkby
% 
%%%%%%%%%%%%%%%
% Original paper:
% Farmer & Toda (2017) - Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments
% They how that this method outperforms both Tauchen and Rouwenhorst for almost all discretization of Gaussian AR(1).

if rho>=0.99
    fprintf('COMMENT: When discretizing gaussian AR(1) process with autocorrelation (rho) greater than 0.99 (which you currently have), the Rouwenhorst method tends to outperform Farmer-Toda method. \n')
    % This is based on findings of paper of Farmer & Toda (2017): last para on pg 678
    if rho>=1
        error('Farmer-Toda error, autocorellation is 1. You cannot discretize an AR(1) with an autocorrelation coefficient of 1')
    end
end

%% Set defaults
if ~exist('farmertodaoptions','var')
    farmertodaoptions.nMoments=2;
    if abs(rho) <= 1-2/(znum-1)
        farmertodaoptions.nSigmas = min(sqrt(2*(znum-1)),3);
    else
        farmertodaoptions.nSigmas = min(sqrt(znum-1),3); % Set max of 3
    end
    if rho<=0.8
        farmertodaoptions.method='gauss-hermite';
    else
        farmertodaoptions.method='even';
    end
    farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    farmertodaoptions.verbose=1;
else
    if ~isfield(farmertodaoptions,'nMoments')
        farmertodaoptions.nMoments = 2; % Default number of moments to match is 2      
    end
    % define grid spacing parameter if not provided
    if ~isfield(farmertodaoptions,'nSigmas') % This is just direct from Farmer-Toda code. I am not aware of any results showing it performs 'better'
        if abs(rho) <= 1-2/(znum-1)
            farmertodaoptions.nSigmas = min(sqrt(2*(znum-1)),3);
        else
            farmertodaoptions.nSigmas = min(sqrt(znum-1),3);
        end
    end
    % Set method based on findings of paper of Farmer & Toda (2017): last para on pg 678
    %   method='even' for rho>0.8, 'gauss-hermite' for rho<=0.8
    if ~isfield(farmertodaoptions,'method')
        if rho<=0.8
            farmertodaoptions.method='gauss-hermite';
        else
            farmertodaoptions.method='even';
        end
    end
    if ~isfield(farmertodaoptions,'parallel')
        farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(farmertodaoptions,'verbose')
        farmertodaoptions.verbose=1;
    end
end
% Note: the choice of setting nSigmas to sqrt(znum-1) is based on asymptotic theory in Corrallary 3.5(ii) of Farmer & Toda (2017)

%% Check inputs are correctly formatted
% Check that Nm is a valid number of grid points
if ~isnumeric(znum) || znum < 3 || rem(znum,1) ~= 0
    error('Nm must be a positive integer greater than 3')
end

% Check that nMoments is a valid number
if ~isnumeric(farmertodaoptions.nMoments) || farmertodaoptions.nMoments < 1 || farmertodaoptions.nMoments > 4 || ~((rem(farmertodaoptions.nMoments,1) == 0) || (farmertodaoptions.nMoments == 1))
    error('farmertodaoptions.nMoments must be either 1, 2, 3, 4')
end

sigmaz = sigma/sqrt(1-rho^2); % unconditional standard deviation
mewz=mew/(1-rho); % unconditional mean

switch farmertodaoptions.method
    case 'even'
        z_grid = linspace(mewz-farmertodaoptions.nSigmas*sigmaz,mewz+farmertodaoptions.nSigmas*sigmaz,znum);
        W = ones(1,znum);
    case 'gauss-legendre'
        [z_grid,W] = legpts(znum,[mewz-farmertodaoptions.nSigmas*sigmaz,mewz+farmertodaoptions.nSigmas*sigmaz]);
        z_grid = z_grid';
    case 'clenshaw-curtis'
        [z_grid,W] = fclencurt(znum,mewz-farmertodaoptions.nSigmas*sigmaz,mewz+farmertodaoptions.nSigmas*sigmaz);
        z_grid = fliplr(z_grid');
        W = fliplr(W');
    case 'gauss-hermite'
        [z_grid,W] = GaussHermite(znum);
        z_grid = mewz+sqrt(2)*sigma*z_grid';
        W = W'./sqrt(pi);
end

%% define conditional central moments that Farmer-Toda method targets
T1 = 0;
T2 = sigma^2;
T3 = 0;
T4 = 3*sigma^4;

TBar = [T1 T2 T3 T4]'; % vector of conditional central moments


%% Farmer-Toda method
pi_z = NaN(znum);
scalingFactor = max(abs(z_grid));
kappa = 1e-8;

for ii = 1:znum
    
    condMean = mew+rho*z_grid(ii); % conditional mean
    if strcmp(farmertodaoptions.method,'gauss-hermite')  % define prior probabilities
        q = W;
    else
        q = W.*normpdf(z_grid,condMean,sigma);
    end
    
    if any(q < kappa)
        q(q < kappa) = kappa; % replace by small number for numerical stability
    end
    
    if farmertodaoptions.nMoments == 1 % match only 1 moment
        pi_z(ii,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
    else % match 2 moments first
        [p,lambda,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
            ((x-condMean)./scalingFactor).^2],...
            TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
        if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
            if farmertodaoptions.verbose==1
                warning('Failed to match first 2 moments. Just matching 1.')
            end
            pi_z(ii,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,0,q,0);
        elseif farmertodaoptions.nMoments == 2
            pi_z(ii,:) = p;
        elseif farmertodaoptions.nMoments == 3 % 3 moments
            [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
            if norm(momentError) > 1e-5
                if farmertodaoptions.verbose==1
                    warning('Failed to match first 3 moments.  Just matching 2.')
                end
                pi_z(ii,:) = p;
            else
                pi_z(ii,:) = pnew;
            end
        elseif farmertodaoptions.nMoments == 4 % 4 moments
            [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
                ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
            if norm(momentError) > 1e-5
                %warning('Failed to match first 4 moments.  Just matching 3.')
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError) > 1e-5
                    if farmertodaoptions.verbose==1
                        warning('Failed to match first 3 moments.  Just matching 2.')
                    end
                    pi_z(ii,:) = p;
                else
                    pi_z(ii,:) = pnew;
                    if farmertodaoptions.verbose==1
                        warning('Failed to match first 4 moments.  Just matching 3.')
                    end
                end
            else
                pi_z(ii,:) = pnew;
            end
        end
    end
end

% HAVE DONE THE LAZY OPTION. THIS SHOULD REALLY BE REWRITTEN SO THAT JUST
% CREATE ON GPU OR CPU AS APPROPRIATE. (AVOID THE OVERHEAD OF MOVING TO GPU)
if farmertodaoptions.parallel==2 
    z_grid=gpuArray(z_grid);
    pi_z=gpuArray(pi_z); %(z,zprime)  
end

z_grid=z_grid'; % Output as column vector

end