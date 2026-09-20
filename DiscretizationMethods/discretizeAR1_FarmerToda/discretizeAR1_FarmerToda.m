function [z_grid,pi_z,otheroutputs] = discretizeAR1_FarmerToda(mew,rho,sigma,znum,farmertodaoptions)
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
%   znum           - number of states in discretization of z (minimum of 3)
% Optional Inputs (farmertodaoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre', 'clenshaw-curtis','gauss-hermite')
%   nMoments       - Number of conditional moments to match (default=2)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigmaz (default depends on znum)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
%   z_grid         - (znum-by-1 or 1-by-znum) skip grid construction and build pi_z on this grid;
%                    method/nSigmas are ignored, prior is treated as in 'even' (q=normpdf)
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   pi_z           - transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - shows how many moments were matched from each grid point (for the conditional distribution)
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
% They show that this method outperforms both Tauchen and Rouwenhorst for almost all discretization of Gaussian AR(1).

if rho>=0.99
    fprintf('COMMENT: When discretizing gaussian AR(1) process with autocorrelation (rho) greater than 0.99 (which you currently have), the Rouwenhorst method tends to outperform Farmer-Toda method. \n')
    % This is based on findings of paper of Farmer & Toda (2017): last para on pg 678
    if rho>=1
        error('Farmer-Toda error, autocorellation is >=1. You cannot discretize an AR(1) with an autocorrelation coefficient of >=1')
    end
end

%% Set defaults
if ~exist('farmertodaoptions','var')
    farmertodaoptions.nMoments=2;
    % The grid half-width, in units of the standard deviation of z. sqrt(znum-1) is the width
    % the Rouwenhorst construction requires, and it is used here so the three AR(1) methods share
    % a default. It replaces min(sqrt(2*(znum-1)),3) / min(sqrt(znum-1),3), which capped the
    % width at 3 standard deviations and so stopped widening the grid past znum=10.
    farmertodaoptions.nSigmas = sqrt(znum-1);
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
        farmertodaoptions.nSigmas = sqrt(znum-1); % see the note above; one default, no cap
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

%% Check for user-supplied grid
if isfield(farmertodaoptions,'z_grid')
    farmertodaoptions.usergrid=1;
    % Must be on the cpu: the Farmer-Toda method solves an entropy problem per grid point with
    % fminunc(), which cannot take gpuArrays. This matters because a gpuArray is exactly what this
    % command RETURNS by default when a gpu is present (parallel defaults to 1+(gpuDeviceCount>0)),
    % so handing back the grid you were just given - the obvious use of this option - would
    % otherwise fail with 'FMINUNC requires all values returned by functions to be of data type
    % double'.
    farmertodaoptions.z_grid=gather(farmertodaoptions.z_grid);
    if size(farmertodaoptions.z_grid,1)>1
        farmertodaoptions.z_grid=farmertodaoptions.z_grid'; % use row internally
    end
    if length(farmertodaoptions.z_grid)~=znum
        error('length of farmertodaoptions.z_grid must equal znum')
    end
else
    farmertodaoptions.usergrid=0;
end

%% Check inputs are correctly formatted
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
if ~strcmp(farmertodaoptions.method,'even') && ~strcmp(farmertodaoptions.method,'gauss-legendre') && ~strcmp(farmertodaoptions.method,'clenshaw-curtis') && ~strcmp(farmertodaoptions.method,'gauss-hermite')
    error('farmertodaoptions.method must be one of even, gauss-legendre, clenshaw-curtis, or gauss-hermite')
end

if farmertodaoptions.nSigmas<1.2
    warning('Trying to hit the 2nd moment with farmertodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for farmertodaoptions.nSigmas<1.2).')
end


sigmaz = sigma/sqrt(1-rho^2); % unconditional standard deviation
mewz=mew/(1-rho); % unconditional mean

if farmertodaoptions.usergrid==1
    z_grid = farmertodaoptions.z_grid; % row vector
    W = ones(1,znum); % treat like 'even' for the prior q in moment matching
else
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
end

%% define conditional central moments that Farmer-Toda method targets
T1 = 0;
T2 = sigma^2;
T3 = 0;
T4 = 3*sigma^4;

TBar = [T1 T2 T3 T4]'; % vector of conditional central moments


%% Farmer-Toda method
pi_z = NaN(znum);
nMoments_grid=zeros(znum,1); % Used to record number of moments matched in transition from each point
scalingFactor = max(abs(z_grid));
kappa = 1e-8;

for ii = 1:znum
    
    condMean = mew+rho*z_grid(ii); % conditional mean
    if strcmp(farmertodaoptions.method,'gauss-hermite') && farmertodaoptions.usergrid==0  % define prior probabilities
        q = W;
    else
        q = W.*(exp(-0.5*((z_grid-condMean)./sigma).^2)./(sigma*sqrt(2*pi)));
    end
    
    if any(q < kappa)
        q(q < kappa) = kappa; % replace by small number for numerical stability
    end
    
    if farmertodaoptions.nMoments == 1 % match only 1 moment
        pi_z(ii,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
        nMoments_grid(ii)=1;
    else % match 2 moments first
        [p,lambda,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
            ((x-condMean)./scalingFactor).^2],...
            TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
        if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
            pi_z(ii,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,0,q,0);
            nMoments_grid(ii)=1;
        elseif farmertodaoptions.nMoments == 2
            pi_z(ii,:) = p;
            nMoments_grid(ii)=2;
        elseif farmertodaoptions.nMoments == 3 % 3 moments
            [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
            if norm(momentError) > 1e-5
                pi_z(ii,:) = p;
                nMoments_grid(ii)=2;
            else
                pi_z(ii,:) = pnew;
                nMoments_grid(ii)=3;
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
                    pi_z(ii,:) = p;
                    nMoments_grid(ii)=2;
                else
                    pi_z(ii,:) = pnew;
                    nMoments_grid(ii)=3;
                end
            else
                pi_z(ii,:) = pnew;
                nMoments_grid(ii)=4;
            end
        end
    end
end

% Report the maximum entropy fallbacks once for the whole call, rather than once per grid point.
% The count is what matters, and nMoments_grid says exactly where.
if farmertodaoptions.verbose==1 && sum(nMoments_grid<farmertodaoptions.nMoments)>0
    warning('Matched fewer than the requested %i moments from %i of the %i grid points, as few as %i. See otheroutputs.nMoments_grid for which.',farmertodaoptions.nMoments,sum(nMoments_grid<farmertodaoptions.nMoments),znum,min(nMoments_grid))
end

% HAVE DONE THE LAZY OPTION. THIS SHOULD REALLY BE REWRITTEN SO THAT JUST
% CREATE ON GPU OR CPU AS APPROPRIATE. (AVOID THE OVERHEAD OF MOVING TO GPU)
if farmertodaoptions.parallel==2 
    z_grid=gpuArray(z_grid);
    pi_z=gpuArray(pi_z); %(z,zprime)  
end

z_grid=z_grid'; % Output as column vector

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid; % How many moments were hit by the conditional distribution from each grid point

end
