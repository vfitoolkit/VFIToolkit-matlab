function [z_grid_J, P_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_Kirkby(rho,sigma,znum,J,kirkbyoptions)
% Please cite: Kirkby (working paper)
%
% Kirkby discretization method for a 'life-cycle non-stationary AR(1) process with 
%    gaussian innovations'. 
% This is an extension of the Farmer-Toda method to 'age-dependent parameters' 
%    (essentially combine Farmer & Toda (2017) with Fella, Gallipoli & Pan (2019))
% 
%  Extended-Farmer-Toda method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = rho(j)*z(j-1)+ epsilon(j),   epsilion(j)~iid  N(0,sigma(j))
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1))
%
% Inputs:
%   rho 	     - (1-by-J) vector of serial correlation coefficients
%   sigma        - (1-by-J) standard deviations of the gaussian innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
% Optional inputs (kirkbyoptions)
%   parallel:    - set equal to 2 to use GPU, 0 to use CPU
% Output: 
%   z_grid       - an znum-by-J matrix, each column stores the Markov state space for period j
%   P            - znum-by-znum-by-J matrix of J (znum-by-znum) transition matrices. 
%                  Transition probabilities are arranged by row.
%   jequaloneDistz - initial distribution of shocks for j=1
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - shows how many moments were matched from each grid point (for the conditional distribution)
%        otheroutputs.sigma_z     - the standard deviation of z at each age (used to determine grid)
%
% !========================================================================%
% Original paper:
% Kirkby (working paper)

sigma_z = zeros(1,J);
% z_grid = zeros(znum,J); 
P_J = zeros(znum,znum,J);


%% Set options
if ~exist('kirkbyoptions','var')
    kirkbyoptions.method='even'; % Informally I have the impression even is more robust
    kirkbyoptions.nMoments=4; % Have used 4 as the point of gaussian mixtures is typically to get higher order moments. 4 moments covers skewness and kurtosis.
    if rho <= 1-2/(znum-1)  % This is just what Toda used.
        kirkbyoptions.nSigmas = min(sqrt(2*(znum-1)),4); % Maximum of +-4 standard deviation
    else
        kirkbyoptions.nSigmas = min(sqrt(znum-1),4); % Maximum of +-4 standard deviations
    end
    kirkbyoptions.parallel=1+(gpuDeviceCount>0);
    kirkbyoptions.setmixturemutoenforcezeromean=0;
else
    if ~isfield(kirkbyoptions,'method')
        kirkbyoptions.method='even'; % Informally I have the impression even is more robust
    end
    if ~isfield(kirkbyoptions,'nMoments')
        kirkbyoptions.nMoments = 4;  % Have used 4 as the point of gaussian mixtures is typically to get higher order moments. 4 covers skewness and kurtosis.
    end
    if ~isfield(kirkbyoptions,'nSigmas')
        if rho <= 1-2/(znum-1) % This is just what Toda used.
            kirkbyoptions.nSigmas = min(sqrt(2*(znum-1)),4); % Maximum of +-4 standard deviation
        else
            kirkbyoptions.nSigmas = min(sqrt(znum-1),4); % Maximum of +-4 standard deviation
        end
    end
    if ~isfield(kirkbyoptions,'parallel')
        kirkbyoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(kirkbyoptions,'setmixturemutoenforcezeromean')
        kirkbyoptions.setmixturemutoenforcezeromean=0;
    end
end

%% some error checking
if znum < 2
    error('The state space has to have dimension znum>1.')
end
if J < 2
    error('The time horizon has to have dimension J>1.')
end

if any(sigma < 0)
    error('standard deviations must be positive')
end
if size(sigma,2)~=J
    error('sigma_j must have J columns')
end
% if any(rho >= 1)
%     error('autocorrelation coefficient (spectral radius) must be less than one')
% end

% Check that znum is a valid number of grid points
if ~isnumeric(znum) || znum < 3 || rem(znum,1) ~= 0
    error('znum must be a positive integer greater than 3')
end
% Check that nMoments is a valid number
if ~isnumeric(kirkbyoptions.nMoments) || kirkbyoptions.nMoments < 1 || kirkbyoptions.nMoments > 4 || ~((rem(kirkbyoptions.nMoments,1) == 0) || (kirkbyoptions.nMoments == 1))
    error('kirkbyoptions.nMoments must be either 1, 2, 3, 4')
end

% % Everything has to be on cpu otherwise fminunc throws an error
% if kirkbyoptions.parallel==2
%     rho=gather(rho);
%     sigma_j=gather(sigma_j);
%     znum=gather(znum);
%     J=gather(J);
% end

%% Step 1: compute the conditional moments (will need the standard deviations to create grid)
% sigma is of size (1,J); % standard deviation of innovations

% Because the innovations are zero-mean normal distributions we can
% calculate the first four uncentered moments as,
T1 = zeros(1,J); % mean
T2 = sigma.^2; % uncentered second moment
T3 = zeros(1,J); % uncentered third moment
T4 = 3*sigma.^4; % uncentered fourth moment
TBar_J = [T1; T2; T3; T4]; % (4,J), The 4 conditional moments for each period

%% Step 2: construct the state space z_grid_J(j) in each period j.
% Evenly-spaced N-state space over [-kirkbyoptions.nSigmas*sigma_z(j),kirkbyoptions.nSigmas*sigma_z(j)].
% By default I assume z0=0, but you can set it as N(0,sigma_z0) using
if isfield(kirkbyoptions,'initialj0sigma_z')
	[z_grid_0,P_0] = discretizeAR1_FarmerToda(0,0,kirkbyoptions.initialj0sigma_z,znum);
    jequalzeroDistz=P_0(1,:)'; % iid, so first row is the dist
else
    z_grid_0=zeros(znum,1);
    jequalzeroDistz=[1;zeros(znum-1,1)]; % Is irrelevant where we put the mass
end
clear P_0

if isfield(kirkbyoptions,'initialj0sigma_z')
    sigma_z(1) = sqrt(rho(1)^2*kirkbyoptions.initialj0sigma_z^2+sigma(1)^2);
else
    sigma_z(1) = sigma(1);
end
for jj = 2:J
    sigma_z(jj) = sqrt(rho(jj)^2*sigma_z(jj-1)^2+sigma(jj)^2);
end

mew=0; % It is enforced that the process is mean zero
z_grid_J=zeros(znum,J);
for jj=1:J
    % construct the one dimensional grid
    switch kirkbyoptions.method
        case 'even' % evenly-spaced grid
            X1 = linspace(mew-kirkbyoptions.nSigmas*sigma_z(jj),mew+kirkbyoptions.nSigmas*sigma_z(jj),znum);
            W = ones(1,znum);
        case 'gauss-legendre' % Gauss-Legendre quadrature
            [X1,W] = legpts(znum,[mew-kirkbyoptions.nSigmas*sigma_z(jj),mew+kirkbyoptions.nSigmas*sigma_z(jj)]);
            X1 = X1';
        case 'clenshaw-curtis' % Clenshaw-Curtis quadrature
            [X1,W] = fclencurt(znum,mew-kirkbyoptions.nSigmas*sigma_z(jj),mew+kirkbyoptions.nSigmas*sigma_z(jj));
            X1 = fliplr(X1');
            W = fliplr(W');
        case 'gauss-hermite' % Gauss-Hermite quadrature
            if rho(jj) > 0.8
                warning('Model is persistent; even-spaced grid is recommended')
            end
            [X1,W] = GaussHermite(znum);
            X1 = mew+sqrt(2)*sigma_z(jj)*X1';
            W = W'./sqrt(pi);
    end
    
    z_grid = allcomb2(X1); % Nm*1 matrix of grid points
    z_grid_J(:,jj)=z_grid;
end

%% Step 3: Compute the transition matrices trans(:,:,t) from period (t-1) to period t

nMoments_grid=zeros(znum,J); % Used to record number of moments matched in transition from each point

kappa = 1e-8;

% Note: In what follows I include mew, even though it is necessarily zero.
for jj=1:J
    
    if jj>1
        zlag_grid=z_grid_J(:,jj-1);
    else
        zlag_grid=z_grid_0;
    end
    z_grid=z_grid_J(:,jj)';
    
    TBar=TBar_J(:,jj);

    P = nan(znum,znum); % The transition matrix for age jj
    scalingFactor = max(abs(zlag_grid)); %SHOULD THIS BE lag???
    
    for ii = 1:znum
        
        condMean = mew*(1-rho(jj))+rho(jj)*zlag_grid(ii); % conditional mean
        if strcmp(kirkbyoptions.method,'gauss-hermite')  % define prior probabilities
            q = W;
        else
            q = W.*normpdf(z_grid,condMean,sigma(jj));
        end
        
        if any(q < kappa)
            q(q < kappa) = kappa; % replace by small number for numerical stability
        end
        
        if kirkbyoptions.nMoments == 1 % match only 1 moment
            P(ii,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
        else % match 2 moments first
            [p,lambda,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2],...
                TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
            if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                warning('Failed to match first 2 moments. Just matching 1.')
                P(ii,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,0,q,0);
            elseif kirkbyoptions.nMoments == 2
                P(ii,:) = p;
            elseif kirkbyoptions.nMoments == 3 % 3 moments
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError) > 1e-5
                    warning('Failed to match first 3 moments.  Just matching 2.')
                    P(ii,:) = p;
                else
                    P(ii,:) = pnew;
                end
            elseif kirkbyoptions.nMoments == 4 % 4 moments
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
                    ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
                if norm(momentError) > 1e-5
                    %warning('Failed to match first 4 moments.  Just matching 3.')
                    [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                    if norm(momentError) > 1e-5
                        warning('Failed to match first 3 moments.  Just matching 2.')
                        P(ii,:) = p;
                    else
                        P(ii,:) = pnew;
                        warning('Failed to match first 4 moments.  Just matching 3.')
                    end
                else
                    P(ii,:) = pnew;
                end
            end
        end
    end
    P_J(:,:,jj)=P;
    
end

%%
jequaloneDistz=P_J(:,:,1)'*jequalzeroDistz;

%% Change P_J so that P_J(:,:,jj) is the transition matrix from period jj to period jj+1
P_J(:,:,1:end-1)=P_J(:,:,2:end);

%% For jj=J, P_J(:,:,J) is kind of meaningless (there is no period J+1 to transition to). I just fill it in as a uniform distribution
P_J(:,:,J)=ones(znum,znum)/znum;

%% I AM BEING LAZY AND JUST MOVING RESULT TO GPU RATHER THAN CREATING IT THERE IN THE FIRST PLACE
if kirkbyoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    P_J=gpuArray(P_J);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid; % Heatmap of how many moments where hit by the conditional (difference) distribution
otheroutputs.sigma_z=sigma_z; % Standard deviation of z (for each period)

end



