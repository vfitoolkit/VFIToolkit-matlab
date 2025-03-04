function [z_grid_J, pi_z_J, jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1wGM_KFTT(mew,rho,mixprobs_i,mu_i,sigma_i,znum,J,kfttoptions)
% Please cite: Kirkby (working paper)
%
% KFTT discretization method for a 'life-cycle non-stationary AR(1) process with
%    gaussian-mixture innovations'. 
% This is an extension of the Farmer-Toda method to 'age-dependent parameters' 
%    (essentially combine Farmer & Toda (2017) with Fella, Gallipoli & Pan (2019)
% Which in turn is an extension of Tanaka & Toda (2013).
% Hence KFTT=Kirkby-Farmer-Tanaka-Toda
% 
%  KFTT method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = mew(j)+rho(j)*z(j-1)+ epsilon(j),   epsilion(j)~iid F(j)
%           where F(j)=sum_{i=1}^nmix mixprobs_i(j)*N(mu_i(j),sigma_i(j)^2) is a gaussian mixture
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1)) 
%
%  Note: n, the number of normal distributions in the gaussian mixture, cannot depend on j
%
% Inputs:
%   mew          - (1-by-J) vector of 'drifts'
%   rho 	     - (1-by-J) vector of serial correlation coefficients
%   sigma_i      - (nmix-by-J) vector of standard deviations of innovations
%   mixprobs_i   - (nmix-by-J) mixture probabilities of the gaussian mixture innovations (must sum to 1)
%   mu_i         - (nmix-by-J) means of the gaussian mixture innovations
%   sigma_i      - (nmix-by-J) standard deviations of the gaussian mixture innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
% Optional inputs (kfttoptions)
%   parallel:    - set equal to 2 to use GPU, 0 to use CPU
%   nSigmas      - the grid used will be +-nSigmas*(standard deviation of z)
% Output: 
%   z_grid       - an znum-by-J matrix, each column stores the Markov state space for period j
%   P            - znum-by-znum-by-J matrix of J (znum-by-znum) transition matrices. 
%                  Transition probabilities are arranged by row.
%   jequaloneDistz - initial distribution of shocks for j=1
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - shows how many moments were  matched from each grid point (for the conditional distribution)
%        otheroutputs.sigma_z     - the standard deviation of z at each age (used to determine grid)
%
% !========================================================================%
% Original paper:
% Kirkby (working paper)

mewz=zeros(1,J); % period j mean of z
sigmaz = zeros(1,J);
% z_grid_J = zeros(znum,J); 
pi_z_J = zeros(znum,znum,J);

%% Set options
if ~exist('kfttoptions','var')
    kfttoptions.method='even'; % Informally I have the impression even is more robust
    kfttoptions.nMoments=4; % Have used 4 as the point of gaussian mixtures is typically to get higher order moments. 4 moments covers skewness and kurtosis.
    if rho <= 1-2/(znum-1)  % This is just what Toda used.
        kfttoptions.nSigmas = min(sqrt(2*(znum-1)),4); % Maximum of +-4 standard deviation
    else
        kfttoptions.nSigmas = min(sqrt(znum-1),4); % Maximum of +-4 standard deviations
    end
    kfttoptions.parallel=1+(gpuDeviceCount>0);
    kfttoptions.setmixturemutoenforcezeromean=0;
else
    if ~isfield(kfttoptions,'method')
        kfttoptions.method='even'; % Informally I have the impression even is more robust
    end
    if ~isfield(kfttoptions,'nMoments')
        kfttoptions.nMoments = 4;  % Have used 4 as the point of gaussian mixtures is typically to get higher order moments. 4 covers skewness and kurtosis.
    end
    if ~isfield(kfttoptions,'nSigmas')
        if rho <= 1-2/(znum-1) % This is just what Toda used.
            kfttoptions.nSigmas = min(sqrt(2*(znum-1)),4); % Maximum of +-4 standard deviation
        else
            kfttoptions.nSigmas = min(sqrt(znum-1),4); % Maximum of +-4 standard deviation
        end
    end
    if ~isfield(kfttoptions,'parallel')
        kfttoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(kfttoptions,'setmixturemutoenforcezeromean')
        kfttoptions.setmixturemutoenforcezeromean=0;
    end
end

%% some error checking
if znum < 2
    error('The state space has to have dimension znum>1. Exiting.')
end
if J < 2
    error('The time horizon has to have dimension J>1. Exiting.')
end

if any(mixprobs_i < 0)
    error('mixture proportions must be positive')
end
if ~any(sum(mixprobs_i,1))
    error('mixture proportions must add up to 1 (for each age)')
end

if any(sigma_i < 0)
    error('standard deviations must be positive')
end

if size(mew,2)~=J
    if isscalar(mew)
        mew=mew*ones(1,J); % assume that scalars are simply age-independent parameters
        % No warning, as good odds this is just a zero
    else
        error('mew_j must have J columns')
    end
end
if size(mixprobs_i,2)~=J
    error('mixprobs_i must have J columns')
end
if size(mu_i,2)~=J
    error('mu_i must have J columns')
end
if size(sigma_i,2)~=J
    error('sigma_i must have J columns')
end

nmix=size(sigma_i,1);
if kfttoptions.setmixturemutoenforcezeromean==0
    if size(mu_i,1)~=nmix
        error('sigma_i and mu_i must all have same number of rows (nmix)')
    end
end
if size(mixprobs_i,1)~=nmix
    error('sigma_i and mixprobs_i must all have same number of rows (nmix)')
end

% if any(rho >= 1)
%     error('autocorrelation coefficient (spectral radius) must be less than one')
% end

% Check that znum is a valid number of grid points
if ~isnumeric(znum) || znum < 3 || rem(znum,1) ~= 0
    error('znum must be a positive integer greater than 3')
end
% Check that nMoments is a valid number
if ~isnumeric(kfttoptions.nMoments) || kfttoptions.nMoments < 1 || kfttoptions.nMoments > 4 || ~((rem(kfttoptions.nMoments,1) == 0) || (kfttoptions.nMoments == 1))
    error('kfttoptions.nMoments must be either 1, 2, 3, 4')
end

if kfttoptions.nSigmas<1.2
    warning('Trying to hit the 2nd moment with kfttoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for kfttoptions.nSigmas<1.2).')
end

% % Everything has to be on cpu otherwise fminunc throws an error
% if kfttoptions.parallel==2
rho=gather(rho);
mixprobs_i=gather(mixprobs_i);
mu_i=gather(mu_i);
sigma_i=gather(sigma_i);
znum=gather(znum);
J=gather(J);
% end


%%
if kfttoptions.setmixturemutoenforcezeromean==1
    mu_i=[mu_i;zeros(1,J)]; % Need to fill in the last of the mu_i to get mu=0
    for jj=1:J
        mu_i(end,jj)=-(sum(mu_i(1:end-1,jj).*mixprobs_i(1:end-1,jj)))/mixprobs_i(end,jj); % Simple rearrangement of mu_i(:,jj).*mixprob_i(:,jj)=0, which is the requirement that mean of gaussian-mixture innovations=0
    end
end
% Note: when using kfttoptions.setmixturemutoenforcezeromean it must be
% the 'last' mu_i that is missing and which will be set to enforce zero
% mean of the gaussian-mixture innovations.


%% Step 1: compute the conditional moments (will need the standard deviations to create grid)
sigma=zeros(1,J); % standard deviation of innovations
TBar_J=zeros(4,J); % The 4 conditional moments for each period
for jj=1:J
    %% compute conditional moments
    sigmaC2 = sigma_i(:,jj).^2;
    T1 = mixprobs_i(:,jj)'*mu_i(:,jj); % mean
    T2 = mixprobs_i(:,jj)'*(mu_i(:,jj).^2+sigmaC2); % uncentered second moment
    T3 = mixprobs_i(:,jj)'*(mu_i(:,jj).^3+3*mu_i(:,jj).*sigmaC2); % uncentered third moment
    T4 = mixprobs_i(:,jj)'*(mu_i(:,jj).^4+6*(mu_i(:,jj).^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment
    
    TBar_J(:,jj) = [T1 T2 T3 T4]';
    
    % Convert the uncentered moments to the centered second moment
    % https://stats.stackexchange.com/questions/226138/converting-central-moments-to-non-central-moments-and-back
    % Then take square root of the centered second moment to get the standard deviation
    sigma(jj) = sqrt(T2-T1^2); % conditional standard deviation    
end

% MAYBE I SHOULD ADD A CHECK HERE THAT T1=0
if kfttoptions.setmixturemutoenforcezeromean==1
    if any(TBar_J(1,:)~=0)
        warning('Mean of gaussian-mixture innovations is not equal to zero for all agej')
    end
end

%% Step 2: construct the state space z_grid_J(j) in each period j.
% Evenly-spaced N-state space over [-kfttoptions.nSigmas*sigma_y(t),kfttoptions.nSigmas*sigma_y(t)].
% By default I assume z0=0
z0=0;
% You can change the mean of z0 using
if isfield(kfttoptions,'initialj0mewz')
    z0=kfttoptions.initialj0mewz;
end
% You can add variance to z0 as a N(z0,initialj0sigmaz) using
if isfield(kfttoptions,'initialj0sigma_z')
    farmertodaoptions.nSigmas=kfttoptions.nSigmas;
    farmertodaoptions.method=kfttoptions.method;
    farmertodaoptions.parallel=1; % need to get solution on cpu, otherwise causes errors later
	[z_grid_0,pi_z_0] = discretizeAR1_FarmerToda(0,0,kfttoptions.initialj0sigma_z,znum,farmertodaoptions);
    jequalzeroDistz=pi_z_0(1,:)'; % iid, so first row is the dist
    clear pi_z_0
else
    z_grid_0=zeros(znum,1);
    jequalzeroDistz=[1;zeros(znum-1,1)]; % Is irrelevant where we put the mass
end

if isfield(kfttoptions,'initialj0sigma_z')
    sigmaz(1) = sqrt(rho(1)^2*kfttoptions.initialj0sigma_z^2+sigma(1)^2);
else
    sigmaz(1) = sigma(1);
end
mewz(1)=mew(1)+rho(1)*z0;

% Now that we have period 1, just fill in the rest of the periods
for jj = 2:J
    sigmaz(jj) = sqrt(rho(jj)^2*sigmaz(jj-1)^2+sigma(jj)^2);
end
for jj=2:J
    mewz(jj)=mew(jj)+rho(jj)*mewz(jj-1);
end


z_grid_J=zeros(znum,J);
for jj=1:J
    % construct the one dimensional grid
    switch kfttoptions.method
        case 'even' % evenly-spaced grid
            X1 = linspace(mewz(jj)-kfttoptions.nSigmas*sigmaz(jj),mewz(jj)+kfttoptions.nSigmas*sigmaz(jj),znum);
            W = ones(1,znum);
        case 'gauss-legendre' % Gauss-Legendre quadrature
            [X1,W] = legpts(znum,[mewz(jj)-kfttoptions.nSigmas*sigmaz(jj),mewz(jj)+kfttoptions.nSigmas*sigmaz(jj)]);
            X1 = X1';
        case 'clenshaw-curtis' % Clenshaw-Curtis quadrature
            [X1,W] = fclencurt(znum,mewz(jj)-kfttoptions.nSigmas*sigmaz(jj),mewz(jj)+kfttoptions.nSigmas*sigmaz(jj));
            X1 = fliplr(X1');
            W = fliplr(W');
        case 'gauss-hermite' % Gauss-Hermite quadrature
            if rho(jj) > 0.8
                warning('Model is persistent; even-spaced grid is recommended')
            end
            [X1,W] = GaussHermite(znum);
            X1 = mewz(jj)+sqrt(2)*sigmaz(jj)*X1';
            W = W'./sqrt(pi);
        case 'GMQ' % Gaussian Mixture Quadrature
            if rho(jj) > 0.8
                warning('Model is persistent; even-spaced grid is recommended')
            end
            [X1,W] = GaussianMixtureQuadrature(mixprobs_i(:,jj),mu_i(:,jj),sigma_i(:,jj),znum);
            X1 = X1 + mewz(jj);
    end
    
    z_grid = allcomb2(X1); % Nm*1 matrix of grid points
    z_grid_J(:,jj)=z_grid;
end


%% Step 3: Compute the transition matrices trans(:,:,t) from period (t-1) to period t

nMoments_grid=zeros(znum,J); % Used to record number of moments matched in transition from each point

for jj=1:J
    %% compute conditional moments
    fprintf('discretizeLifeCycleAR1wGM_KFTT: now doing period %i of %i \n',jj,J)
    sigmaC2 = sigma_i(:,jj).^2;
    
    if jj>1
        zlag_grid=z_grid_J(:,jj-1);
    else
        zlag_grid=z_grid_0;
    end
    z_grid=z_grid_J(:,jj)';
    
    TBar=TBar_J(:,jj);
    
    nComp = length(mixprobs_i(:,jj)); % number of mixture components
    temp = zeros(1,1,nComp);
    temp(1,1,:) = sigmaC2;
    gmObj = gmdistribution(mu_i(:,jj),temp,mixprobs_i(:,jj)); % define the Gaussian mixture object
    
    P = NaN(znum,znum); % transition probability matrix
    P1 = NaN(znum,znum); % matrix to store transition probability
    P2 = ones(znum,1); % znum*1 matrix used to construct P
    scalingFactor = max(abs(z_grid));
    kappa = 1e-8;
    
    for z_c = 1:znum % For each value z(jj-1) compute the conditional distribution for z(jj) [the row of the transition matrix]
       
        % First, calculate what Farmer & Toda (2017) call qnn', which are essentially an inital guess for pnn'
        condMean = rho(jj)*zlag_grid(z_c); % z_grid(ii) here is the lag grid point
        xPDF = (z_grid-condMean)';
        switch kfttoptions.method
            case 'gauss-hermite'
                q = W.*(pdf(gmObj,xPDF)./normpdf(xPDF,0,sigma(jj)))';
            case 'GMQ'
                q = W.*(pdf(gmObj,xPDF)./pdf(gmObj,z_grid'))';
            otherwise
                q = W.*(pdf(gmObj,xPDF))';
        end
        
        if any(q < kappa)
            q(q < kappa) = kappa;
        end
        
        if kfttoptions.nMoments == 1 % match only 1 moment
            P1(z_c,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
            nMoments_grid(z_c,jj)=1;
        else % match 2 moments first
            [p,lambda,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2],...
                TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
            if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                % warning('Failed to match first 2 moments. Just matching 1.') % too many warnings in Life-Cycle AR(1), people can just look at heatmap instead
                P1(z_c,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
                nMoments_grid(z_c,jj)=1;
            elseif kfttoptions.nMoments == 2
                P1(z_c,:) = p;
                nMoments_grid(z_c,jj)=2;
            elseif kfttoptions.nMoments == 3
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError) > 1e-5
                    % warning('Failed to match first 3 moments.  Just matching 2.') % too many warnings in Life-Cycle AR(1), people can just look at heatmap instead
                    P1(z_c,:) = p;
                    nMoments_grid(z_c,jj)=2;
                else
                    P1(z_c,:) = pnew;
                    nMoments_grid(z_c,jj)=3;
                end
            else % 4 moments
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
                    ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
                if norm(momentError) > 1e-5
                    %warning('Failed to match first 4 moments.  Just matching 3.')
                    [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                    if norm(momentError) > 1e-5
                        % warning('Failed to match first 3 moments.  Just matching 2.') % too many warnings in Life-Cycle AR(1), people can just look at heatmap instead
                        P1(z_c,:) = p;
                        nMoments_grid(z_c,jj)=2;
                    else
                        % warning('Failed to match first 4 moments.  Just matching 3.') % too many warnings in Life-Cycle AR(1), people can just look at heatmap instead
                        P1(z_c,:) = pnew;
                        nMoments_grid(z_c,jj)=3;
                    end
                else
                    P1(z_c,:) = pnew;
                    nMoments_grid(z_c,jj)=4;
                end
            end
            P(z_c,:) = kron(P1(z_c,:),P2(z_c,:));
        end
        
    end
    pi_z_J(:,:,jj)=P;
    
end

%%
jequaloneDistz=pi_z_J(:,:,1)'*jequalzeroDistz;

%% Change P_J so that P_J(:,:,jj) is the transition matrix from period jj to period jj+1
pi_z_J(:,:,1:end-1)=pi_z_J(:,:,2:end);

%% For jj=J, P_J(:,:,J) is kind of meaningless (there is no period J+1 to transition to). I just fill it in as a uniform distribution
pi_z_J(:,:,J)=ones(znum,znum)/znum;

%% I AM BEING LAZY AND JUST MOVING RESULT TO GPU RATHER THAN CREATING IT THERE IN THE FIRST PLACE
if kfttoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    pi_z_J=gpuArray(pi_z_J);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid; % Heatmap of how many moments where hit by the conditional (difference) distribution
otheroutputs.sigma_z=sigmaz; % Standard deviation of z (for each period)
otheroutputs.mew_z=mewz; % Mean of z (for each period)


end
