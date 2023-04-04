function [z_grid_J, P_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1wGM_KirkbyFull(rho,mixprobs_i,mu_i,sigma_i,znum,J,kirkbyoptions)
% Please cite: Kirkby (working paper)
%
%  'Full' allows the mixture probabilies (mixprobs_i) to depend on both age  j and on the lag value of z
% 
%  Extended-Farmer-Toda method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = rho(j)*z(j-1)+ epsilon(j),   epsilion(j)~iid F(j)
%           where F(j)=sum_{i=1}^nmix mixprobs_i(j,zlag)*N(mu_i(j),sigma_i(j)^2) is a gaussian mixture
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1)) 
%
%  Note: n, the number of normal distributions in the gaussian mixture, cannot depend on j
%
% Inputs:
%   rho 	     - (1-by-J) vector of serial correlation coefficients
%   sigma_i      - (nmix-by-J) vector of standard deviations of innovations
%   mixprobs_i   - function which outputs nmix depending on J and zlag
%   mu_i        - (nmix-by-J) means of the gaussian mixture innovations
%   sigma_i      - (nmix-by-J) standard deviations of the gaussian mixture innovations
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
%        otheroutputs.nMoments_grid  - shows how many moments were  matched from each grid point (for the conditional distribution)
%        otheroutputs.sigma_z     - the standard deviation of z at each age (used to determine grid)
%
% !========================================================================%
% Original paper:
% Kirkby (working paper)

sigma_z = zeros(1,J);
% z_grid_J = zeros(znum,J); 
P_J = zeros(znum,znum,J);

%% Set options
if ~exist('kirkbyoptions','var')
    kirkbyoptions.method='even'; % Informally I have the impression even is more robust
    kirkbyoptions.nMoments=4; % Have used 4 as the point of gaussian mixtures is typically to get higher order moments. 4 covers skewness and kurtosis.
    if rho <= 1-2/(znum-1)  % This is just what Toda used.
        kirkbyoptions.nSigmas = min(sqrt(2*(znum-1)),4);
    else
        kirkbyoptions.nSigmas = min(sqrt(znum-1),4);
    end
    kirkbyoptions.parallel=1+(gpuDeviceCount>0);
    kirkbyoptions.setmixturemutoenforcezeromean=0;
    kirkbyoptions.customGKOSmodel4=0; % Special treatment required for Guvenen, Karahan, Ozkan & Song (2022)
else
    if ~isfield(kirkbyoptions,'method')
        kirkbyoptions.method='even'; % Informally I have the impression even is more robust
    end
    if ~isfield(kirkbyoptions,'nMoments')
        kirkbyoptions.nMoments = 4;  % Have used 4 as the point of gaussian mixtures is typically to get higher order moments. 4 covers skewness and kurtosis.
    end
    if ~isfield(kirkbyoptions,'nSigmas')
        if rho <= 1-2/(znum-1) % This is just what Toda used.
            kirkbyoptions.nSigmas = min(sqrt(2*(znum-1)),4);
        else
            kirkbyoptions.nSigmas = min(sqrt(znum-1),4);
        end
    end
    if ~isfield(kirkbyoptions,'parallel')
        kirkbyoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(kirkbyoptions,'setmixturemutoenforcezeromean')
        kirkbyoptions.setmixturemutoenforcezeromean=0;
    end
    if ~isfield(kirkbyoptions,'customGKOSmodel4')
        kirkbyoptions.customGKOSmodel4=0; % Special treatment required for Guvenen, Karahan, Ozkan & Song (2022)
    end
end

%% some error checking
if znum < 2
    error('The state space has to have dimension znum>1. Exiting.')
end
if J < 2
    error('The time horizon has to have dimension J>1. Exiting.')
end

% if any(mixprobs_i < 0)
%     error('mixture proportions must be positive')
% end
% if ~any(sum(mixprobs_i,1))
%     error('mixture proportions must add up to 1 (for each age)')
% end
    
if any(sigma_i < 0)
    error('standard deviations must be positive')
end

% if size(mixprobs_i,2)~=J
%     error('mixprobs_i must have J columns')
% end
if size(mu_i,2)~=J
    error('mu_i must have J columns')
end
if size(sigma_i,2)~=J
    error('sigma_i must have J columns')
end

nmix=size(sigma_i,1);
if kirkbyoptions.setmixturemutoenforcezeromean==0
    if size(mu_i,1)~=nmix
        error('sigma_i and mu_i must all have same number of rows (nmix)')
    end
end
if ~isa(mixprobs_i,'function_handle')
    if size(mixprobs_i,1)~=nmix
        error('sigma_i and mixprobs_i must all have same number of rows (nmix)')
    end
end

if any(rho >= 1)
    error('autocorrelation coefficient (spectral radius) must be less than one')
end

% Check that Nm is a valid number of grid points
if ~isnumeric(znum) || znum < 3 || rem(znum,1) ~= 0
    error('Nm must be a positive integer greater than 3')
end
% Check that nMoments is a valid number
if ~isnumeric(kirkbyoptions.nMoments) || kirkbyoptions.nMoments < 1 || kirkbyoptions.nMoments > 4 || ~((rem(kirkbyoptions.nMoments,1) == 0) || (kirkbyoptions.nMoments == 1))
    error('kirkbyoptions.nMoments must be either 1, 2, 3, 4')
end

%% Step E1: Challenge is that sigma depends on zlag. But without grid we don't know zlag and to create grid we need sigma.
% I just assume that sigma is not too sensitive to zlag, and use z=0 to
% determine sigma and set grid based on this.
% (A better approach might be to then do a second step where I use this
% grid for zlag to recompute sigma at different points on grid, and then
% based on these I can recreate grid; obviously you could iterate this multiple times).

% Check mixprob_i is function, if yes then evaluate it using zlag=0
if isa(mixprobs_i, 'function_handle')
    mixprobs_iz=zeros(nmix,J);
    for jj=1:J
        mixprobs_iz(:,jj)=mixprobs_i(jj,0); % 0 is value of zlag
    end
else
    mixprobs_iz=mixprobs_i;
end

fprintf('Based on zlag=0 we get')
mixprobs_iz

if any(mixprobs_iz < 0)
    error('mixture proportions must be positive')
end
if ~any(sum(mixprobs_iz,1))
    error('mixture proportions must add up to 1 (for each age)')
end

if kirkbyoptions.setmixturemutoenforcezeromean==1
    mu_i=[mu_i;zeros(1,J)]; % Need to fill in the last of the mu_i to get mu=0
    for jj=1:J
        mu_i(end,jj)=-(sum(mu_i(1:end-1,jj).*mixprobs_iz(1:end-1,jj)))/mixprobs_iz(end,jj); % Simple rearrangement of mu_i(:,jj).*mixprob_iz(:,jj)=0, which is the requirement that mean of gaussian-mixture innovations=0
    end
end
% Note: when using kirkbyoptions.setmixtureprobtoenforcezeromean it must be
% the 'last' mu_i that is missing and which will be set to enforce zero
% mean of the gaussian-mixture innovations.

if kirkbyoptions.customGKOSmodel4==1 % assume: kirkbyoptions.setmixturemutoenforcezeromean==1
    % GKOS (2022) use 'special treatment' for their Model 4.    
    % Only works with two gaussian distributions in the mixture
    % I emailed Serdar Ozkan because using the setup as I understood
    % the paper I was getting values of mu_{eta,2} like 1913 (which was
    % problematic as it implied enormous standard errors).
    % Serdar Ozkan emailed me the following explanation: "I first shift the
    % distributions of mixtures by -p_z*mu_{eta,1}. Then the mean for the
    % first mixture is simply p_z*mu_{eta,2}. Tus the means of both
    % mixtures are reasonable numbers. And the overall mean is zero"
    mu_i_original=mu_i(1,:); % keep so can be reused later
    mu_i=zeros(2,J);
    mu_i=[mu_i_original-mixprobs_iz(1,:).*mu_i_original;-mixprobs_iz(1,:).*mu_i_original]; % Serdar Ozkan sent me (Robert Kirkby) this equation via email.
end

%% Step 1: compute the conditional moments (will need the standard deviations to create grid)
sigma=zeros(1,J); % standard deviation of innovations
TBar_J=zeros(4,J); % The 4 conditional moments for each period
for jj=1:J
    %% compute conditional moments
    sigmaC2 = sigma_i(:,jj).^2;
    T1 = mixprobs_iz(:,jj)'*mu_i(:,jj); % mean
    T2 = mixprobs_iz(:,jj)'*(mu_i(:,jj).^2+sigmaC2); % uncentered second moment
    T3 = mixprobs_iz(:,jj)'*(mu_i(:,jj).^3+3*mu_i(:,jj).*sigmaC2); % uncentered third moment
    T4 = mixprobs_iz(:,jj)'*(mu_i(:,jj).^4+6*(mu_i(:,jj).^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment
    
    TBar_J(:,jj) = [T1 T2 T3 T4]';
    
    sigma(jj) = sqrt(T2-T1^2); % conditional standard deviation    
end

%% Step 2: construct the state space z_grid_J(j) in each period j.
% Evenly-spaced N-state space over [-kirkbyoptions.nSigmas*sigma_y(t),kirkbyoptions.nSigmas*sigma_y(t)].

if isfield(kirkbyoptions,'initialj0sigma_z')
    disp('Here1')
	[z_grid_0,P_0] = discretizeAR1_FarmerToda(0,0,kirkbyoptions.initialj0sigma_z,znum);
    jequalzeroDistz=P_0(1,:)'; % iid, so first row is the dist
else
    disp('Here2')
    z_grid_0=zeros(znum,1);
    jequalzeroDistz=[1;zeros(znum-1,1)]; % Is irrelevant where we put the mass
end
clear P_0

% 1.a Compute unconditional standard deviation of y(t)
if isfield(kirkbyoptions,'initialj0sigma_z')
    sigma_z(1)=sqrt(rho(1)^2*kirkbyoptions.initialj0sigma_z^2+sigma(1)^2);
else
    sigma_z(1) = sigma(1);
end
for jj = 2:J
    sigma_z(jj) = sqrt(rho(jj)^2*sigma_z(jj-1)^2+sigma(jj)^2);
end

% 1.b Construct evenly-spaced state space
h = 2*kirkbyoptions.nSigmas*sigma_z/(znum-1); % grid step
% Fella, Gallipoli & Pan (2019) have kirkbyoptions.nSigmas=sqrt(znum-1)
% I follow Farmer-Toda (2017) who use the same for larger rho, but modify for smaller rho
z_grid_J = repmat(h,znum,1);
z_grid_J(1,:)=-kirkbyoptions.nSigmas * sigma_z;
z_grid_J = cumsum(z_grid_J,1);
% Note: this is all based on zero mean innovations

% Using an evenly spaced grid means we use
W=ones(1,znum);

% NOTE TO SELF: I SHOULD CHANGE THIS SO GRID IS AN OPTION

%% Step E2: Now recompute the mixprob_i to depend on zlag (when appropriate)
% Check mixprobs_i is function, if yes then evaluate it using zlag=0
if isa(mixprobs_i,'function_handle')
    mixprobs_iz=zeros(nmix,J,znum);
    for z_c=1:znum
        if jj==1
            zlag_grid=0;
        else
            zlag_grid=z_grid_J(z_c,jj-1);
        end
        for jj=1:J
            mixprobs_iz(:,jj,z_c)=mixprobs_i(jj,zlag_grid);
        end
    end
else
    mixprobs_iz=repmat(mixprobs_i,1,1,znum);
end

% fprintf('Based on z grid we get')
% mixprobs_iz

% If mixprob_iz depends on z, then it is possible that mu_i does too.
if kirkbyoptions.setmixturemutoenforcezeromean==1
    mu_iz=zeros(nmix,J,znum);
    for z_c=1:znum
        for jj=1:J
            mu_i(end,jj)=-(mu_i(1:end-1,jj).*mixprobs_iz(1:end-1,jj,z_c))/mixprobs_iz(end,jj,z_c); % Simple rearrangement of mu_i(:,jj).*mixprob_iz(:,jj)=0, which is the requirement that mean of gaussian-mixture innovations=0
        end
        mu_iz(:,:,znum)=mu_i;
    end
else
    mu_iz=repmat(mu_i,1,1,znum);
end
% Note: when using kirkbyoptions.setmixtureprobtoenforcezeromean it must be
% the 'last' mu_i that is missing and which will be set to enforce zero
% mean of the gaussian-mixture innovations.

if kirkbyoptions.customGKOSmodel4==1 % assume: kirkbyoptions.setmixturemutoenforcezeromean==1
    % GKOS (2022) use 'special treatment' of their Model 4.    
    % Only works with two gaussian distributions in the mixture

    mixprobs_iz(1,:,:)=max(0.0001,mixprobs_iz(1,:,:)); % Some mixture probabilities were zero which was problematic
    mixprobs_iz(2,:,:)=1-mixprobs_iz(1,:,:);
    mixprobs_iz(1,:,:)=min(0.9999,mixprobs_iz(1,:,:)); % Some mixture probabilities were one which was problematic
    mixprobs_iz(2,:,:)=1-mixprobs_iz(1,:,:);
    
    
    mu_iz=zeros(nmix,J,znum);
    for z_c=1:znum
        mu_iz(1,:,z_c)=mu_i_original;
        for jj=1:J
            mu_iz(:,jj,z_c)=[mu_i_original(jj)-mixprobs_iz(1,jj,z_c)*mu_i_original(jj);-mixprobs_iz(1,jj,z_c)*mu_i_original(jj)]; % Serdar Ozkan sent me (Robert Kirkby) this equation via email.
        end
    end
end

%% Step E*: Could recompute z_grid and then mixprob_iz and mu_iz
% For now I don't bother.

%% Now, compute all the moments that depend on z
sigma_z=zeros(1,J); % standard deviation of innovations
TBar_Jz=zeros(4,J); % The 4 conditional moments for each period
for jj=1:J
    for z_c=1:znum
        %% compute conditional moments
        sigmaC2 = sigma_i(:,jj).^2; % Note: do not allow for sigma_i to depend on z
        T1 = mixprobs_iz(:,jj)'*mu_iz(:,jj); % mean
        T2 = mixprobs_iz(:,jj)'*(mu_iz(:,jj).^2+sigmaC2); % uncentered second moment
        T3 = mixprobs_iz(:,jj)'*(mu_iz(:,jj).^3+3*mu_iz(:,jj).*sigmaC2); % uncentered third moment
        T4 = mixprobs_iz(:,jj)'*(mu_iz(:,jj).^4+6*(mu_iz(:,jj).^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment
        
        TBar_Jz(:,jj,z_c) = [T1 T2 T3 T4]';
        
        sigma_z(jj,z_c) = sqrt(T2-T1^2); % conditional standard deviation
    end
end

%% Step 3: Compute the transition matrices trans(:,:,t) from period (t-1) to period t

nMoments_grid=zeros(znum,J);

for jj=1:J
    %% compute conditional moments
    sigmaC2 = sigma_i(:,jj).^2; % Note: Do not allow sigma_i to depend on z
    
    if jj>1
        zlag_grid=z_grid_J(:,jj-1);
    else
        zlag_grid=z_grid_0;
    end
    z_grid=z_grid_J(:,jj)';
                    
    P = NaN(znum,znum); % transition probability matrix
    P1 = NaN(znum,znum); % matrix to store transition probability
    P2 = ones(znum,1); % znum*1 matrix used to construct P
    scalingFactor = max(abs(z_grid_J(:,jj)));
    kappa = 1e-8;
    
    for z_c = 1:znum % For each value z(jj-1) compute the conditional distribution for z(jj) [the row of the transition matrix]
        TBar=TBar_Jz(:,jj,z_c);
        
        nComp = length(mixprobs_iz(:,jj,z_c)); % number of mixture components
        temp = zeros(1,1,nComp);
        temp(1,1,:) = sigmaC2;

        gmObj = gmdistribution(mu_iz(:,jj,z_c),temp,mixprobs_iz(:,jj,z_c)); % define the Gaussian mixture object
        
        % First, calculate what Farmer & Toda (2017) call qnn', which are essentially an inital guess for pnn'
        condMean = rho(jj)*zlag_grid(z_c); % z_grid(ii) here is the lag grid point
        xPDF = (z_grid-condMean)';
        switch kirkbyoptions.method
            case 'gauss-hermite'
                q = W.*(pdf(gmObj,xPDF)./normpdf(xPDF,0,sigma_z(z_c)))';
            case 'GMQ'
                q = W.*(pdf(gmObj,xPDF)./pdf(gmObj,z_grid'))';
            otherwise
                q = W.*(pdf(gmObj,xPDF))';
        end
        
        if any(q < kappa)
            q(q < kappa) = kappa;
        end
        
        if kirkbyoptions.nMoments == 1 % match only 1 moment
            P1(z_c,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
            nMoments_grid(z_c,jj)=1;
        else % match 2 moments first
            [p,lambda,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2],...
                TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
            if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                warning('Failed to match first 2 moments. Just matching 1.')
                P1(z_c,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
                nMoments_grid(z_c,jj)=1;
            elseif kirkbyoptions.nMoments == 2
                P1(z_c,:) = p;
                nMoments_grid(z_c,jj)=2;
            elseif kirkbyoptions.nMoments == 3
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError) > 1e-5
                    warning('Failed to match first 3 moments.  Just matching 2.')
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
                        warning('Failed to match first 3 moments.  Just matching 2.')
                        P1(z_c,:) = p;
                        nMoments_grid(z_c,jj)=2;
                    else
                        warning('Failed to match first 4 moments.  Just matching 3.')
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
    P_J(:,:,jj)=P;
    
end


%%
jequaloneDistz=P_J(:,:,1)'*jequalzeroDistz;

%% Change P_J so that P_J(:,:,jj) is the transition matrix from period jj to period jj+1
P_J(:,:,1:end-1)=P_J(:,:,2:end);

%% For jj=J, P_J(:,:,J) is kind of meaningless (there is no period jj+1 to transition to). I just fill it in as a uniform distribution
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
