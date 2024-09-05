function [z_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_KFTT(mew,rho,sigma,znum,J,kfttoptions)
% Please cite: Kirkby (working paper)
%
% KFTT discretization method for a 'life-cycle non-stationary AR(1) process with
%    gaussian innovations'. 
% This is an extension of the Farmer-Toda method to 'age-dependent parameters' 
%    (apply concepts of Farmer & Toda (2017) in combination with extension of Fella, Gallipoli & Pan (2019))
% Which in turn is an extension of the Tanaka & Toda (2013).
% Hence: KFTT=Kirkby-Farmer-Tanaka-Toda
% 
%  KFTT method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = mew(j)+rho(j)*z(j-1)+ epsilon(j),   epsilon(j)~iid  N(0,sigma(j))
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1))
%
% Inputs:
%   mew          - (1-by-J) vector of 'drifts'
%   rho 	     - (1-by-J) vector of serial correlation coefficients
%   sigma        - (1-by-J) standard deviations of the gaussian innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
% Optional inputs (kfttoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre', 'clenshaw-curtis','gauss-hermite')
%   nMoments       - Number of conditional moments to match (default=4)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigmaz (default depends on znum)
%   parallel:          - set equal to 2 to use GPU, 0 to use CPU
%        You can control the initial period with the following:
%        By default, assume z0=0
%   initialj0sigmaz:  - Set period 0 to be a N(z0, initialj0sigma_z^2) using initialj0sigma_z
%   initialj0mewz:    - Give period 0 a mean of z0=initialj0mew_z (as a point, or as mean of normal dist is you are also setting initialj0sigma_z
%        Or you set the period 1 (instead of period 0) using
%   initialj1mewz:    - Set period 1 to be mean of initialj1mew_z
%   initialj1sigmaz:  - Set period 1 to be a N(z0, initialj1sigma_z^2) using initialj1sigma_z
%        Note, for both period 0 and period 1, you can set one or both of
%        mean and standard deviation (the other is interpreted as zero
%        valued if not specified).
% Output: 
%   z_grid       - an znum-by-J matrix, each column stores the Markov state space for period j
%   pi_z_J       - znum-by-znum-by-J matrix of J (znum-by-znum) transition matrices. 
%                  Transition probabilities are arranged by row.
%   jequaloneDistz - initial distribution of shocks for j=1
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - shows how many moments were matched from each grid point (for the conditional distribution)
%        otheroutputs.sigma_z        - the standard deviation of z at each age (used to determine grid)
%
% !========================================================================%
% Original paper:
% Kirkby (working paper)

mewz=zeros(1,J); % period j mean of z
sigmaz = zeros(1,J); % period j std dev of z
% z_grid_J = zeros(znum,J); % period j grid on z
pi_z_J = zeros(znum,znum,J); % period j transition probabilities for z


%% Set options
if ~exist('kfttoptions','var')
    kfttoptions.method='even'; % Informally I have the impression even is more robust
    kfttoptions.nMoments=4; % Innovations are normal, but probably still nice to hit these higher moments.
    if rho <= 1-2/(znum-1)  % This is just what Toda used.
        kfttoptions.nSigmas = min(sqrt(2*(znum-1)),3); % Maximum of +-3 standard deviation
    else
        kfttoptions.nSigmas = min(sqrt(znum-1),3); % Maximum of +-3 standard deviations
    end
    kfttoptions.parallel=1+(gpuDeviceCount>0);
    kfttoptions.setmixturemutoenforcezeromean=0;
else
    if ~isfield(kfttoptions,'method')
        kfttoptions.method='even'; % Informally I have the impression even is more robust
    end
    if ~isfield(kfttoptions,'nMoments')
        kfttoptions.nMoments = 4;  % Innovations are normal, but probably still nice to hit these higher moments.
    end
    if ~isfield(kfttoptions,'nSigmas')
        if rho <= 1-2/(znum-1) % This is just what Toda used.
            kfttoptions.nSigmas = min(sqrt(2*(znum-1)),3); % Maximum of +-3 standard deviation
        else
            kfttoptions.nSigmas = min(sqrt(znum-1),3); % Maximum of +-3 standard deviation
        end
    end
    if ~isfield(kfttoptions,'parallel')
        kfttoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(kfttoptions,'setmixturemutoenforcezeromean')
        kfttoptions.setmixturemutoenforcezeromean=0;
    end
end
% Note: the choice of setting nSigmas to sqrt(znum-1) is based on
% asymptotic theory in Corrallary 3.5(ii) of Farmer & Toda (2017)

%% some error checking
if znum < 2
    error('The state space has to have dimension znum>1.')
end
if J < 2
    error('The time horizon has to have dimension J>1.')
end

if size(mew,2)~=J
    if isscalar(mew)
        mew=mew*ones(1,J); % assume that scalars are simply age-independent parameters
        % No warning, as good odds this is just a zero
    else
        error('mew_j must have J columns')
    end
end
if any(sigma < 0)
    error('standard deviations must be positive')
end
if size(sigma,2)~=J
    if isscalar(sigma)
        warning('Input for sigma (std dev) was scalar. Interpreting as a constant vector.')
        sigma=sigma*ones(1,J); % assume that scalars are simply age-independent parameters
    else
        error('sigma_j must have J columns')
    end
end
if size(rho,2)~=J
    if isscalar(rho)
        warning('Input for autocorrelation was scalar. Interpreting as a constant vector.')
        rho=rho*ones(1,J); % assume that scalars are simply age-independent parameters
    else
        error('rho_j must have J columns')
    end
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

% For convenience, make kfttoptions.nSigmas an age-dependent vector
if isscalar(kfttoptions.nSigmas)
    kfttoptions.nSigmas=kfttoptions.nSigmas*ones(J,1);
end

% % Everything has to be on cpu otherwise fminunc throws an error
% if kfttoptions.parallel==2
%     mew=gather(mew);
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
% Evenly-spaced N-state space over [-kfttoptions.nSigmas*sigmaz(j),kfttoptions.nSigmas*sigmaz(j)].

% Note: I set up the period 0, but this won't end up used if you have used options to set period 1.

% By default I assume z0=0
z0=0;
% You can change the mean of z0 using
if isfield(kfttoptions,'initialj0mewz')
    z0=kfttoptions.initialj0mewz;
end
% You can add variance to z0 as a N(z0,initialj0sigmaz) using
if isfield(kfttoptions,'initialj0sigmaz')
	[z_grid_0,pi_z_0] = discretizeAR1_FarmerToda(z0,0,kfttoptions.initialj0sigmaz,znum);
    jequalzeroDistz=pi_z_0(1,:)'; % iid, so first row is the dist
else
    z_grid_0=z0*ones(znum,1);
    jequalzeroDistz=ones(znum,1)/znum; % Is irrelevant where we put the mass (because z_grid_0 is all just same value anyway)
end
clear pi_z_0

% If you are not setting period 1, then period 1 follows from this
% if ~isfield(kfttoptions,'initialj1mewz') && ~ isfield(kfttoptions,'initialj1sigmaz')
if isfield(kfttoptions,'initialj0sigmaz')
    sigmaz(1) = sqrt(rho(1)^2*kfttoptions.initialj0sigma_z^2+sigma(1)^2);
else
    sigmaz(1) = sigma(1);
end
mewz(1)=mew(1)+rho(1)*z0;
% If you have set period 1, then overwrite some of this
if isfield(kfttoptions,'initialj1mewz') && isfield(kfttoptions,'initialj1sigmaz')
    mewz(1)=kfttoptions.initialj1mewz;
    sigmaz(1)=kfttoptions.initialj1sigmaz;
elseif isfield(kfttoptions,'initialj1mewz')
    mewz(1)=kfttoptions.initialj1mewz;
    sigmaz(1)=0;
elseif isfield(kfttoptions,'initialj1sigmaz')
    mewz(1)=0;
    sigmaz(1)=kfttoptions.initialj1sigmaz;
end

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
            X1 = linspace(mewz(jj)-kfttoptions.nSigmas(jj)*sigmaz(jj),mewz(jj)+kfttoptions.nSigmas(jj)*sigmaz(jj),znum);
            W = ones(1,znum);
        case 'gauss-legendre' % Gauss-Legendre quadrature
            [X1,W] = legpts(znum,[mewz(jj)-kfttoptions.nSigmas(jj)*sigmaz(jj),mewz(jj)+kfttoptions.nSigmas(jj)*sigmaz(jj)]);
            X1 = X1';
        case 'clenshaw-curtis' % Clenshaw-Curtis quadrature
            [X1,W] = fclencurt(znum,mewz(jj)-kfttoptions.nSigmas(jj)*sigmaz(jj),mewz(jj)+kfttoptions.nSigmas(jj)*sigmaz(jj));
            X1 = fliplr(X1');
            W = fliplr(W');
        case 'gauss-hermite' % Gauss-Hermite quadrature
            if rho(jj) > 0.8
                warning('Model is persistent; even-spaced grid is recommended')
            end
            [X1,W] = GaussHermite(znum);
            X1 = mewz(jj)+sqrt(2)*sigmaz(jj)*X1';
            W = W'./sqrt(pi);
    end
    
    z_grid = allcomb2(X1); % Nm*1 matrix of grid points
    z_grid_J(:,jj)=z_grid;
end


%% Step 3: Compute the transition matrices trans(:,:,t) from period (t-1) to period t

nMoments_grid=zeros(znum,J); % Used to record number of moments matched in transition from each point

kappa = 1e-8;

for jj=1:J
    
    if jj>1
        zlag_grid=z_grid_J(:,jj-1);
    else
        if ~isfield(kfttoptions,'initialj1mewz') && ~ isfield(kfttoptions,'initialj1sigmaz')
            zlag_grid=z_grid_0; % Need to get pi_z_J(:,:,1) so we can compute jequaloneDistz
        else % Have set period 1
            continue % We already have jequaloneDistz
        end
    end
    z_grid=z_grid_J(:,jj)';
    
    TBar=TBar_J(:,jj);

    P = nan(znum,znum); % The transition matrix for age jj
    scalingFactor = max(abs(z_grid));
    
    for z_c = 1:znum
        
        condMean = mew(jj)+rho(jj)*zlag_grid(z_c); % conditional mean
        if strcmp(kfttoptions.method,'gauss-hermite')  % define prior probabilities
            q = W;
        else
            q = W.*normpdf(z_grid,condMean,sigma(jj));
        end
        
        if any(q < kappa)
            q(q < kappa) = kappa; % replace by small number for numerical stability
        end

        if kfttoptions.nMoments == 1 % match only 1 moment
            P(z_c,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
            nMoments_grid(z_c,jj)=1;
        else % match 2 moments first
            [p,lambda,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2],...
                TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
            if norm(momentError) > 1e-5 % if 2 moments fail, then just match 1 moment
                % warning('Failed to match first 2 moments. Just matching 1.')
                P(z_c,:) = discreteApproximation(z_grid,@(x)(x-condMean)/scalingFactor,0,q,0);
                nMoments_grid(z_c,jj)=1;
            elseif kfttoptions.nMoments == 2
                P(z_c,:) = p;
                nMoments_grid(z_c,jj)=2;
            elseif kfttoptions.nMoments == 3 % 3 moments
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError) > 1e-5
                    % warning('Failed to match first 3 moments.  Just matching 2.')
                    P(z_c,:) = p;
                    nMoments_grid(z_c,jj)=2;
                else
                    P(z_c,:) = pnew;
                    nMoments_grid(z_c,jj)=3;
                end
            elseif kfttoptions.nMoments == 4 % 4 moments
                [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2; ((x-condMean)./scalingFactor).^3;...
                    ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
                if norm(momentError) > 1e-5
                    %warning('Failed to match first 4 moments.  Just matching 3.')
                    [pnew,~,momentError] = discreteApproximation(z_grid,@(x) [(x-condMean)./scalingFactor;...
                        ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                        TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                    if norm(momentError) > 1e-5
                        % warning('Failed to match first 3 moments.  Just matching 2.')
                        P(z_c,:) = p;
                        nMoments_grid(z_c,jj)=2;
                    else
                        P(z_c,:) = pnew;
                        % warning('Failed to match first 4 moments.  Just matching 3.')
                        nMoments_grid(z_c,jj)=3;
                    end
                else
                    P(z_c,:) = pnew;
                    nMoments_grid(z_c,jj)=4;
                end
            end
        end
    end
    pi_z_J(:,:,jj)=P;
    
end

% Instead of warning about each time it fails to match, just give a summary of how many matches succeed
hits=zeros(1,4);
hits(1)=sum(sum(nMoments_grid==1));
hits(2)=sum(sum(nMoments_grid==2));
hits(3)=sum(sum(nMoments_grid==3));
hits(4)=sum(sum(nMoments_grid==4));
hits=hits./sum(hits);
fprintf('discretizeLifeCycleAR1_Kirkby: 1 moment in %1.2f cases, 2 moments in %1.2f cases, 3 moments in %1.2f cases, 4 moments in %1.2f cases (target was %i moments) \n', hits(1), hits(2), hits(3), hits(4), kfttoptions.nMoments)
if hits(4)<0.8 && kfttoptions.nMoments==4
    warning('discretizeLifeCycleAR1_Kirkby: failed to hit four moments in more than 20% of conditional distributions')
end

%%
if isfield(kfttoptions,'initialj1mewz') || isfield(kfttoptions,'initialj1sigmaz')
    % If period 1 was set, we need to get the jequaloneDistz
    if sigmaz(1)>0
        [~,pi_z_1] = discretizeAR1_FarmerToda(mewz(1),0,sigmaz(1),znum);
        jequaloneDistz=pi_z_1(1,:)'; % iid, so first row is the dist
    else
        % All grid points are same, so just pick an arbitrary one
        jequaloneDistz=zeros(znum,1);
        jequaloneDistz(ceil(znum/2))=1; % put the mass on the median point (irrelevant as all points will anyway be same value)
    end
else
    % Otherwise, we already have the jequalzeroDistz, and just use this
    jequaloneDistz=pi_z_J(:,:,1)'*jequalzeroDistz;
    otheroutputs.jequalzeroDistz=jequalzeroDistz; % store this so that user can see it to check it looks like they intend (a way to double-check the input options)
end

%% Change pi_z_J so that pi_z_J(:,:,jj) is the transition matrix from period jj to period jj+1
pi_z_J(:,:,1:end-1)=pi_z_J(:,:,2:end);

%% For jj=J, pi_z_J(:,:,J) is kind of meaningless (there is no period J+1 to transition to). I just fill it in as a uniform distribution
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



