function [z_grid_J, pi_z_J, jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1wGM_KFTT(mew,rho,mixprobs_i,mu_i,sigma_i,znum,J,kfttoptions)
% Please cite: Kirkby (2025) - Discretizing Earnings Dynamics: Implications of Gaussian-Mixture Shocks for Life-Cycle Models
%
% KFTT discretization method for a 'life-cycle non-stationary AR(1) process with
%    gaussian-mixture innovations'.
% This is an extension of the Farmer-Toda method to 'age-dependent parameters'
%    (essentially combine Farmer & Toda (2017) with Fella, Gallipoli & Pan (2019)
% Which in turn is an extension of Tanaka & Toda (2013).
% Hence KFTT=Kirkby-Farmer-Tanaka-Toda
%
%  KFTT method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = mew(j)+rho(j)*z(j-1)+ epsilon(j),   epsilon(j)~iid F(j)
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
%                   Can be a scalar, or a vector with one element per age.
%                   Note: nSigmas is only used by the 'even', 'gauss-legendre' and 'clenshaw-curtis'
%                   methods. The 'gauss-hermite' and 'GMQ' methods determine their own grids and
%                   ignore it.
%        You can control the initial period with the following:
%        By default, assume z0=0
%   initialj0mewz:    - Give period 0 a mean of z0=initialj0mewz (as a point, or as the mean of a normal dist if you are also setting initialj0sigmaz)
%   initialj0sigmaz:  - Set period 0 to be a N(z0, initialj0sigmaz^2)
%        Or you set the period 1 (instead of period 0) using
%   initialj1mewz:    - Set period 1 to have mean initialj1mewz
%   initialj1sigmaz:  - Set period 1 to be a N(initialj1mewz, initialj1sigmaz^2)
%        Note, for both period 0 and period 1, you can set one or both of
%        mean and standard deviation (the other is interpreted as zero valued if not specified).
%        Period 1 can also be set as a gaussian mixture, rather than as a normal
%        distribution, by giving initialj1sigmaz as a VECTOR:
%   initialj1sigmaz:    - (nmix1-by-1) standard deviations of the components
%   initialj1mixprobs:  - (nmix1-by-1) mixture probabilities of the components (required whenever
%                         initialj1sigmaz is a vector; must sum to one)
%   initialj1mewz:      - (nmix1-by-1) means of the components (a scalar is expanded to give every
%                         component the same mean; if omitted the components are mean zero)
%        Note, nmix1 need not equal the nmix of the innovations.
% Output:
%   z_grid_J       - an znum-by-J matrix, each column stores the Markov state space for period j
%   pi_z_J         - znum-by-znum-by-(J-1) matrix of J-1 (znum-by-znum) transition matrices.
%                       Transition probabilities are arranged by row.
%                       pi_z_J(:,:,j) is the transition matrix from age j to age j+1. There are
%                       only J-1 of them, as there is no period J+1 to transition to.
%   jequaloneDistz - initial distribution of shocks for j=1
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - shows how many moments were  matched from each grid point (for the conditional distribution)
%        otheroutputs.sigma_z     - the standard deviation of z at each age (used to determine grid)
%        otheroutputs.mew_z       - the mean of z at each age (used to determine grid)
%        otheroutputs.jequalzeroDistz - the distribution of z in period 0 (a way to double-check the initialj0 options did what you intended)
%        otheroutputs.z_grid_0    - the period 0 grid that jequalzeroDistz sits on
%
% !========================================================================%
% Original paper:
% Kirkby (2025) - Discretizing Earnings Dynamics: Implications of Gaussian-Mixture Shocks for Life-Cycle Models

mewz=zeros(1,J); % period j mean of z
sigmaz = zeros(1,J);
% z_grid_J = zeros(znum,J);
pi_z_J = zeros(znum,znum,J-1); % pi_z_J(:,:,jj) is the transition from period jj to period jj+1, so there are only J-1 of them

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
    kfttoptions.verbose=1;
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
    if ~isfield(kfttoptions,'verbose')
        kfttoptions.verbose=1;
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

% Make sure method is set appropriately
% Without this the switch on method below simply falls through, leaving the grid variable
% unassigned, and the failure surfaces a line later as an undefined-variable error naming a
% variable the caller has never heard of. Checking here names the option instead.
if ~strcmp(kfttoptions.method,'even') && ~strcmp(kfttoptions.method,'gauss-legendre') && ~strcmp(kfttoptions.method,'clenshaw-curtis') && ~strcmp(kfttoptions.method,'gauss-hermite') && ~strcmp(kfttoptions.method,'GMQ')
    error('kfttoptions.method must be one of even, gauss-legendre, clenshaw-curtis, gauss-hermite, or GMQ')
end

% For convenience, make kfttoptions.nSigmas an age-dependent vector
if isscalar(kfttoptions.nSigmas)
    kfttoptions.nSigmas=kfttoptions.nSigmas*ones(J,1);
end
if length(kfttoptions.nSigmas)~=J
    error('kfttoptions.nSigmas must be a scalar, or a vector with one element per age (J)')
end

if any(kfttoptions.nSigmas<1.2) % note: nSigmas is an age-dependent vector by this point, so 'any' (otherwise 'if' would require every age to be below 1.2)
    warning('Trying to hit the 2nd moment with kfttoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for kfttoptions.nSigmas<1.2).')
end

% Everything has to be on cpu otherwise fminunc throws an error
rho=gather(rho);
mixprobs_i=gather(mixprobs_i);
mu_i=gather(mu_i);
sigma_i=gather(sigma_i);
znum=gather(znum);
J=gather(J);


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
if isfield(kfttoptions,'initialj0sigmaz')
    farmertodaoptions.nSigmas=kfttoptions.nSigmas(1); % nSigmas was made an age-dependent vector above; discretizeAR1_FarmerToda() needs a scalar, and period 1 is the relevant age here
    farmertodaoptions.method=kfttoptions.method;
    if strcmp(kfttoptions.method,'GMQ')
        % GMQ (gaussian mixture quadrature) is a method of this command but not of
        % discretizeAR1_FarmerToda(), which has no mixture to build a quadrature on. Period 0 here
        % is a plain normal, N(z0,initialj0sigmaz^2), so there is no mixture to tune a grid to and
        % 'even' is the right translation. Without this, method='GMQ' together with initialj0sigmaz
        % - two documented options of this command, both legal - crashed inside
        % discretizeAR1_FarmerToda() on an undefined z_grid.
        farmertodaoptions.method='even';
    end
    farmertodaoptions.parallel=1; % need to get solution on cpu, otherwise causes errors later
	[z_grid_0,pi_z_0] = discretizeAR1_FarmerToda(z0,0,kfttoptions.initialj0sigmaz,znum,farmertodaoptions);
    jequalzeroDistz=pi_z_0(1,:)'; % iid, so first row is the dist
    clear pi_z_0
else
    z_grid_0=z0*ones(znum,1); % period 0 is a point mass at z0
    jequalzeroDistz=[1;zeros(znum-1,1)]; % all the grid points are z0, so it does not matter which one carries the mass
end
% Note: z_grid_0 and jequalzeroDistz are only reported (as otheroutputs); nothing is computed from
% them, as the period 1 distribution is built directly further below.

if isfield(kfttoptions,'initialj0sigmaz')
    sigmaz(1) = sqrt(rho(1)^2*kfttoptions.initialj0sigmaz^2+sigma(1)^2);
else
    sigmaz(1) = sigma(1);
end
mewz(1)=mew(1)+rho(1)*z0;

% If you have set period 1, then overwrite some of this.
% A scalar kfttoptions.initialj1sigmaz means period 1 is a normal distribution;
% a vector means period 1 is a gaussian mixture (and kfttoptions.initialj1mixprobs is then required).
initialj1mixture=0;
if isfield(kfttoptions,'initialj1sigmaz')
    if length(kfttoptions.initialj1sigmaz)>1
        initialj1mixture=1;
    end
end
if isfield(kfttoptions,'initialj1mewz')
    if length(kfttoptions.initialj1mewz)>1 && initialj1mixture==0
        error('discretizeLifeCycleAR1wGM_KFTT: kfttoptions.initialj1mewz is a vector but kfttoptions.initialj1sigmaz is not. To set period 1 as a gaussian mixture you must give initialj1sigmaz as a vector of the same length (and initialj1mixprobs)')
    end
end

if initialj1mixture==1
    % Period 1 is a gaussian mixture
    initialj1sigma_i=kfttoptions.initialj1sigmaz(:);
    nmix1=length(initialj1sigma_i); % number of components in the period 1 gaussian mixture (need not equal nmix of the innovations)
    if ~isfield(kfttoptions,'initialj1mixprobs')
        error('discretizeLifeCycleAR1wGM_KFTT: kfttoptions.initialj1sigmaz is a vector, so period 1 is being set as a gaussian mixture, and you must therefore also set kfttoptions.initialj1mixprobs (the mixture probabilities)')
    end
    initialj1mixprobs=kfttoptions.initialj1mixprobs(:);
    if length(initialj1mixprobs)~=nmix1
        error('discretizeLifeCycleAR1wGM_KFTT: kfttoptions.initialj1mixprobs must be the same length as kfttoptions.initialj1sigmaz')
    end
    if abs(sum(initialj1mixprobs)-1)>10^(-12)
        error('discretizeLifeCycleAR1wGM_KFTT: kfttoptions.initialj1mixprobs must sum to one')
    end
    if isfield(kfttoptions,'initialj1mewz')
        initialj1mu_i=kfttoptions.initialj1mewz(:);
        if length(initialj1mu_i)==1
            initialj1mu_i=initialj1mu_i*ones(nmix1,1); % scalar means every component has this same mean
        elseif length(initialj1mu_i)~=nmix1
            error('discretizeLifeCycleAR1wGM_KFTT: kfttoptions.initialj1mewz must be either a scalar or the same length as kfttoptions.initialj1sigmaz')
        end
    else
        initialj1mu_i=zeros(nmix1,1); % components are mean zero if you did not say otherwise
    end
    mewz(1)=sum(initialj1mixprobs.*initialj1mu_i); % mean of the gaussian mixture
    sigmaz(1)=sqrt(sum(initialj1mixprobs.*(initialj1sigma_i.^2+initialj1mu_i.^2))-mewz(1)^2); % standard deviation of the gaussian mixture
elseif isfield(kfttoptions,'initialj1mewz') && isfield(kfttoptions,'initialj1sigmaz')
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
                if kfttoptions.verbose==1 && jj==1 % once per call, not once per age
                    warning('Model is persistent; even-spaced grid is recommended')
                end
            end
            [X1,W] = GaussHermite(znum);
            X1 = mewz(jj)+sqrt(2)*sigmaz(jj)*X1';
            W = W'./sqrt(pi);
        case 'GMQ' % Gaussian Mixture Quadrature
            if rho(jj) > 0.8
                if kfttoptions.verbose==1 && jj==1 % once per call, not once per age
                    warning('Model is persistent; even-spaced grid is recommended')
                end
            end
            [X1,W] = GaussianMixtureQuadrature(mixprobs_i(:,jj),mu_i(:,jj),sigma_i(:,jj),znum);
            X1 = X1 + mewz(jj);
    end

    z_grid = allcomb2(X1); % Nm*1 matrix of grid points
    z_grid_J(:,jj)=z_grid;
end


%% Step 3: Compute the transition matrices pi_z_J(:,:,jj), from period jj to period jj+1
% Note: the transition from jj to jj+1 is governed by the period jj+1 parameters, hence the jj+1 indexes below

nMoments_grid=zeros(znum,J-1); % Used to record number of moments matched in transition from each point

parfor jj=1:J-1
    %% compute conditional moments
    % fprintf('discretizeLifeCycleAR1wGM_KFTT: now doing period %i of %i \n',jj,J) % commented out as parfor
    sigmaC2 = sigma_i(:,jj+1).^2;

    zlag_grid=z_grid_J(:,jj);
    z_grid=z_grid_J(:,jj+1)';

    TBar=TBar_J(:,jj+1);

    % gmdistribution and pdf() need the Statistics Toolbox, and a gaussian mixture density is
    % one line without them: sum_i p_i*normpdf(x,mu_i,s_i). The component parameters are kept
    % as plain vectors and the density is written out at each use below.
    gmmu = mu_i(:,jj+1); gmsd = sqrt(sigmaC2); gmp = mixprobs_i(:,jj+1); % the mixture, as plain vectors

    P = NaN(znum,znum); % transition probability matrix
    P1 = NaN(znum,znum); % matrix to store transition probability
    P2 = ones(znum,1); % znum*1 matrix used to construct P
    scalingFactor = max(abs(z_grid));
    kappa = 1e-8;

    for z_c = 1:znum % For each value z(jj-1) compute the conditional distribution for z(jj) [the row of the transition matrix]

        % First, calculate what Farmer & Toda (2017) call qnn', which are essentially an initial guess for pnn'
        % B30: mew(jj+1) used to be missing here, so the drift never reached the transitions -
        % it entered only the grid-centring recursion above. The gaussian sibling
        % discretizeLifeCycleAR1_KFTT has always had it, so this was copy-paste drift into the
        % mixture version. With mew=0 it is invisible, which is why it survived; with a drift the
        % chain's mean follows m(j)=rho(j)*m(j-1)+E(e_j) instead of the truth, and on P7's
        % calibration that put the age-29 mean at 0.081 against a true 0.315, flat across a
        % tenfold grid refinement.
        condMean = mew(jj+1)+rho(jj+1)*zlag_grid(z_c); % z_grid(ii) here is the lag grid point
        xPDF = (z_grid-condMean)';
        switch kfttoptions.method
            case 'gauss-hermite'
                q = W.*(sum(gmp'.*exp(-0.5*((xPDF-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2)./(exp(-0.5*(xPDF./sigma(jj+1)).^2)./(sigma(jj+1)*sqrt(2*pi))))';
            case 'GMQ'
                q = W.*(sum(gmp'.*exp(-0.5*((xPDF-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2)./sum(gmp'.*exp(-0.5*((z_grid'-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2))';
            otherwise
                q = W.*(sum(gmp'.*exp(-0.5*((xPDF-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2))';
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

%% Get the distribution of z in period 1, jequaloneDistz
% Period 1 is iid, so just discretize it directly onto the period 1 grid.
% Note: pass method, nSigmas and nMoments so that the grid this builds is the same as z_grid_J(:,1)
farmertodaoptions_j1.nSigmas=kfttoptions.nSigmas(1); % nSigmas is an age-dependent vector by this point, period 1 is the relevant age
farmertodaoptions_j1.method=kfttoptions.method;
farmertodaoptions_j1.nMoments=kfttoptions.nMoments;
farmertodaoptions_j1.parallel=1; % need to get solution on cpu, otherwise causes errors later
% Note: a normal distribution is just a one-component gaussian mixture, so every case goes through
% discretizeAR1wGM_FarmerToda(). Using it rather than discretizeAR1_FarmerToda() also means every
% kfttoptions.method is supported here, including 'GMQ' which only the mixture command has.
if initialj1mixture==1
    % Period 1 was set as a gaussian mixture
    [z_grid_1,pi_z_1] = discretizeAR1wGM_FarmerToda(0,0,initialj1mixprobs,initialj1mu_i,initialj1sigma_i,znum,farmertodaoptions_j1);
    jequaloneDistz=pi_z_1(1,:)'; % iid, so first row is the dist
elseif isfield(kfttoptions,'initialj1mewz') || isfield(kfttoptions,'initialj1sigmaz')
    % Period 1 was set as a normal distribution
    if sigmaz(1)>0
        % B31: the mean used to be passed as mu_i, with mew=0. discretizeAR1wGM_FarmerToda
        % centres its grid on mew/(1-rho) and does not look at the mixture mean, so that built
        % the period 1 distribution on a grid centred at zero, which was then applied to
        % z_grid_J(:,1) centred at mewz(1) - the two offsets added and the period 1 mean came
        % back DOUBLED. The guard below detected it and warned, but the wrong object was
        % returned anyway. Passing the mean as mew centres the grid where it belongs.
        [z_grid_1,pi_z_1] = discretizeAR1wGM_FarmerToda(mewz(1),0,1,0,sigmaz(1),znum,farmertodaoptions_j1);
        jequaloneDistz=pi_z_1(1,:)'; % iid, so first row is the dist
    else
        % All grid points are same, so just pick an arbitrary one
        z_grid_1=z_grid_J(:,1);
        jequaloneDistz=zeros(znum,1);
        jequaloneDistz(ceil(znum/2))=1; % put the mass on the median point (irrelevant as all points will anyway be same value)
    end
else
    % Period 1 follows from period 0: z(1)=mew(1)+rho(1)*z(0)+e(1), with e(1) the period 1 gaussian mixture.
    % If period 0 was given a variance, that convolves with the mixture, and the convolution of a normal
    % with a gaussian mixture is just a gaussian mixture with larger component variances.
    j1sigma_i=sigma_i(:,1);
    if isfield(kfttoptions,'initialj0sigmaz')
        j1sigma_i=sqrt(rho(1)^2*kfttoptions.initialj0sigmaz^2+sigma_i(:,1).^2);
    end
    [z_grid_1,pi_z_1] = discretizeAR1wGM_FarmerToda(mew(1)+rho(1)*z0,0,mixprobs_i(:,1),mu_i(:,1),j1sigma_i,znum,farmertodaoptions_j1);
    jequaloneDistz=pi_z_1(1,:)'; % iid, so first row is the dist
end
if max(abs(z_grid_1(:)-z_grid_J(:,1)))>10^(-9)
    warning('discretizeLifeCycleAR1wGM_KFTT: the period 1 grid used to build jequaloneDistz differs from z_grid_J(:,1), so jequaloneDistz is putting mass on the wrong points')
end
clear pi_z_1 z_grid_1


%% I AM BEING LAZY AND JUST MOVING RESULT TO GPU RATHER THAN CREATING IT THERE IN THE FIRST PLACE
if kfttoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    pi_z_J=gpuArray(pi_z_J);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid; % Heatmap of how many moments where hit by the conditional (difference) distribution
otheroutputs.sigma_z=sigmaz; % Standard deviation of z (for each period)
otheroutputs.mew_z=mewz; % Mean of z (for each period)
otheroutputs.jequalzeroDistz=jequalzeroDistz; % store this so that user can see it to check it looks like they intend (a way to double-check the input options)
otheroutputs.z_grid_0=z_grid_0; % the period 0 grid that jequalzeroDistz sits on


end
