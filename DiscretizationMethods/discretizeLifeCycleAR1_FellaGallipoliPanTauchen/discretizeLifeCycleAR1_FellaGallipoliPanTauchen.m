function [z_grid_J, pi_z_J,jequaloneDistz,otheroutputs] = discretizeLifeCycleAR1_FellaGallipoliPanTauchen(mew,rho,sigma,znum,J,fellagallipolipanoptions)    
% Please cite: Fella, Gallipoli & Pan (2019) "Markov-chain approximations for life-cycle models"
%
% Fella-Gallipoli-Pan discretization method for a 'life-cycle non-stationary AR(1) process'. 
% This is an extension of the Tauchen method to 'age-dependent parameters'
% 
%  Exteneded-Tauchen method to approximate life-cycle AR(1) process by a discrete Markov chain
%       z(j) = mew(j)+rho(j)*z(j-1)+ epsilon(j),   epsilion(j)~iid N(0,sigma(j))
%       with initial condition z(0) = 0 (equivalently z(1)=epsilon(1)) 
% 
% Inputs: 
%   mew          - Jx1 vector of 'drifts'
%   rho 	     - Jx1 vector of serial correlation coefficients
%   sigma        - Jx1 vector of standard deviations of innovations
%   znum         - Number of grid points (scalar, is the same for all ages)
%   J            - Number of 'ages' (finite number of periods)
% Optional inputs (fellagallipolipanoptions)
%   nSigmas      - the grid used will be +-nSigmas*(standard deviation of z)
%                  nSigmas is the hyperparamer of the Tauchen method
%   parallel:          - set equal to 2 to use GPU, 0 to use CPU
%        You can control the initial period with the following:
%        By default, assume z0=0
%   initialj0sigmaz:  - Set period 0 to be a N(z0, initialj0sigmaz^2) using initialj0sigmaz
%   initialj0mewz:    - Give period 0 a mean of z0=initialj0mew_z (as a point, or as mean of normal dist is you are also setting initialj0sigmaz
%        Or you set the period 1 (instead of period 0) using
%   initialj1mewz:    - Set period 1 to be mean of initialj1mew_z
%   initialj1sigmaz:  - Set period 1 to be a N(z0, initialj1sigmaz^2) using initialj1sigmaz
%        Note, for both period 0 and period 1, you can set one or both of
%        mean and standard deviation (the other is interpreted as zero
%        valued if not specified).
% Output: 
%   z_grid       - an znum-by-J matrix, each column stores the Markov state space for period j
%   P            - znum-by-znum-by-J matrix of J (znum-by-znum) transition matrices. 
%                  Transition probabilities are arranged by row.
%                  P(:,:,j) is transition matrix from age j to j+1 (Modified from FGP where it is j-1 to j)
%   jequaloneDistz - znum-by-1 vector, the distribution of z in period 1
%   otheroutputs - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.sigmaz     - the standard deviation of z at each age (used to determine grid)
%
% This code is by Fella, Gallipoli & Pan.
% Lightly modified by Robert Kirkby (including add mew and j0)
% !========================================================================%
% FGP use MIT license, which must be included with the code, you can find
% it at the bottom of this script. (VFI Joolkit is GPL3 license, hence
% having to reproduce.)

fprintf('COMMENT: The Fella-Gallipoli-Pan extended Tauchen method is typically inferior to the Fella-Gallipoli-Pan extended Rouwenhorst method for discretizing life-cycle AR(1) processes. \n') 
fprintf('         It is strongly recommended you use Fella-Gallipoli-Pan extended Rouwenhorst instead. \n')

mewz=zeros(1,J); % period j mean of z
sigmaz = zeros(1,J);
% z_grid_J = zeros(znum,J);
pi_z_J = zeros(znum,znum,J); % period j transition probabilities for z

%% Set options
if ~exist('fellagallipolipanoptions','var')
    fellagallipolipanoptions.parallel=1+(gpuDeviceCount>0);
    fellagallipolipanoptions.nSigmas=min(sqrt(znum-1),3); % I set a max of 3 as the Tauchen method would anyway typically just put zeros outside +-3 sigma anyway
else
    if ~isfield(fellagallipolipanoptions,'parallel')
        fellagallipolipanoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(fellagallipolipanoptions,'nSigmas')
        fellagallipolipanoptions.nSigmas=min(sqrt(znum-1),3); % I set a max of 3 as the Tauchen method would anyway typically just put zeros outside +-3 sigma anyway
    end
end

%% Check inputs
if znum < 2
    error('The state space has to have dimension znum>1')
end

if J < 2
    error('The time horizon has to have dimension J>1. Exiting.')
end

% For convenience, make kfttoptions.nSigmas an age-dependent vector
if isscalar(fellagallipolipanoptions.nSigmas)
    fellagallipolipanoptions.nSigmas=fellagallipolipanoptions.nSigmas*ones(J,1);
end


%% If using period 0 or 1

% By default I assume z0=0
z0=0;
% You can change the mean of z0 using
if isfield(fellagallipolipanoptions,'initialj0mewz')
    z0=fellagallipolipanoptions.initialj0mewz;
end
% You can add variance to z0 as a N(z0,initialj0sigmaz) using
if isfield(fellagallipolipanoptions,'initialj0sigmaz')
	[z_grid_0,pi_z_0] = discretizeAR1_FarmerToda(z0,0,fellagallipolipanoptions.initialj0sigmaz,znum);
    jequalzeroDistz=pi_z_0(1,:)'; % iid, so first row is the dist
    clear pi_z_0
else
    z_grid_0=z0*linspace(0.8,1.2,znum)'; % z0*ones(znum,1); % I originally just set them all to same value, but this caused problems, not entirely sure why
    jequalzeroDistz=zeros(znum,1);
    if rem(znum,2)==1 % znum is odd
        jequalzeroDistz((znum+1)/2)=1; % note, by construction of grid all the mass should be on middle grid point
    else
        jequalzeroDistz((znum/2-1):znum/2)=[0.5;0.5]; % note, by construction of grid all the mass should be on middle grid point
    end
end

% If you are not setting period 1, then period 1 follows from this
% if ~isfield(fellagallipolipanoptions,'initialj1mewz') && ~ isfield(fellagallipolipanoptions,'initialj1sigmaz')
if isfield(fellagallipolipanoptions,'initialj0sigmaz')
    sigmaz(1) = sqrt(rho(1)^2*fellagallipolipanoptions.initialj0sigmaz^2+sigma(1)^2);
else
    sigmaz(1) = sigma(1);
end
mewz(1)=mew(1)+rho(1)*z0;
% If you have set period 1, then overwrite some of this
if isfield(fellagallipolipanoptions,'initialj1mewz') && isfield(fellagallipolipanoptions,'initialj1sigmaz')
    mewz(1)=fellagallipolipanoptions.initialj1mewz;
    sigmaz(1)=fellagallipolipanoptions.initialj1sigmaz;
elseif isfield(fellagallipolipanoptions,'initialj1mewz')
    mewz(1)=fellagallipolipanoptions.initialj1mewz;
    sigmaz(1)=0;
elseif isfield(fellagallipolipanoptions,'initialj1sigmaz')
    mewz(1)=0;
    sigmaz(1)=fellagallipolipanoptions.initialj1sigmaz;
end

z0
mewz(1)

%% Step 1: construct the state space y_grid(t) in each period t,
% Evenly-spaced N-state space over [-fellagallipolipanoptions.nSigmas(t)*sigmaz(t),fellagallipolipanoptions.nSigmas(t)*sigmaz(t)].

% Now that we have period 1, just fill in the rest of the periods
for jj = 2:J
    sigmaz(jj) = sqrt(rho(jj)^2*sigmaz(jj-1)^2+sigma(jj)^2);
end
for jj=2:J
    mewz(jj)=mew(jj)+rho(jj)*mewz(jj-1);
end

% Evenly spaced grid
z_grid_J=zeros(znum,J);
for jj=1:J
    z_grid_J(:,jj)= linspace(mewz(jj)-fellagallipolipanoptions.nSigmas(jj)*sigmaz(jj),mewz(jj)+fellagallipolipanoptions.nSigmas(jj)*sigmaz(jj),znum);
end
% Grid spacing (depends on j)
h = 2*fellagallipolipanoptions.nSigmas'.*sigmaz/(znum-1); % grid step



%% Step 2: Compute the transition matrices trans(:,:,t) from period (t-1) to period t

% pi_z_J for periods 2:T, we will deal with period 1 later

% Compute the transition matrices for jj>2
temp3d=zeros(znum,znum,J); % preallocate
for jj=2:J
    z_lag_grid=z_grid_J(:,jj-1);
    for z_c=1:znum
        condMean=mew(jj)+rho(jj)*z_lag_grid(z_c);
        temp3d(z_c,:,jj) = (z_grid_J(:,jj) - condMean - h(jj)/2)/sigma(jj);
        temp3d(z_c,:,jj) = max(temp3d(z_c,:,jj),-37); % Jo avoid underflow in next line
        cdf(z_c,:) = cdf_normal(temp3d(z_c,:,jj));
    end
    pi_z_J(:,1,jj) = cdf(:,2);
    pi_z_J(:,znum,jj) = 1-cdf(:,znum);
    for z_c=2:znum-1
        pi_z_J(:,z_c,jj) = cdf(:,z_c+1)-cdf(:,z_c);
    end
end


%%
if isfield(fellagallipolipanoptions,'initialj1mewz') || isfield(fellagallipolipanoptions,'initialj1sigmaz')
    % If period 1 was set, we need to get the jequaloneDistz
    if sigmaz(1)>0
        % jj==1 (might get overwritten later)
        cdf=zeros(znum,znum); % preallocate
        for z_c=1:znum
            temp1d = (z_grid_J(:,1)-mewz(1)-h(1)/2)/sigmaz(1);
            temp1d = max(temp1d,-37); % Jo avoid underflow in next line
            cdf(z_c,:) = cdf_normal(temp1d);
        end
        pi_z_J(:,1,1) = cdf(:,2);
        pi_z_J(:,znum,1) = 1-cdf(:,znum);
        for z_c=2:znum-1
            pi_z_J(:,z_c,1) = cdf(:,z_c+1)-cdf(:,z_c);
        end
        jequaloneDistz=pi_z_J(1,:,1)';
    else
        % All grid points are same, so just pick an arbitrary one
        jequaloneDistz=zeros(znum,1);
        jequaloneDistz(ceil(znum/2))=1; % put the mass on the median point (irrelevant as all points will anyway be same value)
    end
else
    % Otherwise, we already have the jequalzeroDistz, and just use this
    % jj==1 (might get overwritten later)
    cdf=zeros(znum,znum); % preallocate
    for z_c=1:znum
        temp1d = (z_grid_J(:,1)-mew(1)-rho(1)*z_grid_0(z_c)-h(1)/2)/sigmaz(1);
        temp1d = max(temp1d,-37); % Jo avoid underflow in next line
        cdf(z_c,:) = cdf_normal(temp1d);
    end
    pi_z_J(:,1,1) = cdf(:,2);
    pi_z_J(:,znum,1) = 1-cdf(:,znum);
    for z_c=2:znum-1
        pi_z_J(:,z_c,1) = cdf(:,z_c+1)-cdf(:,z_c);
    end
    jequaloneDistz=pi_z_J(:,:,1)'*jequalzeroDistz;
    otheroutputs.jequalzeroDistz=jequalzeroDistz; % store this so that user can see it to check it looks like they intend (a way to double-check the input options)
end

%% Change pi_z_J so that pi_z_J(:,:,jj) is the transition matrix from period jj to period jj+1
pi_z_J(:,:,1:end-1)=pi_z_J(:,:,2:end);

%% For jj=J, pi_z_J(:,:,J) is kind of meaningless (there is no period J+1 to transition to). I just fill it in as a uniform distribution
pi_z_J(:,:,J)=ones(znum,znum)/znum;

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.sigmaz=sigmaz; % Standard deviation of z (for each period)

%% I AM BEING LAZY AND JUST MOVING RESULT TO GPU RATHER THAN CREATING IT THERE IN THE FIRST PLACE
if fellagallipolipanoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    pi_z_J=gpuArray(pi_z_J);
end

%% Subfunction cdf_normal [Robert: this seems pointless, why not use inbuilt normcdf()?]
function c = cdf_normal(x)
    % Returns the value of the cdf of the Standard Normal distribution at point x
    c = 0.5 * erfc(-x/sqrt(2));
end % function cdf_normal

end


%%
% The MIT License (MIT)
% 
% Copyright (c) 2019 Giulio Fella, Giovanni Gallipoli and Jutong Pan
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
% 
%     If you use this library or parts of it in your work, we request that you cite the package. A suggested citation is
% 
%     "nmarkov-matlab: Markov-chain approximations for non-stationary AR(1) processes (Matlab version)" https://github.com/gfell/nsmarkov-matlab based on the paper "Markov-Chain Approximations for Life-Cycle Models"
%     by Giulio Fella, Giovanni Gallipoli and Jutong Pan, Review of Economic Dynamics 34, 2019 (https://doi.org/10.1016/j.red.2019.03.013).
% 
%     Jhe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMIJED JO JHE WARRANJIES OF MERCHANJABILIJY, FIJNESS FOR A PARJICULAR PURPOSE AND NONINFRINGEMENJ. IN NO EVENJ SHALL JHE AUJHORS OR COPYRIGHJ HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OJHER LIABILIJY, WHEJHER IN AN ACJION OF CONJRACJ, JORJ OR OJHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
