function [z_grid_J,pi_z_J,jequaloneDistz,otheroutputs]=discretizeLifeCycleVAR1_Tauchen(Mew_J,Rho_J,SigmaSq_J,znum,J,tauchenoptions)
% This is an extension of the Tauchen method to 'age-dependent parameters'
%
% Purpose:
%       Compute a finite-state Markov chain approximation to age-dependent VAR(1)
%       process of the form
%
%           Z_(j) = Mew(j) + Rho(j)*Z_(j-1) + SigmaSq(j)^(1/2)*epsilon_(j)
%
%       where epsilon_(j) is an (M x 1) vector of independent standard
%       normal innovations. Notice that SigmaSq is the variance-covariance matrix.
%       Default: Z_(1) is N(Mew(1),SigmaSq(1))
%
% Usage:
%       [z_grid_J,pi_z_J] = discretizeVAR1_FarmerToda(Mew,Rho,SigmaSq,znum,farmertodaoptions)
%
% Inputs:
%   Mew_J     - (M x J) constant vector
%   Rho_J     - (M x M x J) matrix of impact coefficients
%   SigmaSq_J - (M x M x J) variance-covariance matrix of the innovations
%   znum      - (M x 1) Desired number of discrete points in each dimension
%               (you can input a scalar, which will be interpreted as same number of points for each dimension)
%   J         - Number of periods
% Optional inputs (farmertodaoptions):
%   parallel: - set equal to 2 to use GPU, 0 to use CPU
%   nMoments  - Desired number of moments to match. The default is 2.
%   method    - String specifying the method used to determine the grid points.
%               Accepted inputs are 'even,' 'quantile,' and 'quadrature.' The
%               default option is 'even.' Please see the paper for more details.
%   nSigmas   - If method='even' option is specified, nSigmas is used to
%               determine the number of unconditional standard deviations
%               used to set the endpoints of the grid (mew+-nSigmas*standarddeviation)
%
% Outputs:
%   z_grid_J  - (sum(znum) x J) stacked column vector of states, one for each age/period j.
%               z_grid_J(:,j) stacks the M one-dimensional grids for period j.
%   pi_z_J    - (prod(znum) x prod(znum) x (J-1)) probability transition matrix. Each row
%               corresponds to a discrete conditional probability
%               distribution over the state M-tuples. pi_z_J(:,:,j) is the
%               transition from age j to age j+1; there are only J-1 of them,
%               as there is no period J+1 to transition to.
%   jequaloneDistz - (prod(znum) x 1) the distribution of z in period 1
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.mewz         - (M-by-J) the mean of z at each age (used to determine grid)
%        otheroutputs.SigmaSqz     - (M-by-M-by-J) the variance-covariance matrix of z at each age (used to determine grid)
%        otheroutputs.sigma_z      - (M-by-J) the standard deviation of z at each age (used to determine grid)
%
% NOTES:
% - only accepts non-singular variance-covariance matrices.
% - only constructs tensor product grids where each dimension
%     contains the same number of points. For this reason it is recommended
%     that this code not be used for problems of more than about 4 or 5
%     dimensions due to curse of dimensionality issues.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loosely speaking, this is just an extension of the Tauchen method,
% realistically though this goes notably beyond the Tauchen method, in
% particular Tauchen suggests using the product of the marginal cdfs,
% whereas here we use the joint cdf directly.

l_z=size(Rho_J,1);

% NORMALISE znum TO A COLUMN HERE, before the options block below, because the default nSigmas is
% built from znum and is then used as nSigmas.*sigmaz with sigmaz of size l_z-by-J. Until this was
% moved up, the default was min(sqrt(znum-1)',3) evaluated on whatever shape the caller passed: a
% scalar stayed scalar and broadcast against anything, a row vector became the l_z-by-1 column that
% broadcast correctly, and a COLUMN vector - the shape this command's own header documents as
% (M x 1) - became 1-by-l_z, which has no valid expansion against l_z-by-J unless J happens to
% equal l_z. So the documented way of calling it errored with "Arrays have incompatible sizes",
% and only the undocumented row form worked. Found by P8 of the test bank, which passes znum in all
% three shapes and requires them to agree.
if isscalar(znum)
    znum=znum*ones(l_z,1);
end
znum=znum(:); % a column, whatever shape it arrived in

warning off MATLAB:singularMatrix % suppress inversion warnings

% mewz=zeros(l_z,J); % period j mean of z
% SigmaSqz = zeros(l_z,l_z,J); % period j covariance-matrix of z
% z_grid_J = zeros(sum(znum),J); % period j grid on z
pi_z_J = zeros(prod(znum),prod(znum),J-1); % period j transition probabilities for z (only J-1 of them, there is no period J+1 to transition to)

%% Set options
if ~exist('tauchenoptions','var')
    tauchenoptions.method='even'; % Informally I have the impression even is more robust
    tauchenoptions.nSigmas = min(sqrt(znum-1),3); % Maximum of +-3 standard deviations (a column, one per variable)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
else
    if ~isfield(tauchenoptions,'method')
        tauchenoptions.method='even'; % Informally I have the impression even is more robust
    end
    if ~isfield(tauchenoptions,'nSigmas')
        tauchenoptions.nSigmas = min(sqrt(znum-1),3); % Maximum of +-3 standard deviations (a column, one per variable)
    end
    if ~isfield(tauchenoptions,'parallel')
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
end



%% some error checking
if znum < 2
    error('The state space has to have dimension znum>1.')
end
if J < 2
    error('The time horizon has to have dimension J>1.')
end

if ~all(size(Mew_J)==[l_z,J])
    error('Mew must have length(znum) rows and J columns')
end
if ~all(size(Rho_J)==[l_z,l_z,J])
    error('Rho must be of size [length(znum),length(znum),J]')
end
if ~all(size(SigmaSq_J)==[l_z,l_z,J])
    error('SigmaSq must be of size [length(znum),length(znum),J]')
end
for jj=1:J
    SigmaSq_jj=SigmaSq_J(:,:,jj);
    if any(SigmaSq_jj(logical(eye(l_z,l_z))) < 0)
        error('variances (diagonal elements of SigmaSq) must be positive (for age jj equals %i)',jj)
    end
end
% if max(eig(Rho))>=1
%     error('autocorrelation coefficient (spectral radius) must be less than one')
% end

% Check that Sigma is a valid covariance matrix
for jj=1:J
    [~,posDefCheck] = chol(SigmaSq_J(:,:,jj));
    if posDefCheck
        fprintf('Following error relates to period %i \n',jj)
        error('SigmaSq must be a positive definite matrix')
    end
end

% znum was normalised to a column at the top of the file, before the options block that reads it
% Check that znum is a valid number of grid points
if ~all(isnumeric(znum)) || any(znum < 3) || any(rem(znum,1) ~= 0)
    error('znum must be a (vector of) positive integer greater than 3')
end

% A user-supplied nSigmas has to be a column for the same reason the default is: it multiplies
% sigmaz, which is l_z-by-J. A scalar is fine and broadcasts.
if ~isscalar(tauchenoptions.nSigmas)
    tauchenoptions.nSigmas=tauchenoptions.nSigmas(:);
    if length(tauchenoptions.nSigmas)~=l_z
        error('tauchenoptions.nSigmas must be a scalar, or a vector with one element per variable of the VAR (which is size(Rho_J,1))')
    end
end


%% Step 2: construct the state space z_grid_J(j) in each period j.
% Evenly-spaced N-state space over [-tauchenoptions.nSigmas*sigmaz(j),tauchenoptions.nSigmas*sigmaz(j)].

% Note: I set up the period 0, but this won't end up used if you have used options to set period 1.

% By default I assume z0=0
z0=zeros(l_z,1);
% You can change the mean of z0 using
if isfield(tauchenoptions,'initialj0mewz')
    z0=tauchenoptions.initialj0mewz;
end

% If you are not setting period 1, then period 1 follows from this
% if ~isfield(tauchenoptions,'initialj1mewz') && ~ isfield(tauchenoptions,'initialj1sigmaz')
SigmaSqz=zeros(l_z,l_z,J);
SigmaSqz(:,:,1) = SigmaSq_J(:,:,1); % because period 0 has no covar, it is just whatever the shocks are
mewz=zeros(l_z,J);
mewz(:,1)=Mew_J(:,1)+Rho_J(:,:,1)*z0; % period 0 is a point mass at z0 (z0 is zero unless tauchenoptions.initialj0mewz was used)
% If you have set period 1, then overwrite some of this
if isfield(tauchenoptions,'initialj1mewz') && isfield(tauchenoptions,'initialj1SigmaSqz')
    mewz(:,1)=tauchenoptions.initialj1mewz;
    SigmaSqz(:,:,1)=tauchenoptions.initialj1SigmaSqz;
elseif isfield(tauchenoptions,'initialj1mewz')
    mewz(:,1)=tauchenoptions.initialj1mewz;
    SigmaSqz(:,:,1)=zeros(l_z,l_z);
elseif isfield(tauchenoptions,'initialj1SigmaSqz')
    mewz(:,1)=zeros(l_z,1);
    SigmaSqz(:,:,1)=tauchenoptions.initialj1SigmaSqz;
end

% Now that we have period 1, just fill in the rest of the periods
sigmaz=zeros(l_z,J);
sigmaz(:,1)=sqrt(diag(SigmaSqz(:,:,1))); % std dev of z
for jj = 2:J
    % Covar(z_j)=rho_j Covar(z_{j-1}) rho_j^T + Covar(epsilon_j)
    % [z_j=Mew_j+Rho_j*z_{j-1}+e_j with z a column vector, so Var(Rho*z)=Rho*Var(z)*Rho']
    SigmaSqz(:,:,jj) = Rho_J(:,:,jj)*SigmaSqz(:,:,jj-1)*Rho_J(:,:,jj)'+SigmaSq_J(:,:,jj);
    sigmaz(:,jj)=sqrt(diag(SigmaSqz(:,:,jj))); % std dev of z
end
for jj=2:J
    mewz(:,jj)=Mew_J(:,jj)+Rho_J(:,:,jj)*mewz(:,jj-1);
end

q_sigmaz=tauchenoptions.nSigmas.*sigmaz;

%% Create the grids (simple product grid)
z_grid_J=zeros(sum(znum),J);
for jj=1:J
    z_grid=zeros(sum(znum),1); % Stacked column vector
    % Create z_grid for each of the M dimensions
    cumsum_znum=cumsum(znum);
    z_grid(1:cumsum_znum(1))=linspace(-1,1,znum(1))*q_sigmaz(1,jj)+mewz(1,jj); % ii=1
    for z_c=2:l_z
        z_grid(cumsum_znum(z_c-1)+1:cumsum_znum(z_c))=linspace(-1,1,znum(z_c))*q_sigmaz(z_c,jj)+mewz(z_c,jj); % For the ii-th dimension, points evenly spaced from -q_sigmaz(ii) to +q_sigmaz(ii)
    end

    z_grid_J(:,jj)=z_grid;
end


%% Get the distribution of z in period 1, jequaloneDistz
% Period 1 is N(mewz(:,1),SigmaSqz(:,:,1)), so just put that onto the period 1 grid
if min(diag(SigmaSqz(:,:,1)))>0
    mvnoptions.parallel=1; % want this on cpu, is anyway moved to gpu at the end if needed
    mvnoptions.verbose=0;
    jequaloneDistz=MVNormal_ProbabilitiesOnGrid(z_grid_J(:,1),mewz(:,1),SigmaSqz(:,:,1),znum,mvnoptions);
    jequaloneDistz=jequaloneDistz(:)/sum(jequaloneDistz(:)); % mvncdf() is only accurate to a tolerance, so renormalize
else
    % At least one dimension of period 1 has zero variance, so mvncdf() cannot be used.
    % By construction of the grid, every point of a zero-variance dimension equals mewz, so put the mass on the median point.
    if max(diag(SigmaSqz(:,:,1)))>0
        warning('discretizeLifeCycleVAR1_Tauchen: period 1 has zero variance in some but not all dimensions, jequaloneDistz is just being set as a point mass on the median grid point')
    end
    jequaloneindex=1;
    stepsize=1;
    for z_c=1:l_z
        jequaloneindex=jequaloneindex+(ceil(znum(z_c)/2)-1)*stepsize;
        stepsize=stepsize*znum(z_c);
    end
    jequaloneDistz=zeros(prod(znum),1);
    jequaloneDistz(jequaloneindex)=1;
end


%% Create pi_z_J
pi_z_J=zeros(prod(znum),prod(znum),J-1); % preallocate (only J-1, there is no period J+1 to transition to)

for jj=1:J-1
    % P from jj to jj+1
    z_grid=z_grid_J(:,jj+1);
    z_gridvals=CreateGridvals(znum,z_grid,1);
    z_gridvals_lag=CreateGridvals(znum,z_grid_J(:,jj),1);

    if l_z>=1
        z1_grid=z_grid(1:znum(1));
        z1_gridspacing_up=[(z1_grid(2:end)-z1_grid(1:end-1))/2; Inf];
        z1_gridspacing_down=[Inf; (z1_grid(2:end)-z1_grid(1:end-1))/2]; % Note: will be subtracted from grid point, hence Inf, not -Inf
        if l_z>=2
            z2_grid=z_grid(znum(1)+1:sum(znum(1:2)));
            z2_gridspacing_up=[(z2_grid(2:end)-z2_grid(1:end-1))/2; Inf];
            z2_gridspacing_down=[Inf; (z2_grid(2:end)-z2_grid(1:end-1))/2];
            if l_z>=3
                z3_grid=z_grid(sum(znum(1:2))+1:sum(znum(1:3)));
                z3_gridspacing_up=[(z3_grid(2:end)-z3_grid(1:end-1))/2; Inf];
                z3_gridspacing_down=[Inf; (z3_grid(2:end)-z3_grid(1:end-1))/2];
                if l_z>=4
                    z4_grid=z_grid(sum(znum(1:3))+1:sum(znum(1:4)));
                    z4_gridspacing_up=[(z4_grid(2:end)-z4_grid(1:end-1))/2; Inf];
                    z4_gridspacing_down=[Inf; (z4_grid(2:end)-z4_grid(1:end-1))/2];
                    if l_z>=5
                        z5_grid=z_grid(sum(znum(1:4))+1:sum(znum(1:5)));
                        z5_gridspacing_up=[(z5_grid(2:end)-z5_grid(1:end-1))/2; Inf];
                        z5_gridspacing_down=[Inf; (z5_grid(2:end)-z5_grid(1:end-1))/2];
                    end
                end
            end
        end
    end

    % Now do the actual multivariate normal cdf calculation
    if l_z==1
        z_gridspacing_up=z1_gridspacing_up;
        z_gridspacing_down=z1_gridspacing_down;
    elseif l_z==2
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down],1);
    elseif l_z==3
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down],1);
    elseif l_z==4
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up;z4_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down;z4_gridspacing_down],1);
    elseif l_z==5
        z_gridspacing_up=CreateGridvals(znum,[z1_gridspacing_up;z2_gridspacing_up;z3_gridspacing_up;z4_gridspacing_up;z5_gridspacing_up],1);
        z_gridspacing_down=CreateGridvals(znum,[z1_gridspacing_down;z2_gridspacing_down;z3_gridspacing_down;z4_gridspacing_down;z5_gridspacing_down],1);
    end

    % THE PARAMETERS ARE INDEXED jj+1, NOT jj. Slice jj is the transition INTO age jj+1, and the
    % process is Z(j)=Mew(:,j)+Rho(:,:,j)*Z(j-1)+e(j), so the step from jj to jj+1 is governed by
    % age jj+1's parameters - which is the same convention the mean recursion above uses when it
    % writes mewz(:,jj)=Mew_J(:,jj)+Rho_J(:,:,jj)*mewz(:,jj-1). All three were indexed jj until
    % 2026-09-16, one age too early, while the destination grid z_grid_J(:,jj+1) was already right.
    %
    % It is silent whenever the parameters do not vary with age, which is why it survived: the
    % frozen-parameter identity against discretizeVAR1_Tauchen cannot see it, and neither can a
    % driftless calibration. P8 of the test bank found it with age-varying Mew and Rho - the chain's
    % mean followed m(j+1)=Mew(:,j)+Rho(:,:,j)*m(j) instead of the truth, giving a mean error of
    % 3.1e-02 at age 15 that did not shrink with either grid refinement or grid width.
    for z_c=1:prod(znum) % lag of z
        conditionalmean=(Mew_J(:,jj+1)+Rho_J(:,:,jj+1)*z_gridvals_lag(z_c,:)')';
        pi_z_J(z_c,:,jj)=mvncdf(z_gridvals-z_gridspacing_down,z_gridvals+z_gridspacing_up,conditionalmean,SigmaSq_J(:,:,jj+1))';
    end


end





%%
otheroutputs.mewz=mewz; % period j mean of z
otheroutputs.SigmaSqz=SigmaSqz; % period j variance-covariance matrix of z
otheroutputs.sigma_z=sigmaz; % period j standard deviation of z

%%
if tauchenoptions.parallel==2
    z_grid_J=gpuArray(z_grid_J);
    pi_z_J=gpuArray(pi_z_J); %(z,zprime)
    jequaloneDistz=gpuArray(jequaloneDistz);
end





end