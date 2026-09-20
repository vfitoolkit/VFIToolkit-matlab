function [z_grid,pi_z,otheroutputs] = discretizeARpwGM_FarmerToda(mew,Rho,mixprobs_i,mu_i,sigma_i,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments"
%
% Create states vector, z_grid, and transition matrix, pi_z, for the discrete markov process
%    approximation of the AR(p) process
%       z'=mew+Rho(1)*z+Rho(2)*zlag1+...+Rho(p)*zlag(p-1)+e,   e~F
%          where F=sum_{i=1}^nmix mixprobs_i*N(mu_i,sigma_i^2) is a gaussian mixture
%    by the Farmer-Toda method.
%
% We use "nmix" to denote the number of normal distributions being mixed in the gaussian mixture.
%
% An AR(p) needs p lags in the state, so the discretization is p-dimensional: the state is
% (z, zlag1, ..., zlag(p-1)) and the chain has znum^p states. Set n_z=znum*ones(1,p) when you pass
% the result to the rest of the toolkit. Only the FIRST of the p dimensions is stochastic; the other
% p-1 are the deterministic shift that moves each lag along one period, so each row of pi_z has at
% most znum non-zero entries out of znum^p.
%
% Inputs
%   mew            - constant term coefficient (the INTERCEPT of the AR(p), not the mean of z)
%   Rho            - (1-by-p or p-by-1) the p autoregressive coefficients, Rho(k) multiplies the k-th lag
%   mixprobs_i     - (nmix-by-1) mixture probabilities of the gaussian mixture innovations (must sum to 1)
%   mu_i           - (nmix-by-1) means of the gaussian mixture innovations
%   sigma_i        - (nmix-by-1) standard deviations of the gaussian mixture innovations
%   znum           - number of grid points PER DIMENSION (scalar, or a p-vector whose entries are
%                    all equal; minimum of 3)
% Optional Inputs (farmertodaoptions)
%   method         - The method used to determine the grid ('even','gauss-legendre','clenshaw-curtis','gauss-hermite','GMQ')
%   nMoments       - Number of conditional moments to match (default=4; the point of a gaussian
%                    mixture is usually the higher moments, so the default is 4 rather than 2)
%   nSigmas        - (Hyperparameter) Defines max/min grid points as mewz+-nSigmas*sigmaz
%                    (default sqrt(2*(znum-1)) or sqrt(znum-1), on a persistence branch)
%   parallel       - set equal to 2 to use GPU, 0 to use CPU
%   verbose        - set to zero to suppress the report of how many grid points matched fewer moments
%   z_grid         - (znum-by-1 or 1-by-znum) skip grid construction and build pi_z on this grid.
%                    This is the SHARED one-dimensional grid; all p lag dimensions use it, so it has
%                    znum entries and not p*znum. method/nSigmas are ignored.
% Outputs
%   z_grid         - (p*znum-by-1) STACKED grid: the shared one-dimensional grid repeated p times,
%                    which is the toolkit's convention for an exogenous state with p dimensions.
%                    The p blocks are identical because the p dimensions are the same variable at
%                    different lags. Use CreateGridvals(znum*ones(p,1),z_grid,1) to get the joint
%                    values, with the first dimension (the current value of z) varying FASTEST.
%   pi_z           - (znum^p-by-znum^p) transition matrix; pi_z(i,j) is the probability of going
%                    from joint state i to joint state j
%   otheroutputs   - structure containing
%        otheroutputs.nMoments_grid - how many moments the conditional distribution matched from each
%                                     of the znum^p states
%        otheroutputs.mewz          - the unconditional mean of z, mew/(1-sum(Rho))
%        otheroutputs.sigmaz        - the unconditional standard deviation of z
%        otheroutputs.maxabseig     - the largest modulus companion eigenvalue, which is the
%                                     persistence measure this command branches its defaults on
%
% Helpful info:
%   E(z)=(mew+E(e))/(1-sum(Rho)), where E(e)=sum_i mixprobs_i*mu_i. As in discretizeAR1wGM_FarmerToda
%   the grid is centred on mew/(1-sum(Rho)) rather than on E(z); the mixture is not required to be
%   mean zero, and when it is not the two differ.
%   Var(z) solves the Lyapunov equation for the companion form: with F the p-by-p companion matrix
%   and Q=sigma^2 in its (1,1) entry and zero elsewhere, vec(Sigma)=(I-kron(F,F))\vec(Q), and
%   Var(z)=Sigma(1,1). At p=1 this is the familiar sigma^2/(1-rho^2).
%
% NOTE ON p: at most 5 lags, inherited from CreateGridvals which builds the joint grid.
%
% NOTE ON MEMORY: pi_z is znum^p square, so it grows like znum^(2p). At p=2 and znum=15 that is a
% 225-by-225 matrix; at p=3 and znum=15 it is 3375-by-3375, and at p=4 and znum=15 it would be
% 50625-by-50625. The matrix is very sparse by construction but is returned dense, to match the rest
% of the toolkit.
%
%
% A DIFFERENCE FROM THE p=1 COMMANDS, deliberate and worth knowing about. The defaults here branch
% on maxabseig, the largest modulus companion eigenvalue, where discretizeAR1_FarmerToda branches on
% rho itself and discretizeAR1wGM_FarmerToda on rho relative to 1-2/(znum-1). For rho>=0 the two
% agree, since maxabseig is then just rho. For rho<0 they do not: at rho=-0.9 the p=1 commands read
% -0.9<=0.8 and pick the low-persistence default, while this command reads maxabseig=0.9 and picks
% the high-persistence one. The modulus is what governs how fast the process mean-reverts, so a
% strongly negative rho is no less persistent than its positive twin, and this command's reading is
% the one that generalises. It does mean the p=1 reduction identity only holds automatically for
% rho>=0; pin method and nSigmas explicitly to compare at negative rho.
%
% This is an extension of discretizeAR1wGM_FarmerToda to p lags, and reduces to it exactly at p=1.
% It is also discretizeARp_FarmerToda with the gaussian innovation replaced by a mixture, and
% reduces to that when the mixture has one component.
% The method is from Toda's code (https://github.com/alexisakira/discretization), rewritten to the
% toolkit's conventions rather than his: mew is the INTERCEPT here and the unconditional mean there,
% the options are a struct rather than positional, and the grid comes back stacked rather than joint.
% Please cite Farmer & Toda (2017) if you use this.
%
%%%%%%%%%%%%%%%

%% Rho and znum, normalised before anything reads them
Rho=Rho(:)'; % row, one entry per lag
p=length(Rho);
if isscalar(znum)
    znum=znum*ones(1,p);
end
znum=znum(:)';
if length(znum)~=p
    error('znum must be a scalar, or a vector with one element per lag (which is length(Rho))')
end
if ~all(znum==znum(1))
    error('znum must use the same number of grid points for every lag dimension: the p dimensions are the same variable observed at different lags, so they share one grid')
end
znum=znum(1); % from here on znum is the scalar number of points per dimension
% The joint grid is built with CreateGridvals, which handles at most five dimensions. Checked here
% so the error names p rather than surfacing from CreateGridvals as a message about n_x, which the
% caller has never heard of. The same reasoning as the method check further down.
if p>5
    error('Cannot handle p>5 lags (the joint grid is built with CreateGridvals, which handles at most five dimensions). You currently have p=%i',p)
end
if p<1
    error('Rho must have at least one element')
end

%% The companion form, which is where stationarity and persistence come from
% F is the matrix that writes the AR(p) as a VAR(1) in (z,zlag1,...,zlag(p-1)). It is used here only
% to get the unconditional variance and the persistence measure - the transition matrix below is
% NOT built as a VAR(1), because the companion innovation covariance is singular and the p-1
% shift rows are deterministic rather than something to be discretized.
F=zeros(p,p);
F(1,:)=Rho;
if p>1
    F(2:p,1:p-1)=eye(p-1);
end
maxabseig=max(abs(eig(F)));
if maxabseig>=1
    error('Farmer-Toda error: the AR(p) is not stationary (the largest companion eigenvalue has modulus %g, which is not less than one). You cannot discretize a non-stationary process',maxabseig)
end
if maxabseig>=0.99
    fprintf('COMMENT: When discretizing a gaussian AR(p) with persistence (largest companion eigenvalue) greater than 0.99 (which you currently have), the Rouwenhorst method tends to outperform the Farmer-Toda method at p=1. There is no AR(p) Rouwenhorst in the toolkit, so this is a comment rather than a suggestion. \n')
    % Based on Farmer & Toda (2017), last para on pg 678
end

%% The mixture, checked and reduced to the moments the method needs
mixprobs_i=mixprobs_i(:); mu_i=mu_i(:); sigma_i=sigma_i(:);
nmix=length(mixprobs_i);
if length(mu_i)~=nmix || length(sigma_i)~=nmix
    error('mixprobs_i, mu_i and sigma_i must all have one element per mixture component')
end
if abs(sum(mixprobs_i)-1)>10^(-13)
    error('mixture probabilities (mixprobs_i) must sum to one')
end
if any(sigma_i<=0)
    error('mixture standard deviations (sigma_i) must be strictly positive')
end

%% Set defaults
% The persistence branch uses maxabseig where discretizeAR1wGM_FarmerToda uses rho. At p=1 the
% companion matrix is the scalar rho, so maxabseig is abs(rho) and the two agree except in sign -
% which is the right behaviour, since it is the modulus that governs how fast the process
% mean-reverts, and a negative rho is no less persistent than its positive twin.
if ~exist('farmertodaoptions','var')
    farmertodaoptions.method='even'; % Informally I have the impression even is more robust
    farmertodaoptions.nMoments=4; % the point of gaussian mixtures is typically to get higher order moments
    if maxabseig <= 1-2/(znum-1)  % This is just what Toda used, with maxabseig in place of rho
        farmertodaoptions.nSigmas=sqrt(2*(znum-1));
    else
        farmertodaoptions.nSigmas=sqrt(znum-1);
    end
    farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    farmertodaoptions.verbose=1;
else
    if ~isfield(farmertodaoptions,'method')
        farmertodaoptions.method='even';
    end
    if ~isfield(farmertodaoptions,'nMoments')
        farmertodaoptions.nMoments=4;
    end
    if ~isfield(farmertodaoptions,'nSigmas')
        if maxabseig <= 1-2/(znum-1)
            farmertodaoptions.nSigmas=sqrt(2*(znum-1));
        else
            farmertodaoptions.nSigmas=sqrt(znum-1);
        end
    end
    if ~isfield(farmertodaoptions,'parallel')
        farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(farmertodaoptions,'verbose')
        farmertodaoptions.verbose=1;
    end
end

%% Check for a user-supplied grid
if isfield(farmertodaoptions,'z_grid')
    farmertodaoptions.usergrid=1;
    % Must be on the cpu: the entropy problem is solved with fminunc(), which cannot take gpuArrays.
    % Same reasoning as discretizeAR1_FarmerToda, and it matters for the same reason - this command
    % returns a gpuArray by default, so handing back the grid you were just given would fail.
    farmertodaoptions.z_grid=gather(farmertodaoptions.z_grid);
    if size(farmertodaoptions.z_grid,1)>1
        farmertodaoptions.z_grid=farmertodaoptions.z_grid'; % use a row internally
    end
    if length(farmertodaoptions.z_grid)~=znum
        error('length of farmertodaoptions.z_grid must equal znum: it is the SHARED one-dimensional grid, not the stacked p*znum output grid')
    end
else
    farmertodaoptions.usergrid=0;
end

%% Check inputs are correctly formatted
if ~isnumeric(znum) || znum<3 || rem(znum,1)~=0
    error('znum must be a positive integer greater than 3')
end
if ~isnumeric(farmertodaoptions.nMoments) || farmertodaoptions.nMoments<1 || farmertodaoptions.nMoments>4 || ~((rem(farmertodaoptions.nMoments,1)==0) || (farmertodaoptions.nMoments==1))
    error('farmertodaoptions.nMoments must be either 1, 2, 3, 4')
end
if ~strcmp(farmertodaoptions.method,'even') && ~strcmp(farmertodaoptions.method,'gauss-legendre') && ~strcmp(farmertodaoptions.method,'clenshaw-curtis') && ~strcmp(farmertodaoptions.method,'gauss-hermite') && ~strcmp(farmertodaoptions.method,'GMQ')
    error('farmertodaoptions.method must be one of even, gauss-legendre, clenshaw-curtis, gauss-hermite, or GMQ')
end
if farmertodaoptions.nSigmas<1.2
    warning('Trying to hit the 2nd moment with farmertodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for farmertodaoptions.nSigmas<1.2).')
end

%% The conditional moments the Farmer-Toda method targets
% z' given the state is condMean plus the innovation, so the central moments of z' about condMean
% are the (uncentered) moments of the mixture.
sigmaC2=sigma_i.^2;
T1=mixprobs_i'*mu_i; % mean
T2=mixprobs_i'*(mu_i.^2+sigmaC2); % uncentered second moment
T3=mixprobs_i'*(mu_i.^3+3*mu_i.*sigmaC2); % uncentered third moment
T4=mixprobs_i'*(mu_i.^4+6*(mu_i.^2).*sigmaC2+3*sigmaC2.^2); % uncentered fourth moment
TBar=[T1 T2 T3 T4]';

% gmdistribution and pdf() need the Statistics Toolbox, and a gaussian mixture density is one line
% without them: sum_i p_i*normpdf(x,mu_i,s_i). The component parameters are kept as plain vectors
% and the density is written out at each use below.
gmmu=mu_i; gmsd=sigma_i; gmp=mixprobs_i;

sigma=sqrt(T2-T1^2); % conditional standard deviation, i.e. the standard deviation of the mixture

%% Unconditional moments, from the companion form
% The grid is centred on mew/(1-sum(Rho)) rather than on E(z)=(mew+E(e))/(1-sum(Rho)), matching
% discretizeAR1wGM_FarmerToda: the mixture is not required to be mean zero, and when it is not the
% two differ. otheroutputs.mewz below reports the grid centre, and otheroutputs.Ez the actual mean.
mewz=mew/(1-sum(Rho)); % the grid centre
Ez=(mew+T1)/(1-sum(Rho)); % the unconditional mean of z
Q=zeros(p,p); Q(1,1)=sigma^2;
Sigma=reshape((eye(p^2)-kron(F,F))\Q(:),[p,p]); % vec(Sigma)=(I-kron(F,F))\vec(Q)
sigmaz=sqrt(Sigma(1,1)); % unconditional standard deviation of z

%% The shared one-dimensional grid
if farmertodaoptions.usergrid==1
    X1=farmertodaoptions.z_grid; % row vector
    W=ones(1,znum); % treat like 'even' for the prior q in moment matching
else
    switch farmertodaoptions.method
        case 'even'
            X1=linspace(mewz-farmertodaoptions.nSigmas*sigmaz,mewz+farmertodaoptions.nSigmas*sigmaz,znum);
            W=ones(1,znum);
        case 'gauss-legendre'
            [X1,W]=legpts(znum,[mewz-farmertodaoptions.nSigmas*sigmaz,mewz+farmertodaoptions.nSigmas*sigmaz]);
            X1=X1';
        case 'clenshaw-curtis'
            [X1,W]=fclencurt(znum,mewz-farmertodaoptions.nSigmas*sigmaz,mewz+farmertodaoptions.nSigmas*sigmaz);
            X1=fliplr(X1');
            W=fliplr(W');
        case 'gauss-hermite'
            if maxabseig>0.8 && farmertodaoptions.verbose==1
                warning('Model is persistent; even-spaced grid is recommended')
            end
            [X1,W]=GaussHermite(znum);
            X1=mewz+sqrt(2)*sigma*X1';
            W=W'./sqrt(pi);
        case 'GMQ' % Gaussian Mixture Quadrature
            if maxabseig>0.8 && farmertodaoptions.verbose==1
                warning('Model is persistent; even-spaced grid is recommended')
            end
            [X1,W]=GaussianMixtureQuadrature(mixprobs_i,mu_i,sigma_i,znum);
            X1=X1+mewz;
    end
end

%% Farmer-Toda, over the znum^p joint states
Nz=znum^p;
z_grid=repmat(X1',p,1); % stacked: p identical copies of the shared grid
% zlagvals(ii,:) is the p lag values in joint state ii, with the FIRST dimension varying fastest.
% CreateGridvals is used rather than an index formula so this command agrees with the rest of the
% toolkit about the joint ordering by construction rather than by assertion.
zlagvals=CreateGridvals(znum*ones(p,1),z_grid,1);
condMeans=mew+zlagvals*Rho'; % (Nz-by-1) conditional mean of z' from each joint state

pi_z=zeros(Nz,Nz);
nMoments_grid=zeros(Nz,1);
scalingFactor=max(abs(X1));
kappa=1e-8;
shiftblock=znum^(p-1);

for ii=1:Nz

    condMean=condMeans(ii);
    % The prior. gauss-hermite and GMQ carry quadrature weights that already encode a density, so
    % they are used as-is; every other grid gets the mixture density evaluated at the grid points,
    % shifted by the conditional mean. A user grid takes the 'even' treatment, which is the right
    % translation because 'even' is the one built-in method whose weights are not tied to its nodes.
    if (strcmp(farmertodaoptions.method,'gauss-hermite') || strcmp(farmertodaoptions.method,'GMQ')) && farmertodaoptions.usergrid==0
        q=W;
    else
        q=W.*(sum(gmp'.*exp(-0.5*(((X1'-condMean)-gmmu')./gmsd').^2)./(gmsd'*sqrt(2*pi)),2))';
    end
    if any(q<kappa)
        q(q<kappa)=kappa; % replace by small number for numerical stability
    end

    if farmertodaoptions.nMoments==1 % match only 1 moment
        pvec=discreteApproximation(X1,@(x)(x-condMean)/scalingFactor,TBar(1)./scalingFactor,q,0);
        nMoments_grid(ii)=1;
    else % match 2 moments first
        [pvec,lambda,momentError]=discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
            ((x-condMean)./scalingFactor).^2],...
            TBar(1:2)./(scalingFactor.^(1:2)'),q,zeros(2,1));
        if norm(momentError)>1e-5 % if 2 moments fail, then just match 1 moment
            pvec=discreteApproximation(X1,@(x)(x-condMean)/scalingFactor,0,q,0);
            nMoments_grid(ii)=1;
        elseif farmertodaoptions.nMoments==2
            nMoments_grid(ii)=2;
        elseif farmertodaoptions.nMoments==3 % 3 moments
            [pnew,~,momentError]=discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
            if norm(momentError)>1e-5
                nMoments_grid(ii)=2;
            else
                pvec=pnew;
                nMoments_grid(ii)=3;
            end
        elseif farmertodaoptions.nMoments==4 % 4 moments
            [pnew,~,momentError]=discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
                ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3;...
                ((x-condMean)./scalingFactor).^4],TBar./(scalingFactor.^(1:4)'),q,[lambda;0;0]);
            if norm(momentError)>1e-5
                [pnew,~,momentError]=discreteApproximation(X1,@(x) [(x-condMean)./scalingFactor;...
                    ((x-condMean)./scalingFactor).^2;((x-condMean)./scalingFactor).^3],...
                    TBar(1:3)./(scalingFactor.^(1:3)'),q,[lambda;0]);
                if norm(momentError)>1e-5
                    nMoments_grid(ii)=2;
                else
                    pvec=pnew;
                    nMoments_grid(ii)=3;
                end
            else
                pvec=pnew;
                nMoments_grid(ii)=4;
            end
        end
    end

    % SCATTER, rather than fill a dense row. From joint state ii=(i1,...,ip) the only reachable
    % states are (j,i1,...,i(p-1)) for j=1..znum: the new value of z is drawn, and every lag moves
    % along one place. With the first dimension varying fastest, ii-1 = (i1-1)+znum*(i2-1)+... , so
    % dropping the last lag and shifting the rest up one place is mod(ii-1,znum^(p-1)) multiplied by
    % znum. At p=1 that is zero and the row is dense, which is the AR(1) case.
    destbase=znum*mod(ii-1,shiftblock);
    pi_z(ii,destbase+(1:znum))=pvec;
end

% Report the maximum entropy fallbacks once for the whole call, rather than once per grid point.
if farmertodaoptions.verbose==1 && sum(nMoments_grid<farmertodaoptions.nMoments)>0
    warning('Matched fewer than the requested %i moments from %i of the %i joint states, as few as %i. See otheroutputs.nMoments_grid for which.',farmertodaoptions.nMoments,sum(nMoments_grid<farmertodaoptions.nMoments),Nz,min(nMoments_grid))
end

if farmertodaoptions.parallel==2
    z_grid=gpuArray(z_grid);
    pi_z=gpuArray(pi_z);
end

%% Some additional outputs that can be used to evaluate the discretization
otheroutputs.nMoments_grid=nMoments_grid;
otheroutputs.mewz=mewz; % the grid centre, mew/(1-sum(Rho))
otheroutputs.Ez=Ez;     % the unconditional mean of z, which differs from mewz when the mixture is not mean zero
otheroutputs.sigmaz=sigmaz;
otheroutputs.maxabseig=maxabseig;

end
