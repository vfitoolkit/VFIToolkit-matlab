function [z_grid,pi_z,otheroutputs] = discretizeAR1wSV_FarmerToda(rho,phi,sigmau,sigmae,xnum,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments
%
%  Discretize an AR(1) process with log AR(1) stochastic volatility using Farmer-Toda method
%       z_t = rho*z_{t-1} + u_t
%       u_t ~ N(0,exp(x_t)); 
%       x_t = (1-phi)*mu + phi*x_{t-1} + epsilon_t
%       epsilon_t ~ N(0,sigma_e^2)
%
% Inputs:
%   rho       - persistence of z process
%   phi       - persistence of x process
%   sigmau    - unconditional standard deviation of u_t
%   sigmae    - standard deviation of epsilon_t
%   znum      - number of grid points for z process
%   xnum      - number of grid points for x process
% Optional inputs (farmertodaoptions):
%   method    - quadrature method for x process
%   nSigmas   - grid half-width for z, in units of sd(z) (default = sqrt(znum-1) )
%   verbose   - set to zero to suppress the report of how many grid points matched fewer moments
% Output: 
%   z_grid:   - stacked column vector, x on top, z below (so z_grid(1:xnum) is the grid on x, z_grid(xnum+1:end) is the grid on z)
%   pi_z:     - joint transition matrix on (x,z)
%     Note, the dimensions of the output are thus interpreted as [xnum,znum]
%   otheroutputs   - optional output structure containing info for evaluating the distribution including,
%        otheroutputs.nMoments_grid  - shows how many moments were matched from each grid point (for the conditional distribution)
%              Note, this is indexed like the rows of pi_z, so use reshape(otheroutputs.nMoments_grid,[xnum,znum]) to read it as (x,z)
%
% Useful info: E[z_t]=mu (constant divided by 1-autocorrelation coeff; that is advantage of writing constant as (1-phi*mu).)
%
% Note: nMoments is hard-coded as 2 for z (conditional moments to be matched by Farmer-Toda method)
% Note: method 'even' grid is hard-coded for z
% Note: z uses default nMoments (2), and follows method (default depend on phi, see discretizeAR1_FarmerToda)
%
% (c) 2016 Leland E. Farmer and Alexis Akira Toda (v1.2, 2019)
% This version was lightly modified by Robert Kirkby
%%%%%
% Original paper:
% Farmer & Toda (2017) - Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments


%% Set defaults
if ~exist('farmertodaoptions','var')
    % If farmertodaoptions.method is not declared then just leave it to discretizeAR1_FarmerToda
    farmertodaoptions.nSigmas = sqrt(znum-1); % grid half-width for z; see the note above
    farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    farmertodaoptions.verbose=1;
else
    % define grid spacing parameter if not provided (only used for 'even' method)
    if ~isfield(farmertodaoptions,'nSigmas')
        farmertodaoptions.nSigmas = sqrt(znum-1); % grid half-width for z; see the note above
    end
    if ~isfield(farmertodaoptions,'parallel')
        farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(farmertodaoptions,'verbose')
        farmertodaoptions.verbose=1;
    end
end
% farmertodaoptions.nMoments = 2; % This could be used to change nMoments for x (to 1,2,3 or 4; is set to default of 2 by discretizeAR1_FarmerToda)

if farmertodaoptions.nSigmas<1.2
    warning('Trying to hit the 2nd moment with farmertodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for farmertodaoptions.nSigmas<1.2).')
end


% The grid half-width for z, in units of its unconditional standard deviation. This used to be
% min(sqrt((znum-1)/2),2), the narrowest default in the toolkit, and at that width the excess
% kurtosis of z came out NEGATIVE against a positive truth - the wrong sign for the one moment a
% stochastic-volatility process exists to produce, because the tails that carry it lie outside a
% two-sigma grid. Measured on a rho=0.95, phi=0.9 calibration at xnum=9, reading only the cells
% where the maximum entropy solve does not fall back, the kurtosis error improves with width and
% saturates by about seven sigma: 0.312 at width 4, 0.257 at 5, 0.247 at 7, 0.246 at 10, against
% 1.39 at the old default. sqrt(znum-1) lands on that saturated value for znum>=31 and stays out
% of the high-fallback region at small znum.
%
% WHY THE EXCESS KURTOSIS NEVER REACHES THE TRUTH, AND WHY NO GRID FIXES IT. The remaining 0.246
% is NOT a width problem. It is the product form of the transition built below,
%    pi_z(ii,:) = kron(p,ones(1,xnum));  pi_z(ii,:) = pi_z(ii,:).*repmat(Px(...),1,znum);
% which makes z' and x' conditionally independent given (z,x). The model has u' ~ N(0,exp(x')),
% the NEW volatility, so this construction has to integrate x' out and use
%    Var(z'|z,x) = E[exp(x')|x] = exp((1-phi)*xBar+phi*x+sigmae^2/2)
% in its place. Conditional on the whole volatility path z is gaussian with V=sum_k rho^(2k)*s_k,
% so the excess kurtosis is 3*E[V^2]/E[V]^2-3, and substituting s(x)=E[exp(x')|x] for exp(x)
% turns the lag-h autocovariance of log volatility from sigmaX*phi^h into sigmaX*phi^(h+2). The
% construction loses exactly two lags of volatility persistence.
%
% That is a CEILING, not a grid error: it is what the method converges TO as both grids refine.
% On the rho=0.95, phi=0.9, sigmau=0.2, sigmae=0.3 calibration of P4,
%    truth                                                    0.8285
%    product-form ceiling (exact x grid, exact z grid)         0.6498
%    the same with the x block capped at nSigmas=2, as below  ~0.567
%    measured at xnum=znum=31, 51, 101 (no fall-back rows)     0.279, 0.267, 0.256
% The mean survives the substitution - sigmae^2/2+phi^2*sigmaX/2 = sigmaX/2 exactly - so
% E[s]=exp(xBar+sigmaX/2) and Var(z)=E[s]/(1-rho^2) stay right to four digits (0.40873 predicted
% at xnum=31 against 0.4087 measured). Variance passing while kurtosis fails is the signature.
%
% Two consequences. First, xnum does almost nothing, because farmertodaoptions_x.nSigmas is
% hard-coded to 2 below: the volatility grid is +-2 unconditional sd of log volatility however
% many x points are asked for. Farmer-Toda still matches x's mean and variance exactly on that
% truncated grid (P4 measures the Var(x) error at 8e-10 for every xnum), but E[exp(2x)] - the
% tail functional that kurtosis needs - stays short. Across xnum=5 to 51 at znum=15 the measured
% kurtosis moves 0.3351 to 0.3368; a twentyfold refinement buys 0.002.
%
% Second, an earlier version of this comment blamed nMoments being hard-coded at 2 for the z
% block, and proposed matching m4 = 3*E[exp(2x')|x] = 3*exp(2*((1-phi)*xBar+phi*x)+2*sigmae^2)
% rather than the gaussian 3*m2^2, which is short by exp(sigmae^2). That target is correctly
% derived, and the warning attached to it stands: the naive 3*m2^2 would look like an improvement
% while suppressing the conditional excess kurtosis that is the whole point of the process. But
% it is worth 0.021 - it lifts the ceiling from 0.6498 to 0.6711 and no further - because
% matching conditional moments of z' given x cannot restore the dependence between the size of
% the z' innovation and x' that the product form threw away.
%
% What would actually reach 0.8285 is conditioning the z innovation on x' instead of on
% E[exp(x')|x]: transition Px(x,x')*p(z'|z,x') with variance target exp(x'), which restores
% V=sum_k rho^(2k)*exp(x_(t-k)) exactly, at the same number of entropy solves. That is a
% departure from the published Farmer-Toda construction, so it belongs behind an option rather
% than as a silent change. NOT IMPLEMENTED.
%
% Still unexplained: the fall from the ~0.567 ceiling to the 0.256 measured at znum=101. It is
% not the fall-back rows, since P4 reports zero of them at znum>=31, so it is something in the
% z-block entropy fit itself. Measured in P4 of the DiscretizationMethodTests test bank; the
% ceiling arithmetic above is exact and was cross-checked against P4's own Var(z).

%% Compute some unconditional moments

sigmaX = (sigmae^2)/(1-phi^2); % unconditional variance of variance process
xBar = 2*log(sigmau)-sigmaX/2; % unconditional mean of variance process, targeted to match a mean standard deviation of sigmaU
sigmaz = sqrt(exp(xBar+sigmaX/2)/(1-rho^2)); % unconditional standard deviation of technology shock

%% Construct technology process approximation
farmertodaoptions_x=farmertodaoptions;
farmertodaoptions_x.nSigmas=2; % the x (volatility) block is deliberately discretized as nSigmas=2
[x_grid,Px] = discretizeVAR1_FarmerToda(xBar*(1-phi),phi,sigmae^2,xnum,farmertodaoptions_x);
x_grid=gather(x_grid); Px=gather(Px); % the z block below is a cpu entropy solve (fminunc), so the x outputs have to come back from the gpu
% [Px,x_grid] = discreteVAR(xBar*(1-phi),phi,sigmae^2,xnum,2,farmertodaoptions.method); % discretization of variance process


z_grid = linspace(-farmertodaoptions.nSigmas*sigmaz,farmertodaoptions.nSigmas*sigmaz,znum);

Nm = xnum*znum; % total number of state variable pairs
%zxGrids = flipud(combvec(xGrid,zGrid))';
temp1 = repmat(x_grid',1,znum);
temp2 = kron(z_grid,ones(1,xnum));

zx_grid = flipud([temp1; temp2])'; % avoid using combvec, which requires deep learning toolbox
pi_z = zeros(Nm);
lambdaGuess = zeros(2,1);
nMoments_grid=zeros(Nm,1); % Used to record number of moments matched in transition from each point
scalingFactor = max(abs(z_grid));
kappa = 1e-8; % small positive constant for numerical stability

for ii = 1:Nm
    
    q = exp(-0.5*((z_grid-rho*zx_grid(ii,1))./(sqrt(exp((1-phi)*xBar+phi*zx_grid(ii,2)+(sigmae^2)/2)))).^2)./((sqrt(exp((1-phi)*xBar+phi*zx_grid(ii,2)+(sigmae^2)/2)))*sqrt(2*pi));
    if sum(q<kappa) > 0
        q(q<kappa) = kappa;
    end
    [p,~,momentError] = discreteApproximation(z_grid,@(X) [(X-rho*zx_grid(ii,1))./scalingFactor; ((X-rho*zx_grid(ii,1))./scalingFactor).^2],[0; (exp((1-phi)*xBar+phi*zx_grid(ii,2)+(sigmae^2)/2))./(scalingFactor^2)],q,lambdaGuess);
    % If trying to match two conditional moments fails, just match the conditional mean
    if norm(momentError) > 1e-5
        p = discreteApproximation(z_grid,@(X) (X-rho*zx_grid(ii,1))./scalingFactor,0,q,0);
        nMoments_grid(ii)=1;
    else
        nMoments_grid(ii)=2;
    end
    pi_z(ii,:) = kron(p,ones(1,xnum));
    pi_z(ii,:) = pi_z(ii,:).*repmat(Px(mod(ii-1,xnum)+1,:),1,znum);
 
end

if farmertodaoptions.verbose==1 && sum(nMoments_grid==1)>0
    warning('Failed to match first 2 moments from %i of the %i grid points (just matched 1 from those). See otheroutputs.nMoments_grid for which.',sum(nMoments_grid==1),Nm)
end

otheroutputs.nMoments_grid=nMoments_grid; % How many moments were hit by the conditional distribution from each grid point

% Original Farmer-Toda code output zx_grid.
% I instead output a stacked vector.
z_grid=[x_grid; z_grid'];

%%

end
