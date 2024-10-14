function [z_grid,P] = discretizeAR1wSV_FarmerToda(rho,phi,sigmau,sigmae,xnum,znum,farmertodaoptions)
% Please cite: Farmer & Toda (2017) - "Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments
%
%  Discretize an AR(1) process with log AR(1) stochastic volatility using Farmer-Toda method
%       z_t = rho*z_{t-1} + u_t
%       u_t ~ N(0,exp(x_t)); 
%       x_t = (1-phi)*mu + phi*x_{t-1} + epsilon_t
%       epsilon_t ~ N(0,sigma_e^2)
%
% Usage:
%       [z_grid,P] = discretizeAR1wSV_FarmerToda(rho,phi,sigmau,sigmae,znum,xnum)
%
% Inputs:
%   rho       - persistence of z process
%   phi       - persistence of x process
%   sigmau    - unconditional standard deviation of u_t
%   sigmae    - standard deviation of epsilon_t
%   znum      - number of grid points for z process
%   xnum      - number of grid points for x process
% Optional nputs (farmertodaoptions):
%   method    - quadrature method for x process
%   nSigmas   - grid spacing parameter for z (default = sqrt((Nz-1)/2)
% Output: 
%   z_grid:   - stacked column vector, x on top, z below (so z_grid(1:xnum)is the grid on x, z_grid(xnum+1:end) is the grid on z)
%   P:        - joint transition matrix on (x,z)
%     Note, the dimensions of the output are thus interpreted as [xnum,znum]
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
    farmertodaoptions.nSigmas = min(sqrt((znum-1)/2),2); % spacing parameter for z process
    farmertodaoptions.parallel=1+(gpuDeviceCount>0);
else
    % define grid spacing parameter if not provided (only used for 'even' method)
    if ~isfield(farmertodaoptions,'nSigmas')
        farmertodaoptions.nSigmas = min(sqrt((znum-1)/2),2); % spacing parameter for z process
    end
    if ~isfield(farmertodaoptions,'parallel')
        farmertodaoptions.parallel=1+(gpuDeviceCount>0);
    end
end
% farmertodaoptions.nMoments = 2; % This could be used to change nMoments for x (to 1,2,3 or 4; is set to default of 2 by discretizeAR1_FarmerToda)

if farmertodaoptions.nSigmas<1.2
    warning('Trying to hit the 2nd moment with farmertodaoptions.nSigmas at 1 or less is odd. It will put lots of probability near edges of grid as you are trying to get the std dev, but you max grid points are only about plus/minus one std dev (warning shows for farmertodaoptions.nSigmas<1.2).')
end


%% Compute some uncondtional moments

sigmaX = (sigmae^2)/(1-phi^2); % unconditional variance of variance process
xBar = 2*log(sigmau)-sigmaX/2; % unconditional mean of variance process, targeted to match a mean standard deviation of sigmaU
sigmaz = sqrt(exp(xBar+sigmaX/2)/(1-rho^2)); % uncondtional standard deviation of technology shock

%% Construct technology process approximation
if farmertodaoptions.parallel==2
    farmertodaoptions.parallel=1;
    farmertodaoptions.nSigmas=2;
    [x_grid,Px] = discretizeVAR1_FarmerToda(xBar*(1-phi),phi,sigmae^2,xnum,farmertodaoptions);
    farmertodaoptions.parallel=2;
else
    farmertodaoptions.nSigmas=2;
    [x_grid,Px] = discretizeVAR1_FarmerToda(xBar*(1-phi),phi,sigmae^2,xnum,farmertodaoptions);
end
% [Px,x_grid] = discreteVAR(xBar*(1-phi),phi,sigmae^2,xnum,2,farmertodaoptions.method); % discretization of variance process


z_grid = linspace(-farmertodaoptions.nSigmas*sigmaz,farmertodaoptions.nSigmas*sigmaz,znum);

Nm = xnum*znum; % total number of state variable pairs
%zxGrids = flipud(combvec(xGrid,zGrid))';
temp1 = repmat(x_grid',1,znum);
temp2 = kron(z_grid,ones(1,xnum));

zx_grid = flipud([temp1; temp2])'; % avoid using combvec, which requires deep learning toolbox
P = zeros(Nm);
lambdaGuess = zeros(2,1);
scalingFactor = max(abs(z_grid));
kappa = 1e-8; % small positive constant for numerical stability

for ii = 1:Nm
    
    q = normpdf(z_grid,rho*zx_grid(ii,1),sqrt(exp((1-phi)*xBar+phi*zx_grid(ii,2)+(sigmae^2)/2)));
    if sum(q<kappa) > 0
        q(q<kappa) = kappa;
    end
    [p,~,momentError] = discreteApproximation(z_grid,@(X) [(X-rho*zx_grid(ii,1))./scalingFactor; ((X-rho*zx_grid(ii,1))./scalingFactor).^2],[0; (exp((1-phi)*xBar+phi*zx_grid(ii,2)+(sigmae^2)/2))./(scalingFactor^2)],q,lambdaGuess);
    % If trying to match two conditional moments fails, just match the conditional mean
    if norm(momentError) > 1e-5
        warning('Failed to match first 2 moments. Just matching 1.')
        p = discreteApproximation(z_grid,@(X) (X-rho*zx_grid(ii,1))./scalingFactor,0,q,0);
    end
    P(ii,:) = kron(p,ones(1,xnum));
    P(ii,:) = P(ii,:).*repmat(Px(mod(ii-1,xnum)+1,:),1,znum);
 
end
% zx_grid = zx_grid';
% Original Farmer-Toda code output zx_grid.
% I instead output a stacked vector.

z_grid=[x_grid; z_grid'];

%%

end
