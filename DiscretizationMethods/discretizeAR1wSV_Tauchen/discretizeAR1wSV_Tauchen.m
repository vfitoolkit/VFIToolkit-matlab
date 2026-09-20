function [z_grid,pi_z]=discretizeAR1wSV_Tauchen(rho,phi,sigmau,sigmae,xnum,znum,Tauchen_q, tauchenoptions)
% Discretize an AR(1) process with log AR(1) stochastic volatility using the Tauchen method
%       z_t = rho*z_{t-1} + u_t
%       u_t ~ N(0,exp(x_t));
%       x_t = (1-phi)*xBar + phi*x_{t-1} + epsilon_t
%       epsilon_t ~ N(0,sigma_e^2)
%
% Inputs:
%   rho       - persistence of z process
%   phi       - persistence of x process
%   sigmau    - unconditional standard deviation of u_t
%   sigmae    - standard deviation of epsilon_t
%   xnum      - number of grid points for x process
%   znum      - number of grid points for z process
%   Tauchen_q - (Hyperparameter) Defines max/min grid points as +-Tauchen_q*(standard deviation),
%               Set Tauchen_q=[] to use the default of min(sqrt(znum-1),4), as in
%               discretizeAR1_Tauchen. The cap at 4 is measured: past it this method's excess
%               kurtosis error rises sharply (0.087 at width 4, 1.15 at 7, 2.56 at 10, at znum=101),
%               because extra width is paid for in grid spacing.
%               used for both the x and the z grids (I suggest 2 or 3)
% Optional inputs (tauchenoptions):
%   parallel: - set equal to 2 to use GPU, 0 to use CPU
% Output:
%   z_grid:   - stacked column vector, x on top, z below (so z_grid(1:xnum) is the grid on x,
%               z_grid(xnum+1:end) is the grid on z)
%   pi_z:     - joint transition matrix on (x,z)
%     Note, the dimensions of the output are thus interpreted as [xnum,znum], with x varying fastest
%
% Useful info: xBar is set so that the unconditional standard deviation of u_t is sigmau, namely
%   sigmaX=sigma_e^2/(1-phi^2) is the unconditional variance of x, and then
%   xBar=2*log(sigmau)-sigmaX/2 gives E[exp(x)]=exp(xBar+sigmaX/2)=sigmau^2.
%   E[z]=0, Var(z)=sigmau^2/(1-rho^2), and the autocorrelation of z is rho.
%   Note z has EXCESS KURTOSIS even though every conditional distribution is normal; that is what
%   stochastic volatility is for.
%
% Note: the z transition here is conditioned on the REALIZED next-period volatility exp(x_t), which
%   is the exact conditional law: given (x_{t-1},z_{t-1}) and a realized x_t, z_t is
%   N(rho*z_{t-1},exp(x_t)). discretizeAR1wSV_FarmerToda instead conditions on the EXPECTED
%   next-period volatility E[exp(x_t)|x_{t-1}], integrating x_t out of the variance while still
%   tracking it in the state. Both are defensible discretizations and they agree as sigmae->0, but
%   they are not the same object, so do not expect the two commands to reproduce each other.
%%%%%%%%%%%%%%%
% Original paper (for the Tauchen method itself):
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"
% The stochastic-volatility version discretized by Farmer-Toda is:
% Farmer & Toda (2017) - Discretizing Nonlinear, Non-Gaussian Markov Processes with Exact Conditional Moments

%%

if exist('tauchenoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
    tauchenoptions.verbose=1;
else
    %Check tauchenoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(tauchenoptions,'parallel')
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(tauchenoptions,'verbose')
        tauchenoptions.verbose=1;
    end
end

% The recommendation above is printed once per call, and the test bank that exercises these
% commands calls them hundreds of times, so it is gated. Set tauchenoptions.verbose=0 to silence it.
if tauchenoptions.verbose==1
    fprintf('COMMENT: The Tauchen method is likely inferior to the Farmer-Toda method for discretizing AR(1) processes with stochastic volatility. \n')
    fprintf('         It is suggested you consider using discretizeAR1wSV_FarmerToda instead. \n')
end

%% Check inputs are correctly formatted
if abs(rho)>=1
    error('Tauchen error, autocorrelation of z (rho) is >=1 in absolute value. You cannot discretize an AR(1) with an autocorrelation coefficient of >=1')
end
if abs(phi)>=1
    error('Tauchen error, autocorrelation of the log-volatility process (phi) is >=1 in absolute value')
end
if sigmau<=0
    error('the unconditional standard deviation of u (sigmau) must be strictly positive')
end
if sigmae<=0
    error('the standard deviation of the innovations to the log-volatility process (sigmae) must be strictly positive')
end

%% Set up the two processes
sigmaX=(sigmae^2)/(1-phi^2); % unconditional variance of the log-volatility process
xBar=2*log(sigmau)-sigmaX/2; % unconditional mean of the log-volatility process, targeted to match a mean standard deviation of sigmau
sigmaz=sqrt(exp(xBar+sigmaX/2)/(1-rho^2)); % unconditional standard deviation of z (note: exp(xBar+sigmaX/2)=sigmau^2)

% The log-volatility process is an ordinary gaussian AR(1), so discretize it with the ordinary
% Tauchen method. Its intercept is xBar*(1-phi), so that E(x)=xBar*(1-phi)/(1-phi)=xBar.
tauchenoptions_x=struct();
tauchenoptions_x.parallel=1; % need it on the cpu here; the joint matrix is moved to gpu at the end if needed
% Tauchen_q=[] means use the default width
% min(sqrt(znum-1),4), matching discretizeAR1_Tauchen. Unlike discretizeAR1wSV_FarmerToda, which
% wants as much width as its solve can take, this method pays for width in grid spacing: at
% znum=101 the excess kurtosis error is 0.087 at width 4 but 1.15 at 7 and 2.56 at 10.
if isempty(Tauchen_q)
    Tauchen_q=min(sqrt(znum-1),4);
end

[x_grid,Px]=discretizeAR1_Tauchen(xBar*(1-phi),phi,sigmae,xnum,Tauchen_q,tauchenoptions_x);

z_grid=linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)'; % E(z)=0, so the grid is centred on zero
Nm=xnum*znum; % total number of state variable pairs

if znum==1
    error('znum must be at least 2 (a one point grid on z leaves no bins to assign probability to)')
end
omega=z_grid(2)-z_grid(1); % all the points are equidistant by construction
upper=z_grid+omega/2;
lower=z_grid-omega/2;

%% Build the joint transition matrix
% pi_z(ii,jj) with ii=(x_i,z_i) and jj=(x_j,z_j), x varying fastest in both.
%   pi_z(ii,jj) = Px(x_i,x_j) * Pz_{x_j}(z_i,z_j)
% where Pz_{x_j} is the Tauchen transition on z when the realized volatility is exp(x_j). The x
% transition and the z transition are conditionally independent given x_j, which is what makes this
% factorization exact rather than an approximation.
zi=z_grid*ones(1,znum);
upperj=ones(znum,1)*upper';
lowerj=ones(znum,1)*lower';

if tauchenoptions.parallel==0 || tauchenoptions.parallel==1
    pi_z=zeros(Nm,Nm);
    for xj_c=1:xnum
        sigmau_xj=sqrt(exp(x_grid(xj_c))); % the realized standard deviation of u in state x_j

        P_part1=0.5*erfc(-(upperj-rho*zi)./(sigmau_xj*sqrt(2)));
        P_part2=0.5*erfc(-(lowerj-rho*zi)./(sigmau_xj*sqrt(2)));
        Pz=P_part1-P_part2;
        Pz(:,1)=P_part1(:,1);
        Pz(:,znum)=1-P_part2(:,znum);

        % kron(Pz,Px(:,xj_c)) is indexed by (z_i,x_i) down the rows with x_i fastest, which is
        % exactly the ii ordering, and by z_j across the columns.
        pi_z(:,xj_c+((1:znum)-1)*xnum)=kron(Pz,Px(:,xj_c));
    end

elseif tauchenoptions.parallel==2 %Parallelize on GPU
    % Same erfc expression as the cpu branch above, so the two differ only where the cpu and gpu
    % erfc libraries disagree in the last bit. erfc not 1+erf: the left tail cdf is tiny, and
    % 1+erf loses all relative precision there (it is exactly zero past about -8.3 sd).
    z_grid=gpuArray(z_grid);
    upper=gpuArray(upper);
    lower=gpuArray(lower);
    Px=gpuArray(Px);
    pi_z=zeros(Nm,Nm,'gpuArray');
    for xj_c=1:xnum
        sigmau_xj=sqrt(exp(x_grid(xj_c)));

        erfcinput=arrayfun(@(zi,zj,rho,sigmai) -(zj-rho*zi)/(sigmai*sqrt(2)), z_grid,upper', rho,sigmau_xj);
        P_part1=0.5*erfc(erfcinput);

        erfcinput=arrayfun(@(zi,zj,rho,sigmai) -(zj-rho*zi)/(sigmai*sqrt(2)), z_grid,lower', rho,sigmau_xj);
        P_part2=0.5*erfc(erfcinput);

        Pz=P_part1-P_part2;
        Pz(:,1)=P_part1(:,1);
        Pz(:,znum)=1-P_part2(:,znum);

        pi_z(:,xj_c+((1:znum)-1)*xnum)=kron(Pz,Px(:,xj_c));
    end
    x_grid=gpuArray(x_grid);
end

%% Stack the output, x on top, z below
z_grid=[x_grid; z_grid];

end
