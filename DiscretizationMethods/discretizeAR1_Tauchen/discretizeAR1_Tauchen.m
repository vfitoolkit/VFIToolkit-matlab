function [z_grid,pi_z]=discretizeAR1_Tauchen(mew,rho,sigma,znum,Tauchen_q, tauchenoptions)
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation
%    of AR(1) process z'=mew+rho*z+e, e~N(0,sigma^2), by Tauchen method
%
% Inputs
%   mew            - constant term coefficient
%   rho            - autocorrelation coefficient
%   sigma          - standard deviation of (gaussian) innovations
%   znum           - number of states in discretization of z (must be an odd number)
%   Tauchen_q      - (Hyperparameter) Defines max/min grid points as mew+-Tauchen_q*sigmaz (I suggest 2 or 3)
%                    Set Tauchen_q=[] to use the default of min(sqrt(znum-1),4). The sqrt(znum-1)
%                    part is the width discretizeAR1_Rouwenhorst requires and discretizeAR1_FarmerToda
%                    defaults to; the cap at 4 is because Tauchen, unlike those two, pays for extra
%                    width in grid spacing and so in the variance. Raise it yourself if the
%                    autocorrelation matters more to you than the variance.
% Optional Inputs (tauchenoptions)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
%   dshift:        - allows approximating 'trend-reverting' process around a deterministic trend (not part of standard Tauchen method)
%   z_grid:        - (znum-by-1) skip grid construction and build pi_z on this grid;
%                    Tauchen_q is ignored, bin edges are midpoints between adjacent grid points
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   pi_z           - transition matrix of the discrete approximation of z;
%                    pi_z(i,j) is the probability of transitioning from state i to state j
%
% Helpful info:
%   Var(z)=(sigma^2)/(1-rho^2). So sigmaz=sigma/sqrt(1-rho^2);   sigma=sigmaz*sqrt(1-rho^2)
%                                  where sigmaz= standard deviation of z
%     E(z)=mew/(1-rho)
%%%%%%%%%%%%%%%
% Original paper:
% Tauchen (1986) - "Finite state Markov-chain approximations to univariate and vector autoregressions"

if exist('tauchenoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=1+(gpuDeviceCount>0);
else
    %Check tauchenoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(tauchenoptions,'parallel')
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
end

% Check for a deterministic shifter
if ~isfield(tauchenoptions,'dshift')
    tauchenoptions.dshift=0;
end

% Tauchen_q=[] means use the default width
% sqrt(znum-1) is the half-width the Rouwenhorst construction requires and that
% discretizeAR1_FarmerToda takes as its default. Tauchen has no such requirement - the width here
% is a free hyperparameter - and measurement says an uncapped sqrt(znum-1) is too wide at large
% znum. Past the truncation regime Tauchen's variance error is bin-width smearing, h^2/12 with h
% the grid spacing, and a width growing like sqrt(znum-1) holds the spacing at 2/sqrt(znum-1) so
% the variance stops improving. Sweeping width against znum at rho=0.6, the width that minimises
% the worse of the variance and autocorrelation errors is 2.5, 3, 3, 4, 4, 4 for znum=5 to 101 -
% it settles rather than keeps growing - and min(sqrt(znum-1),4) tracks that.
%
% The cap is a trade-off, not a free win. The two moments disagree: the variance wants a width
% near 4, the autocorrelation improves monotonically with width and is ten orders of magnitude
% better at width 10 than at 4 once znum>=31. If autocorrelation is what you need to get right,
% pass Tauchen_q yourself and make it larger than this default.
if isempty(Tauchen_q)
    Tauchen_q=min(sqrt(znum-1),4);
end

%% Check for user-supplied grid
if isfield(tauchenoptions,'z_grid')
    tauchenoptions.usergrid=1;
    if size(tauchenoptions.z_grid,2)>1
        tauchenoptions.z_grid=tauchenoptions.z_grid'; % use column internally
    end
    if length(tauchenoptions.z_grid)~=znum
        error('length of tauchenoptions.z_grid must equal znum')
    end
else
    tauchenoptions.usergrid=0;
end

if znum==1
    if tauchenoptions.usergrid==1
        z_grid=tauchenoptions.z_grid;
    else
        z_grid=mew/(1-rho); %expected value of z
    end
    pi_z=1;
    if tauchenoptions.parallel==2
        z_grid=gpuArray(z_grid);
        pi_z=gpuArray(pi_z);
    end
    return
end

% Note: tauchenoptions.dshift equals zero gives the Tauchen method.
% For nonzero tauchenoptions.dshift this is actually implementing a non-standard Tauchen method.
if tauchenoptions.parallel==0 || tauchenoptions.parallel==1
    if tauchenoptions.usergrid==1
        z_grid=tauchenoptions.z_grid;
        edges=(z_grid(1:end-1)+z_grid(2:end))/2;
        upper=[edges; z_grid(end)]; % last entry overwritten below (extends to +inf)
        lower=[z_grid(1); edges];   % first entry overwritten below (extends to -inf)
    else
        zstar=mew/(1-rho); %expected value of z
        sigmaz=sigma/sqrt(1-rho^2); %stddev of z
        z_grid=zstar*ones(znum,1) + linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)';
        omega=z_grid(2)-z_grid(1); %Note that all the points are equidistant by construction.
        upper=z_grid+omega/2;
        lower=z_grid-omega/2;
    end

    zi=z_grid*ones(1,znum);
    upperj=tauchenoptions.dshift*ones(znum,znum)+ones(znum,1)*upper';
    lowerj=tauchenoptions.dshift*ones(znum,znum)+ones(znum,1)*lower';

    P_part1=0.5*erfc(-((upperj-rho*zi)-mew)./(sigma*sqrt(2)));
    P_part2=0.5*erfc(-((lowerj-rho*zi)-mew)./(sigma*sqrt(2)));

    pi_z=P_part1-P_part2;
    pi_z(:,1)=P_part1(:,1);
    pi_z(:,znum)=1-P_part2(:,znum);

elseif tauchenoptions.parallel==2 %Parallelize on GPU
    if tauchenoptions.usergrid==1
        z_grid=gpuArray(tauchenoptions.z_grid);
        edges=(z_grid(1:end-1)+z_grid(2:end))/2;
        upper=[edges; z_grid(end)]; % last entry overwritten below (extends to +inf)
        lower=[z_grid(1); edges];   % first entry overwritten below (extends to -inf)
    else
        zstar=mew/(1-rho); %expected value of z
        sigmaz=sigma/sqrt(1-rho^2); %stddev of z
        z_grid=gpuArray(zstar*ones(znum,1) + linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)');
        omega=z_grid(2)-z_grid(1); %Note that all the points are equidistant by construction.
        upper=z_grid+omega/2;
        lower=z_grid-omega/2;
    end

    % Same erfc expression as the cpu branch above, so the two differ only where the cpu and gpu
    % erfc libraries disagree in the last bit. erfc not 1+erf: the left tail cdf is tiny, and
    % 1+erf loses all relative precision there (it is exactly zero past about -8.3 sd).

    tauchenoptions.dshift=gpuArray(tauchenoptions.dshift*ones(1,znum));

    erfcinput=arrayfun(@(zi,zj,rho,mew,sigma) -((zj-rho*zi)-mew)/(sigma*sqrt(2)), z_grid,tauchenoptions.dshift+upper', rho,mew,sigma);
    P_part1=0.5*erfc(erfcinput);

    erfcinput=arrayfun(@(zi,zj,rho,mew,sigma) -((zj-rho*zi)-mew)/(sigma*sqrt(2)), z_grid,tauchenoptions.dshift+lower', rho,mew,sigma);
    P_part2=0.5*erfc(erfcinput);

    pi_z=P_part1-P_part2;
    pi_z(:,1)=P_part1(:,1);
    pi_z(:,znum)=1-P_part2(:,znum);

end

end