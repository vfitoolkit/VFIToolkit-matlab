function [z_grid,P]=discretizeAR1_Tauchen(mew,rho,sigma,znum,Tauchen_q, tauchenoptions)
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation 
%    of AR(1) process z'=mew+rho*z+e, e~N(0,sigma^2), by Tauchen method
%
% Inputs
%   mew            - constant term coefficient
%   rho            - autocorrelation coefficient
%   sigma          - standard deviation of (gaussion) innovations
%   znum           - number of states in discretization of z (must be an odd number)
%   Tauchen_q      - (Hyperparameter) Defines max/min grid points as mew+-nSigmas*sigmaz (I suggest 2 or 3)
% Optional Inputs (tauchenoptions)
%   parallel:      - set equal to 2 to use GPU, 0 to use CPU
%   dshift:        - allows approximating 'trend-reverting' process around a deterministic trend (not part of standard Tauchen method)
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   P              - transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
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
    if isfield(tauchenoptions,'parallel')==0
        tauchenoptions.parallel=1+(gpuDeviceCount>0);
    end
end

% Check for a deterministic shifter
if exist('tauchenoptions.dshift','var')==0
    tauchenoptions.dshift=0;
end

if znum==1
    z_grid=mew/(1-rho); %expected value of z
    P=1;
    if tauchenoptions.parallel==2
        z_grid=gpuArray(z_grid);
        P=gpuArray(P);
    end
    return
end

% Note: tauchenoptions.dshift equals zero gives the Tauchen method. 
% For nonzero tauchenoptions.dshift this is actually implementing a non-standard Tauchen method.
if tauchenoptions.parallel==0 || tauchenoptions.parallel==1
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z
    
    z_grid=zstar*ones(znum,1) + linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)';
    omega=z_grid(2)-z_grid(1); %Note that all the points are equidistant by construction.
    
    zi=z_grid*ones(1,znum);
%     zj=ones(znum,1)*z';
    zj=tauchenoptions.dshift*ones(znum,znum)+ones(znum,1)*z_grid';
    
    P_part1=normcdf(zj+omega/2-rho*zi,mew,sigma);
    P_part2=normcdf(zj-omega/2-rho*zi,mew,sigma);
    
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);
    
elseif tauchenoptions.parallel==2 %Parallelize on GPU
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z
    
    z_grid=gpuArray(zstar*ones(znum,1) + linspace(-Tauchen_q*sigmaz,Tauchen_q*sigmaz,znum)');
    omega=z_grid(2)-z_grid(1); %Note that all the points are equidistant by construction.
    
    % NOTE; normcdf NOW WORKS FOR GPU, I SHOULD CHECK IF IT IS FASTER
    %Note: normcdf is not yet a supported function for use on the gpu in Matlab
    %(see list of supported functions at http://www.mathworks.es/es/help/distcomp/run-built-in-functions-on-a-gpu.html)
    %However erf is supported, and we can easily construct our own normcdf
    %from erf (see http://en.wikipedia.org/wiki/Normal_distribution for the
    %formula for normcdf as function of erf)
    %Comparing the output from using erf to that with normpdf the differences are
    %of the order of machine rounding errors (10e-16).
    
    tauchenoptions.dshift=gpuArray(tauchenoptions.dshift*ones(1,znum));
    
    erfinput=arrayfun(@(zi,zj,omega,rho,mew,sigma) ((zj+omega/2-rho*zi)-mew)/sqrt(2*sigma^2), z_grid,tauchenoptions.dshift+z_grid',omega, rho,mew,sigma);
    P_part1=0.5*(1+erf(erfinput));
    
    erfinput=arrayfun(@(zi,zj,omega,rho,mew,sigma) ((zj-omega/2-rho*zi)-mew)/sqrt(2*sigma^2), z_grid,tauchenoptions.dshift+z_grid',omega, rho,mew,sigma);
    P_part2=0.5*(1+erf(erfinput));
    
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);
    
end

end