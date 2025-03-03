function [z_grid,pi_z]=discretizeAR1_Rouwenhorst(mew,rho,sigma,znum,rouwenhorstoptions)
% Create states vector, z_grid, and transition matrix, P, for the discrete markov process approximation 
%    of AR(1) process z'=mew+rho*z+e, e~N(0,sigma^2), by Rouwenhorst method
%
% Inputs
%   mew            - constant term coefficient
%   rho            - autocorrelation coefficient
%   sigma          - standard deviation of (gaussion) innovations
%   znum           - number of states in discretization of z (must be an odd number)
% Optional Inputs (rouwenhorstoptions)
%   parallel: set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   z_grid         - column vector containing the znum states of the discrete approximation of z
%   P              - transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
% Helpful info:
%   Var(z)=(sigma^2)/(1-rho^2). So sigmaz=sigma/sqrt(1-rho^2);   sigma=sigmaz*sqrt(1-rho^2)
%                                  where sigmaz= standard deviation of z
%     E(z)=mew/(1-rho)
%
% Thanks to Iskander Karibzhanov who let me borrow heavily from his codes for this.
%
%%%%%%%%%%%%%%%
% Reference for reading more:
% Original paper:
% Rouwenhorst (1995) - "Asset pricing implications of equilibrium business cycle models"
% Showing Rouwenhorst is more accurate than Tauchen for high rho (nearer 1):
% Kopecky & Suen (2010) - "Finite State Markov-Chain Approximations to Highly Persistent Processes"
% URL: http://www.karenkopecky.net/RouwenhorstPaperFinal.pdf

if exist('rouwenhorstoptions','var')==0
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    rouwenhorstoptions.parallel=1+(gpuDeviceCount>0);
else
    %Check rouwenhorstoptions for missing fields, if there are some fill them with the defaults
    if isfield(rouwenhorstoptions,'parallel')==0
        rouwenhorstoptions.parallel=1+(gpuDeviceCount>0);
    end
end

zbar=sqrt((znum-1))*(sigma/sqrt((1-rho^2)));
z_grid=mew+linspace(-zbar,zbar,znum)';
p=(1+rho)/2; q=p;

pi_z=rouwenhorst(znum,p,q);

% Following is an faster-than-usual shortcut to calculate the stationary
% distribution for Rouwenhorst quadrature.
% s=zeros(znum,1);
% for j=1:znum
%     s(j)=nchoosek(znum-1,j-1);
% end
% s=s/2^(znum-1);
% % I do not actually include this in the outputs. It is just left here for reference.

% HAVE DONE THE LAZY OPTION. THIS SHOULD REALLY BE REWRITTEN SO THAT JUST
% CREATE ON GPU OR CPU AS APPROPRIATE. (AVOID THE OVERHEAD OF MOVING TO GPU)
if rouwenhorstoptions.parallel==2 
    z_grid=gpuArray(z_grid);
    pi_z=gpuArray(pi_z); %(z,zprime)  
end

end