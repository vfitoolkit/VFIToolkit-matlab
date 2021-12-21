function [states,transmatrix]=RouwenhorstMethod(mew,rho,sigmasq,znum,rouwenhorstoptions)
% Create states vector and transition matrix for the discrete markov process approximation of AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq), by Rouwenhorst method
% Rouwenhorst method outperforms the Tauchen method when rho is close to 1.
% Inputs
%   mew            - AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   rho            - AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   sigmasq        - AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   znum           - number of states in discretization of z (must be an odd number)
% Optional Inputs
%   rouwenhorstoptions - allows user to control internal options
%     rouwenhorstoptions.parallel: set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   states         - column vector containing the znum states of the discrete approximation of z
%   transmatrix    - transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
% Thanks to Iskander Karibzhanov who let me borrow heavily from his codes for this.
%%%%%%%%%%%%%%%
% Reference:
% "Finite State Markov-Chain Approximations to Highly Persistent Processes."
% by Karen A. Kopecky and Richard M. H. Suen.
% Review of Economic Dynamics 13 (2010), pp. 701-714.
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

zbar=sqrt((znum-1))*(sqrt(sigmasq)/sqrt((1-rho^2)));
z_grid=mew+linspace(-zbar,zbar,znum)';
p=(1+rho)/2; q=p;

P=rouwenhorst(znum,p,q);
% [z_grid,P]=rouwenhorst(znum,mew/(1-rho),rho,sqrt(sigmasq));

% Following is an faster-than-usual shortcut to calculate the stationary
% distribution for Rouwenhorst quadrature.
% s=zeros(znum,1);
% for j=1:znum
%     s(j)=nchoosek(znum-1,j-1);
% end
% s=s/2^(znum-1);
% % I do not actually include this in the outputs. It is just left here for reference.

% HAVE DONE THE LAZY OPTION. THIS SHOULD REALLY BE REWRITTEN SO THAT JUST
% CREATE ON GPU OR CPU AS APPROPRIATE. (AVOID THE OVERHEAD OF MOVING TO
% GPU)
if rouwenhorstoptions.parallel==2 
    states=gpuArray(z_grid);
    transmatrix=gpuArray(P); %(z,zprime)
else
    states=z_grid;
    transmatrix=P; %(z,zprime)    
end

%     function P=rouwenhorst(h,p,q)
%         if h==2
%             P=[p 1-p; 1-q q];
%         else
%             P1=rouwenhorst(h-1);
%             z=zeros(1,h);
%             z1=zeros(h-1,1);
%             P=[p*P1 z1; z]+[z1 (1-p)*P1; z]+...
%                 [z; (1-q)*P1 z1]+[z; z1 q*P1];
%             P(2:h-1,:)=P(2:h-1,:)/2;
%         end
%     end

end