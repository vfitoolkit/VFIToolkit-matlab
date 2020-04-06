function [states, transmatrix]=TauchenMethod(mew,sigmasq,rho,znum,q, tauchenoptions, dshift)
% Create states vector and transition matrix for the discrete markov process approximation of AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq), by Tauchens method
% Inputs
%   mew            - AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   rho            - AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   sigmasq        - AR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   q              - max number of std devs from mean
%   znum           - number of states in discretization of z (must be an odd number)
% Optional Inputs
%   tauchenoptions - allows user to control internal options
%     tauchenoptions.parallel: set equal to 2 to use GPU, 0 to use CPU
%   dshift: allows approximating 'trend-reverting' process around a deterministic trend (not part of standard Tauchen method)
% Outputs
%   states         - column vector containing the znum states of the discrete approximation of z
%   transmatrix    - transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
% Helpful info:
%   Var(z)=sigmasq/(1-rho^2); note that if mew=0, then sigmasqz=sigmasq/(1-rho^2).
%%%%%%%%%%%%%%%

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
if nargin<7
    dshift=0;
end

if znum==1
    states=mew/(1-rho); %expected value of z
    transmatrix=1;
    if tauchenoptions.parallel==2
        states=gpuArray(states);
        transmatrix=gpuArray(transmatrix);
    end
    return
end

% Note: dshift equals zero gives the Tauchen method. For nonzero dshift
% this is actually implementing a non-standard Tauchen method.
if tauchenoptions.parallel==0
    sigma=sqrt(sigmasq); %stddev of e
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z
    
    z=zstar*ones(znum,1) + linspace(-q*sigmaz,q*sigmaz,znum)';
    omega=z(2)-z(1); %Note that all the points are equidistant by construction.
    
    zi=z*ones(1,znum);
%     zj=ones(znum,1)*z';
    zj=dshift*ones(znum,znum)+ones(znum,1)*z';
    
    P_part1=normcdf(zj+omega/2-rho*zi,mew,sigma);
    P_part2=normcdf(zj-omega/2-rho*zi,mew,sigma);
    
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);
    
elseif tauchenoptions.parallel==1 %Parallelize on CPU
%    disp('TauchenMethod_Param does not support Parallel=1, as parallelizing on CPU does not appear to provide any speed boost (it is matrix algebra); doing TauchenMethod_Param as if Parallel=0')
    
    sigma=sqrt(sigmasq); %stddev of e
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z
    
    z=zstar*ones(znum,1) + linspace(-q*sigmaz,q*sigmaz,znum)';
    omega=z(2)-z(1); %Note that all the points are equidistant by construction.
    
    zi=z*ones(1,znum);
%     zj=ones(znum,1)*z';
    zj=dshift*ones(znum,znum)+ones(znum,1)*z';
    
    P_part1=normcdf(zj+omega/2-rho*zi,mew,sigma);
    P_part2=normcdf(zj-omega/2-rho*zi,mew,sigma);
    
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);
    
    
elseif tauchenoptions.parallel==2 %Parallelize on GPU
    sigma=sqrt(sigmasq); %stddev of e
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z
    
    z=gpuArray(zstar*ones(znum,1) + linspace(-q*sigmaz,q*sigmaz,znum)');
    omega=z(2)-z(1); %Note that all the points are equidistant by construction.
    
    %Note: normcdf is not yet a supported function for use on the gpu in Matlab
    %(see list of supported functions at http://www.mathworks.es/es/help/distcomp/run-built-in-functions-on-a-gpu.html)
    %However erf is supported, and we can easily construct our own normcdf
    %from erf (see http://en.wikipedia.org/wiki/Normal_distribution for the
    %formula for normcdf as function of erf)
    %Comparing the output from using erf to that with normpdf the differences are
    %of the order of machine rounding errors (10e-16).
    
    dshift=gpuArray(dshift*ones(1,znum));
    
    erfinput=arrayfun(@(zi,zj,omega,rho,mew,sigma) ((zj+omega/2-rho*zi)-mew)/sqrt(2*sigma^2), z,dshift+z',omega, rho,mew,sigma);
    P_part1=0.5*(1+erf(erfinput));
    
    erfinput=arrayfun(@(zi,zj,omega,rho,mew,sigma) ((zj-omega/2-rho*zi)-mew)/sqrt(2*sigma^2), z,dshift+z',omega, rho,mew,sigma);
    P_part2=0.5*(1+erf(erfinput));
    
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);
    
end

states=z;
transmatrix=P; %(z,zprime)

end