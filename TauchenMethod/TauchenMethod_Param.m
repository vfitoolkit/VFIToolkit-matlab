function [states, transmatrix]=TauchenMethod_Param(mew,sigmasq,rho,znum,q, tauchenoptions)

if nargin<6
    tauchenoptions.parallel=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;tauchenoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        tauchenoptions.parallel=0;
    end
end
    
%Create states vector and transition matrix for the discrete markov process approximation of AR(1) process z'=rho*z+e, e~N(mew,sigmasq), by Tauchens method

%Recommended choice for Parallel is 3 (on GPU). It is substantially faster
%(albeit only for very large grids; for small grids cpu is just as fast)

%q %max number of std devs from mean
%znum %number of states in discretization of z (must be an odd number)

% if Verbose==1
%     fprintf('Approximating an AR process using Tauchen Method with %i points and q=%i \n',znum,q)
% end

if tauchenoptions.parallel==0
    sigma=sqrt(sigmasq); %stddev of e
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z

    z=zstar*ones(znum,1) + linspace(-q*sigmaz,q*sigmaz,znum)';
    omega=z(2)-z(1); %Note that all the points are equidistant by construction.
    
    zi=z*ones(1,znum);
    zj=ones(znum,1)*z';
    
    P_part1=normcdf(zj+omega/2-rho*zi,mew,sigma);
    P_part2=normcdf(zj-omega/2-rho*zi,mew,sigma);
    
    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);

elseif tauchenoptions.parallel==1 %Parallelize on CPU
    disp('TauchenMethod_Param does not support Parallel=1, as parallelizing on CPU does not appear to provide any speed boost (it is matrix algebra); doing TauchenMethod_Param as if Parallel=0')
    
    sigma=sqrt(sigmasq); %stddev of e
    zstar=mew/(1-rho); %expected value of z
    sigmaz=sigma/sqrt(1-rho^2); %stddev of z

    z=zstar*ones(znum,1) + linspace(-q*sigmaz,q*sigmaz,znum)';
    omega=z(2)-z(1); %Note that all the points are equidistant by construction.
    
    zi=z*ones(1,znum);
    zj=ones(znum,1)*z';
    
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

    erfinput=arrayfun(@(zi,zj,omega,rho,mew,sigma) ((zj+omega/2-rho*zi)-mew)/sqrt(2*sigma^2), z,z',omega, rho,mew,sigma);
    P_part1=0.5*(1+erf(erfinput));
    
    erfinput=arrayfun(@(zi,zj,omega,rho,mew,sigma) ((zj-omega/2-rho*zi)-mew)/sqrt(2*sigma^2), z,z',omega, rho,mew,sigma);
    P_part2=0.5*(1+erf(erfinput));

    P=P_part1-P_part2;
    P(:,1)=P_part1(:,1);
    P(:,znum)=1-P_part2(:,znum);

end

states=z;
transmatrix=P; %(z,zprime)

end