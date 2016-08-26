function [states, transmatrix]=TauchenMethodVAR(mew,sigmasq,rho,znum,q, tauchenoptions)
% Create states vector and transition matrix for the discrete markov process approximation of n-variable VAR(1) process z'=mew+rho*z+e, e~N(0,sigmasq), by Tauchens method
% (Abuse of notation: sigmasq in codes is a column vector, in the VAR notation it is a matrix in which all of the non-diagonal elements are zeros) 
% Inputs
%   mew            - n-by-1 column vector; VAR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   rho            - n-by-n matrix;        VAR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   sigmasq        - n-by-1 matrix;        VAR(1) process z'=mew+rho*z+e, e~N(0,sigmasq)
%   q              - n-by-1 column vector; max number of std devs from mean
%   znum           - n-by-1 number of states in discretization of z (must be an odd number)
% Optional inputs
%   tauchenoptions - allows user to control internal options
%     tauchenoptions.parallel: set equal to 2 to use GPU, 0 to use CPU
% Outputs
%   states         - sum(znum)-by-1 column vector; stacked column vector containing the sum(znum) states of 
%                    the discrete approximation of z for each of the n variables
%   transmatrix    - prod(znum)-by-prod(znum) matrix; transition matrix of the discrete approximation of z;
%                    transmatrix(i,j) is the probability of transitioning from state i to state j
%
% Thanks to Iskander Karibzhanov who let me borrow heavily from his codes for this.
%%%%%%%%%%%%%%%

if nargin<6
    % Recommended choice for Parallel is 2 (on GPU). It is substantially faster (albeit only for very large grids; for small grids cpu is just as fast)
    tauchenoptions.parallel=2;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    eval('fieldexists=1;tauchenoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        tauchenoptions.parallel=2;
    end
end

% Below is largely just a copy of the code of Iskander Karibzhanov. Main difference is
% just the shape of the output of the 'state' variable and transition matrix (mine is transpose of his;
% I follow the economics/maths standard of expressing transition matrix as (z,zprime), he follows 
% computer science/algorithms standard of (zprime,z)). The
% other noteworthy difference from original is that his used dlyap which requires 
% the Control System Toolbox (which I avoid by using 'substitutefor_dlyap').

cumprod_znum=cumprod(znum);
n=length(znum); % number of variables in VAR
prod_znum=cumprod_znum(n); % number of states in Markov chain
sigma=sqrt(sigmasq); %stddev of e

q_sigmaz=real(q.*sqrt(diag(substitutefor_dlyap(rho,diag(sigmasq))))); % std.dev. of y
zgrids=cell(n,1); z=nan(n,prod_znum);
for i=1:n
    if znum(i)>1
        zgrids{i}=linspace(-1,1,znum(i))*q_sigmaz(i); 
    else
        zgrids{i}=0;
    end
    z(i,:)=reshape(repmat(zgrids{i},cumprod_znum(i)/znum(i),prod_znum/cumprod_znum(i)),1,prod_znum);
end
q_sigmaz=q_sigmaz./(znum(:)-1); P=1;
for i=1:n
    h=normcdf(bsxfun(@minus,zgrids{i}'+q_sigmaz(i),rho(i,:)*z)/sigma(i));
    h(znum(i),:)=1; h=permute([h(1,:);diff(h,1,1)],[3 1 2]);
    P=reshape(repmat(h,cumprod_znum(i)/znum(i),prod_znum/cumprod_znum(i)),prod_znum,prod_znum).*P;
end
%z=bsxfun(@plus,z,((eye(n,n)-rho)^(-1))*mew); % This is the form in which
%Iskander's codes output state, as a n-by-prod(znum) matrix, rather than
%the stacked column vector form that I use.

zmean=((eye(n,n)-rho)^(-1))*mew;
zoutput=zgrids{1}'+ones(znum(1),1)*zmean(1);
for i=2:n
    zoutput=[zoutput;zgrids{i}'+ones(znum(i),1)*zmean(i)];
end

if tauchenoptions.parallel==2 
    states=gpuArray(zoutput);
    transmatrix=gpuArray(P'); %(z,zprime)
else
    states=zoutput;
    transmatrix=P'; %(z,zprime)    
end

end
