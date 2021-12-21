function [states, transmatrix, dividingvalues]=TauchenMethod_Data(zdata,n_z,q,Parallel,Verbose)

  disp('WARNING: I have not tested TauchenMethod_Data since I last modified it to use TauchenMethod_Param (main issue is just to double check that sigmasq_epsilon is the right thing to pass to TauchenMethod_Param)')
  %Estimate a first order markov chain for z
  %create states and transition matrix for the Markov chain, z'=rho*z+e, e~N(mew,sigmasq), by Tauchens method

  %The inputs are
  %zdata: the data series for which to generate the markov transition matrix (should be a vector)
  %n_z: the number of states for the resulting markov chain
  %q: the number of standard deviation to use to represent the max & min values for the Tauchen method grid

  %The outputs are
  %states: the values of z that correspont to each states
  %transmatrix: the estimated transition matrix for z (z by z')
  %dividingvalues: the values of z that are the points that define the edges of the states

  % First step is to estimate the AR1 process
  %coeffs=zeros(2,1);
  T=length(zdata);
  intercept=ones(T-1,1);
  X=[intercept,zdata(1:T-1)];
  Y=zdata(2:T);
  coeffs=((X'*X)^(-1))*X'*Y;
  mew=coeffs(1);
  rho=coeffs(2);
  %We now have two of three things we need to use Tauchens method, we just need sigmasq_epsilon
  innovations=zeros(T-1,1); %can't estimate innovation in the very first period
  for t=1:T-1
      innovations(t)=zdata(t+1)-mew-rho*zdata(t);
  end
  %Now we need to use these innovations to calculate sigmasq_epsilon
  %Estimate as 1/sqrt(n-p) *sum(epsilon^2)
  sigmasq_epsilon=(1/(T-2))*sum(innovations.^2);

  if Verbose==1
      %Print the important statistics from the AR1 that we will use for the Tauchen method
      disp('Based on the data, the tauchen method is being used to approximate the AR(1) process:')
      fprintf('   zprime= %n *z+e, e~N(%n,%sigmasq)',rho,mew,sigmasq_epsilon)
%       mew
%       rho
%       sigmasq_epsilon
  end

  [states, transmatrix]=TauchenMethod_Param(mew,sigmasq_epsilon,rho,znum,q, Parallel, Verbose);

  dividingvalues=zeros(n_z-1,1);
  for i=1:n_z-1
      dividingvalues(i)=(states(i)+states(i+1))/2;
  end
  
end