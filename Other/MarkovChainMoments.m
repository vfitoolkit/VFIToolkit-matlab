function [mean,variance,corr,statdist]=MarkovChainMoments(z_grid,pi_z,mcmomentsoptions)

% mcmomentsoptions is a bit of a hack
if exist('mcmomentsoptions','var')==0
    mcmomentsoptions.parallel=1+(gpuDeviceCount>0);
    mcmomentsoptions.T=10^6;
    mcmomentsoptions.Tolerance=10^(-9);
else
    if isfield(mcmomentsoptions,'parallel')==0
        mcmomentsoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(mcmomentsoptions,'Tolerance')==0
        mcmomentsoptions.Tolerance=10^(-9);
    end
    if isfield(mcmomentsoptions,'T')==0
        mcmomentsoptions.T=10^6;
    end
end
Parallel=mcmomentsoptions.parallel;
T=mcmomentsoptions.T;

% Inputs are a column vector z_grid that contains the states of the Markov
% chain, and pi_z which contains the tranisition matrix of the Markov chain
% (which row indexing this state, column next state).
% Also the tolerance to which the stationary distribution is calculated.

% Outputs are the mean and variance of the first order markov chain. And the
% stationary distribution of the states.

% T is needed as correlation is calculated via simulation of this length
% (SURELY THERE IS SOME SMARTER WAY TO CALCULATE THE THEORETICAL ONE
% DIRECTLY USING z_grid & pi_z)


new=pi_z;
currdist=1;
while currdist>mcmomentsoptions.Tolerance
    old=new;
    new=(old^100);
    currdist=max(max(new-old));
end

statdist=((ones(1,length(z_grid))/length(z_grid))*new)'; %A column vector
% statdist=((ones(1,prod(n_z))/prod(n_z))*new)'; %A column vector
% mean_zr=sum(kron(z_grid(1:5),ones(7,1)).*statdist)
% mean_ze=sum(kron(ones(5,1),z_grid(6:end)).*statdist)

% % Eigenvalues approach to stationary distriubtion 
% % (see https://en.wikipedia.org/wiki/Markov_chain#Stationary_distribution_relation_to_eigenvectors_and_simplices )
% % does not appear to be any faster (in fact marginally
% % slower) (It also appears to be less accurate, although this may be a
% % coding error on my part)
% [statdist2,~]=eigs(gather(pi_z),1,1);
% if sum(statdist2)<0; 
%     statdist2=-statdist2; 
% end; 
% statdist2(statdist2<0)=0; 
% statdist2=statdist2./sum(statdist2);

mean=z_grid'*statdist;

secondmoment=(z_grid.^2)'*statdist;
variance=secondmoment-mean^2;

% THIS FUNCTION FROM HERE DOWN IS IN NEED OF A REWRITE TO WORK FASTER 
% (is it possible to calculate correlation directly from transition matrix and stationary distribution rather than simulating?)
% tic;
if Parallel==2 || Parallel==4 % Use GPU (assumes z_grid & pi_z are gpu arrays)
    %Simulate Markov chain with transition state pi_z
    A=gpuArray(ones(T,1)*floor(length(z_grid)/2)); %A contains the time series of states
    shocks_raw=rand(T,1);
    cumsum_pi_z=cumsum(pi_z,2);
    for t=2:T
        temp_cumsum_pi_z=cumsum_pi_z(A(t-1),:);
        temp_cumsum_pi_z(temp_cumsum_pi_z<=shocks_raw(t))=2;
        [~,A(t)]=min(temp_cumsum_pi_z);
    end
    corr_temp=corrcoef(z_grid(A(2:T)),z_grid(A(1:T-1)));
    corr=corr_temp(2,1);
else % On CPU
    %Simulate Markov chain with transition state pi_z
    A=ones(T,1)*floor(length(z_grid)/2); %A contains the time series of states
    shocks_raw=rand(T,1);
    cumsum_pi_z=cumsum(pi_z,2);
    for t=2:T
        temp_cumsum_pi_z=cumsum_pi_z(A(t-1),:);
        temp_cumsum_pi_z(temp_cumsum_pi_z<=shocks_raw(t))=2;
        [~,A(t)]=min(temp_cumsum_pi_z);
    end
    corr_temp=corrcoef(z_grid(A(2:T)),z_grid(A(1:T-1)));
    corr=corr_temp(2,1);
end    
% time3=toc
    
    
end