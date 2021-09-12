function [mean,variance,corr,statdist]=MarkovChainMoments(z_grid,pi_z,mcmomentsoptions)

if exist('mcmomentsoptions','var')==0
    mcmomentsoptions.parallel=1+(gpuDeviceCount>0);
    mcmomentsoptions.T=10^6;
    mcmomentsoptions.Tolerance=10^(-8);
else
    if isfield(mcmomentsoptions,'parallel')==0
        mcmomentsoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(mcmomentsoptions,'Tolerance')==0
        mcmomentsoptions.Tolerance=10^(-8);
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

%% Compute the stationary distribution
% I tried out both just iterating on the distribution and calculating eigenvectors, seems like iteration is faster (everything is measured in thousand-ths of a second anyway)

pi_z_transpose=pi_z';
statdist=ones(length(z_grid),1)/length(z_grid);
currdist=1;
while currdist>mcmomentsoptions.Tolerance
    statdistold=statdist;
    statdist=pi_z_transpose*statdist;
    currdist=sum(abs(statdist-statdistold));
end

% % Eigenvvector approach to stationary distriubtion 
% % (see https://en.wikipedia.org/wiki/Markov_chain#Stationary_distribution_relation_to_eigenvectors_and_simplices )
% % does not appear to be any faster (in fact marginally slower)
% tic;
% [statdist,~]=eigs(gather(pi_z)',1);
% statdist=statdist./sum(statdist);
% toc

%% Calculate the mean and variance
mean=z_grid'*statdist;

secondmoment=(z_grid.^2)'*statdist;
variance=secondmoment-mean^2;

%% Now for the (first-order auto-) correlation
% This takes vast majority of the time of MarkovChainMoments()
% Might be possible to speed this up by using parallelization? Not sure
% about computing correlation using parallelization (does it converge, and
% does it work faster?)

if Parallel==2 || Parallel==4 % Move to cpu for simulation. Is just much faster.
    z_grid=gather(z_grid);
end

% Simulate Markov chain with transition state pi_z
% Maybe I should be doing burnin here??
A=zeros(T,1); % A contains the time series of states
A(1)=floor(length(z_grid)/2); % Start the simulation in the midpoint
shocks_raw=rand(T,1);
cumsum_pi_z=cumsum(gather(pi_z),2);
for t=2:T
    temp_cumsum_pi_z=cumsum_pi_z(A(t-1),:);
    temp_cumsum_pi_z(temp_cumsum_pi_z<=shocks_raw(t))=2;
    [~,A(t)]=min(temp_cumsum_pi_z);
end
corr_temp=corrcoef(z_grid(A(2:T)),z_grid(A(1:T-1)));
corr=corr_temp(2,1);    
    
end