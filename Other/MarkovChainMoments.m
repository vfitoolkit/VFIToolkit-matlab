function [mean,variance,corr,statdist]=MarkovChainMoments(z_grid,pi_z,mcmomentsoptions)

% mcmomentsoptions is a bit of a hack
if nargin<3
    mcmomentsoptions.parallel=0;
    mcmomentsoptions.T=10^6;
    mcmomentsoptions.Tolerance=10^(-9);
else
    eval('fieldexists=1;mcmomentsoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        mcmomentsoptions.parallel=0;
    end
    eval('fieldexists=1;mcmomentsoptions.Tolerance;','fieldexists=0;')
    if fieldexists==0
        mcmomentsoptions.Tolerance=10^(-9);
    end
    eval('fieldexists=1;mcmomentsoptions.T;','fieldexists=0;')
    if fieldexists==0
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

mean=z_grid'*statdist;

secondmoment=(z_grid.^2)'*statdist;
variance=secondmoment-mean^2;

% IN OLD VERIONS OF CODE THE THIRD OUTPUT WAS "corr". "statdist" was the
% fourth output.
% HAVE REVERTED TO OLD VERSION. BUT REALLY THE PART OF THIS FUNCTION FROM
% HERE DOWN IS IN NEED OF A REWRITE TO WORK FASTER ON GPU.

if Parallel==0
    %Simulate Markov chain with transition state pi_z
    A=ones(T,1)*floor(length(z_grid)/2); %A contains the time series of states
    shocks_raw=rand(T,1);
    cumsum_pi_z=cumsum(pi_z,2);
    for t=2:T
        temp_cumsum_pi_z=cumsum_pi_z(A(t-1),:);
        temp_cumsum_pi_z(temp_cumsum_pi_z<=shocks_raw(t))=2;
        [trash,A(t)]=min(temp_cumsum_pi_z);
    end
    corr_temp=corrcoef(z_grid(A(2:T)),z_grid(A(1:T-1)));
    corr=corr_temp(2,1);
elseif Parallel==2 % Use GPU (assumes z_grid & pi_z are gpu arrays)
    %Simulate Markov chain with transition state pi_z
    A=gpuArray(ones(T,1)*floor(length(z_grid)/2)); %A contains the time series of states
    shocks_raw=rand(T,1);
    cumsum_pi_z=cumsum(pi_z,2);
    for t=2:T
        temp_cumsum_pi_z=cumsum_pi_z(A(t-1),:);
        temp_cumsum_pi_z(temp_cumsum_pi_z<=shocks_raw(t))=2;
        [trash,A(t)]=min(temp_cumsum_pi_z);
    end
    corr_temp=corrcoef(z_grid(A(2:T)),z_grid(A(1:T-1)));
    corr=corr_temp(2,1);
end    
    
    
end