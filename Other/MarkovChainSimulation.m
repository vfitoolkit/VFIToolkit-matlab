function z_timeseries=MarkovChainSimulation(T,z_grid,pi_z,simoptions)
% Create a time-series simulation of a (one dimensional) first-order discrete markov process
% Inputs:
%  T        - number of time periods to simulate
%  z_grid   - grid values of the discrete markov process (column vector)
%  pi_z     - transition matrix of the discrete markov process (matrix, row gives probabilities for next period states, should add to one)
% Optional inputs (simoptions)
%  simoptions.burnin    - number of periods to 'burn in' (these are automatically  deleted from simulation to eliminate bias caused by initial seed point value)
%  simoptions.seedpoint - point from which to start the simulation
%  simoptions.nsims     - number of simulations (by default it is 1, so you just get a time series)

%% Set default options
N_z=size(z_grid,1);
if ~exist('simoptions','var')
    simoptions.burnin=30*N_z; % I just made this up, no idea what is appropriate
    simoptions.seedpoint=floor(N_z+1)/2; % Just use midpoint as I'm lazy
    simoptions.nsims=1; % number of simulations (by default it is 1, so you just get a time series)
else
    if ~isfield(simoptions,'burnin')
        simoptions.burnin=30*N_z;% I just made this up, no idea what is appropriate
    end
    if ~isfield(simoptions,'seepoint')
        simoptions.seedpoint=floor(N_z+1)/2; % Just use midpoint as I'm lazy
    end
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=1; % number of simulations (by default it is 1, so you just get a time series)
    end
end

%% Get the dimensions
% N_z=size(z_grid,1);
% Check the sizes of pi_z
if size(pi_z,1)~=N_z
    error('z_grid and pi_z disagree about the size of N_z')
elseif size(pi_z,2)~=N_z
    error('z_grid and pi_z disagree about the size of N_z')
end

%% Simulate
z_grid=gather(z_grid);
pi_z=gather(pi_z);
% Simulate index
z_timeseries=zeros(simoptions.nsims,T);
parfor ii=1:simoptions.nsims
    z_timeseries_index=zeros(1,T+simoptions.burnin);
    z_timeseries_index(1)=simoptions.seedpoint;
    cumpi_z=cumsum(pi_z,2);
    zcurr=z_timeseries_index(1);
    for tt=2:T+simoptions.burnin
        [~,zcurr]=max(cumpi_z(zcurr,:)>rand(1));
        z_timeseries_index(tt)=zcurr;
    end
    % Switch to values
    z_timeseries(ii,:)=z_grid(z_timeseries_index(simoptions.burnin+1:end));
end


end
