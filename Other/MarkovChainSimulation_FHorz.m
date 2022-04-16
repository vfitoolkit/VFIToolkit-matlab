function z_timeseries=MarkovChainSimulation_FHorz(z_grid_J,pi_z_J,jequaloneDist,simoptions)
% Create a time-series simulation of a (one dimensional) first-order discrete markov process
% Inputs:
%  z_grid_J      - grid values of the discrete markov process (column vector for each period j)
%  pi_z_J        - transition matrix of the discrete markov process (matrix for each period j, row gives probabilities for next period states, should add to one)
%  jequaloneDist - initial distribution of z at period 1 from which start of time series is drawn 
% Optional inputs
%  simoptions:
%     simoptions.nsims - number of simulations (by default it is 1, so you just get a time series)

%% Set default options
if ~exist('simoptions','var')
    simoptions.nsims=1; % a single time series
else
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=1;
    end
end

%% Get pi_z_J z_grid_J and jequaloneDist
if ~exist('simoptions','var')
    simoptions=struct();
end

if isfield(simoptions,'jequaloneDist')
    if ~isempty(jequaloneDist)
        warning('MarkovChainSimulation_FHorz: Using simoptions.jequaloneDist to overwrite jequaloneDist')
    end
    jequaloneDist=simoptions.jequaloneDist;
    jequaloneDist=reshape(jequaloneDist,[numel(jequaloneDist),1]);
else
    if isempty(jequaloneDist)
        error('MarkovChainSimulation_FHorz: Need to input either simoptions.jequaloneDist or jequaloneDist')
    end
end
if isfield(simoptions,'pi_z_J')
    if ~isempty(pi_z_J)
        warning('MarkovChainSimulation_FHorz: Using simoptions.pi_z_J to overwrite pi_z_J')
    end
    pi_z_J=simoptions.pi_z_J;
else
    if isempty(pi_z_J)
        error('MarkovChainSimulation_FHorz: Need to input either simoptions.pi_z_J or pi_z_J')
    end
end
if isfield(simoptions,'z_grid_J')
    if ~isempty(z_grid_J)
        warning('MarkovChainSimulation_FHorz: Using simoptions.z_grid_J to overwrite z_grid_J')
    end
    z_grid_J=simoptions.z_grid_J;
else
    if isempty(z_grid_J)
        error('MarkovChainSimulation_FHorz: Need to input either simoptions.z_grid_J or z_grid_J')
    end
end

%% Get the dimensions
N_z=size(z_grid_J,1);
N_j=size(z_grid_J,2);

% Check the sizes of jequaloneDist and pi_z_J
if size(pi_z_J,1)~=N_z
    error('z_grid_J and pi_z_J disagree about the size of N_z')
elseif size(pi_z_J,2)~=N_z
    error('z_grid_J and pi_z_J disagree about the size of N_z')
elseif size(pi_z_J,3)~=N_j
    error('z_grid_J and pi_z_J disagree about the size of N_j')
end
if length(jequaloneDist)~=N_z
    error('z_grid_J and jequaloneDist disagree about the size of N_z')
end

z_grid_J=gather(z_grid_J);

cumjequaloneDist=cumsum(jequaloneDist);

cumsum_pi_z_J=cumsum(pi_z_J,2); % Sum in second dimension (next period state transition probabilities)

% Simulate Markov chain with transition state pi_z
z_indexes=zeros(simoptions.nsims,N_j); % A contains the time series of states
parfor ii=1:simoptions.nsims
    A_ii=zeros(1,N_j);
    % jj=1
    [~,A_ii(1)]=max(cumjequaloneDist>rand(1));
    for jj=2:N_j
        [~,zcurr]=max(cumsum_pi_z_J(A_ii(jj-1),:,jj-1)>rand(1));
        A_ii(jj)=zcurr;
    end
    z_indexes(ii,:)=A_ii;
end

z_timeseries=zeros(simoptions.nsims,N_j);
for jj=1:N_j
    z_timeseries(:,jj)=z_grid_J(z_indexes(:,jj),jj);
end

end
