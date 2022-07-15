function z_timeseries=MarkovChainSimulation_FHorz(z_grid_J,pi_z_J,jequaloneDistz,simoptions)
% Create a time-series simulation of a (one dimensional) first-order discrete markov process
% Inputs:
%  z_grid_J       - grid values of the discrete markov process (column vector for each period j)
%  pi_z_J         - transition matrix of the discrete markov process (matrix for each period j, row gives probabilities for next period states, should add to one)
%  jequaloneDistz - initial distribution of z at period 1 from which start of time series is drawn 
% Optional inputs
%  simoptions:
%   - simoptions.nsims - number of simulations (by default it is 1, so you just get a time series)
%   - simoptions.n_z: if the markov process is multidimensional you will need to input n_z

%% Set default options
if ~exist('simoptions','var')
    simoptions.nsims=1; % a single time series
    simoptions.n_z=size(z_grid_J,1); % is assumed to be uni-dimensional
else
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=1;
    end
    if ~isfield(simoptions,'n_z')
        simoptions.n_z=size(z_grid_J,1); % is assumed to be uni-dimensional
    end
end

%% Get pi_z_J z_grid_J and jequaloneDist
if ~exist('simoptions','var')
    simoptions=struct();
end

if isfield(simoptions,'jequaloneDist')
    if ~isempty(jequaloneDistz)
        warning('MarkovChainSimulation_FHorz: Using simoptions.jequaloneDist to overwrite jequaloneDist')
    end
    jequaloneDistz=simoptions.jequaloneDist;
    jequaloneDistz=reshape(jequaloneDistz,[numel(jequaloneDistz),1]);
else
    if isempty(jequaloneDistz)
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
l_z=length(simoptions.n_z);
N_z=prod(simoptions.n_z);
N_j=size(z_grid_J,2);

% Check the sizes of jequaloneDist and pi_z_J
if size(z_grid_J,1)~=sum(simoptions.n_z)
    error('z_grid_J does not have right number of points for z')
end
if size(pi_z_J,1)~=N_z
    error('pi_z_J does not have right number of points for z')
elseif size(pi_z_J,2)~=N_z
    error('pi_z_J does not have right number of points for z')
elseif size(pi_z_J,3)~=N_j
    error('pi_z_J does not have right number of points for j (compared to z_grid_J)')
end
if length(jequaloneDistz)~=N_z
    error('z_grid_J and jequaloneDist disagree about the size of N_z')
end

z_grid_J=gather(z_grid_J);

cumjequaloneDist=cumsum(jequaloneDistz);

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

if l_z==1
    z_timeseries=zeros(simoptions.nsims,N_j);
    for jj=1:N_j
        z_timeseries(:,jj)=z_grid_J(z_indexes(:,jj),jj);
    end
else % z is multidimensional
    z_gridvals_J=zeros(N_z,l_z,N_j);
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(simoptions.n_z,z_grid_J(:,jj),1);
    end
    z_timeseries=zeros(simoptions.nsims,N_j,l_z);
    for jj=1:N_j
        for ii=1:l_z % This can probably be vectorized to speed it up
            z_timeseries(:,jj,ii)=z_gridvals_J(z_indexes(:,jj),ii,jj);    
        end
    end
end

end
