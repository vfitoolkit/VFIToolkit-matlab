function z_timeseries=MarkovChainSimulation(T,z_grid,pi_z,n_z,simoptions)
% Create a time-series simulation of a (one dimensional) first-order discrete markov process
% Inputs:
%  T        - number of time periods to simulate
%  z_grid   - grid values of the discrete markov process (column vector)
%  pi_z     - transition matrix of the discrete markov process (matrix, row gives probabilities for next period states, should add to one)
% Optional inputs (simoptions)
%  burnin    - number of periods to 'burn in' (these are automatically  deleted from simulation to eliminate bias caused by initial seed point value)
%  seedpoint - point from which to start the simulation

%% Set default options
N_z=prod(n_z);
if ~exist('simoptions','var')
    simoptions.burnin=30*N_z;
    simoptions.seedpoint=floor(N_z+1)/2; % Just use midpoint as I'm lazy
else
    if ~isfield(simoptions,'burnin')
        simoptions.burnin=30*N_z;
    end
    if ~isfield(simoptions,'seepoint')
        simoptions.seedpoint=floor(N_z+1)/2; % Just use midpoint as I'm lazy
    end
end

%% Check if using z_grid_J and pi_z_J, in which case just produce a finite horizon simulation
if isfield(simoptions,'z_grid_J') || isfield(simoptions,'pi_z_J')
    if ~isfield(simoptions,'jequaloneDist')
        error('When using z_grid_J and pi_z_J you must set simoptions.jequaloneDist')
    end
    % Check that time horizon is less than or equal to the finite horizon
    if T>size(simoptions.z_grid_J,2)
        error('Cannot simulate a time series longer than the finite horizon')
    end
    
    z_timeseries_index=zeros(T,1);
    z_timeseries=zeros(T,1);
        
    z_timeseries_index(1)=max(cumsum(simoptions.jequaloneDist)>rand(1)); % Pull initial value from jequaloneDist
    zcurr=z_timeseries_index(1);

    for jj=1:T
        % First, get pi_z and z_grid
        if isfield(simoptions,'pi_z_J')
            z_grid=simoptions.z_grid_J(:,jj);
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif isfield(simoptions,'ExogShockFn')
            if isfield(simoptions,'ExogShockFnParamNames')
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,pi_z]=simoptions.ExogShockFn(jj);
            end
        end
        z_grid=gather(z_grid);
        pi_z=gather(pi_z);
        
        z_timeseries(jj)=z_grid(z_timeseries_index(jj));

        
        if jj<T
            cumpi_z=cumsum(pi_z,2);
            [~,zcurr]=max(cumpi_z(zcurr,:)>rand(1));
            z_timeseries_index(jj+1)=zcurr;
        end
        
    end
        
    return
end



%% Simulate
z_grid=gather(z_grid);
pi_z=gather(pi_z);
% Simulate index
z_timeseries_index=zeros(T+simoptions.burnin,1);
z_timeseries_index(1)=simoptions.seedpoint;
cumpi_z=cumsum(pi_z,2);
zcurr=z_timeseries_index(1);
for tt=2:T+simoptions.burnin
    [~,zcurr]=max(cumpi_z(zcurr,:)>rand(1));
    z_timeseries_index(tt)=zcurr;
end
% Switch to values
z_timeseries=z_grid(z_timeseries_index);

end
