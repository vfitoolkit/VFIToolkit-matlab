function TimeSeries=TimeSeries_Case1(TimeSeriesIndexes, Policy, TimeSeriesFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,simoptions)

% Just being lazy in implementing GPU here for now, will do it properly
% some other time (see the SSAggVars codes for example of doing this
% properly)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

l_a=length(n_a);
l_z=length(n_z);

%% Check which simoptions have been used, set all others to defaults 
if nargin<10
    %If simoptions is not given, just use all the relevant defaults
    simoptions.parallel=2;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
end

%%
PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z);

% Just being lazy in implementing GPU here for now, will do it properly
% some other time (see the SSAggVars codes for example of doing this
% properly)
if simoptions.parallel~=2
    TimeSeriesIndexesKron=zeros(3,length(TimeSeriesIndexes(1,:)));
    
    for t=1:length(TimeSeriesIndexes(1,:))
        TimeSeriesIndexesKron(1,t)=sub2ind_homemade([n_a],TimeSeriesIndexes(1:l_a,t));%a
        TimeSeriesIndexesKron(2,t)=sub2ind_homemade([n_z],TimeSeriesIndexes(l_a+1:l_a+l_z,t));%z
    end
    
    TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, PolicyIndexesKron, TimeSeriesFn, n_d, n_a, n_z, d_grid, a_grid, z_grid);
    
    TimeSeries=TimeSeriesKron;
elseif simoptions.parallel==2
    TimeSeriesIndexes=gather(TimeSeriesIndexes);
    TimeSeriesIndexesKron=zeros(3,length(TimeSeriesIndexes(1,:)));
    
    for t=1:length(TimeSeriesIndexes(1,:))
        TimeSeriesIndexesKron(1,t)=sub2ind_homemade([n_a],TimeSeriesIndexes(1:l_a,t));%a
        TimeSeriesIndexesKron(2,t)=sub2ind_homemade([n_z],TimeSeriesIndexes(l_a+1:l_a+l_z,t));%z
    end
    
    TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, gather(PolicyIndexesKron), TimeSeriesFn, n_d, n_a, n_z, gather(d_grid), gather(a_grid), gather(z_grid));
    
    TimeSeries=gpuArray(TimeSeriesKron);

end

end