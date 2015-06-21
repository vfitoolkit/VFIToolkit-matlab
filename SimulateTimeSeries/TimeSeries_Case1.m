function TimeSeries=TimeSeries_Case1(TimeSeriesIndexes, Policy, TimeSeriesFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,simoptions)

% Just being lazy in implementing GPU here for now, will do it properly
% some other time (see the SSAggVars codes for example of doing this
% properly)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

l_a=length(n_a);
l_z=length(n_z);

PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,simoptions);
% %PolicyIndexes is [num_d_vars+num_a_vars,n_a,n_s,n_z]
% if N_d==0
%     tempPolicyIndexes=reshape(PolicyIndexes,[l_a,N_a,N_z]); %first dim indexes the optimal choice for d and rest of dimensions a,z
%     PolicyIndexesKron=zeros(1,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(:,i1,i2));
%         end
%     end
% else
%     num_d_vars=length(n_d);
%     tempPolicyIndexes=reshape(PolicyIndexes,[l_a+num_d_vars,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%     PolicyIndexesKron=zeros(2,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexes(1:num_d_vars,i1,i2));
%             PolicyIndexesKron(2,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(num_d_vars+1:num_d_vars+l_a,i1,i2));
%         end
%     end
% end

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