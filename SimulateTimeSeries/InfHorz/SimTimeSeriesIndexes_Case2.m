function SteadyStateDist=SimTimeSeriesIndexes_Case2(seedpoint,periods, burnin, PolicyIndexes,Phi_aprimeKron, Case2_Type, n_d,n_a,n_z,pi_z)
%Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

%Phi_aprimeKron is (num_a_vars,d,z,zprime)
%PolicyIndexes is [num_d_vars,n_a,,n_z]

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
num_d_vars=length(n_d);

tempPolicyIndexes=reshape(PolicyIndexes,[num_d_vars,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
PolicyIndexesKron=zeros(N_a,N_z);
for i1=1:N_a
    for i2=1:N_z
        PolicyIndexesKron(i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexes(:,i1,i2));
    end
end

seedtemp=sub2ind_homemade([n_a,n_z],seedpoint);
seedpoint=ind2sub_homemade([N_a,N_z],seedtemp);

SimTimeSeriesKron=SimTimeSeriesIndexes_Case2_raw(seedpoint,periods, burnin, PolicyIndexesKron,Phi_aprimeKron, Case2_Type,N_d,N_a,N_z,pi_z);

SimTimeSeries=zeros(num_a_vars+num_z_vars,periods);
for t=1:periods
    temp=SimTimeSeriesKron(:,t);
    a_c_vec=ind2sub_homemade([n_a],temp(1));
    for i=1:num_a_vars
        SimTimeSeries(i,t)=a_c_vec(i);
    end
    z_c_vec=ind2sub_homemade([n_z],temp(2));
    for i=1:num_z_vars
        SimTimeSeries(num_a_vars+i,t)=z_c_vec(i);
    end
end

end