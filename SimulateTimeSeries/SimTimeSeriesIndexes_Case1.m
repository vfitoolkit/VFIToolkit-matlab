function SimTimeSeries=SimTimeSeriesIndexes_Case1(Policy,n_d,n_a,n_z,pi_z, simoptions)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which vfoptions have been used, set all others to defaults 
if nargin<6
    %If vfoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.burnin=1000;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10000;
    simoptions.parallel=0;
    simoptions.verbose=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        simoptions.polindorval=1;
    end
    eval('fieldexists=1;simoptions.burnin;','fieldexists=0;')
    if fieldexists==0
        simoptions.burnin=1000;
    end
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=0;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=0;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
end

%Simulates a path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped). The burn in begins from point 'seedpoint' (this is not just left
%as being random since some random points may be ones that never 'exist' in
%eqm)

l_a=length(n_a);
l_z=length(n_z);

%Policy is [l_d+l_a,n_a,n_s,n_z]
PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,simoptions);
% if N_d==0
%     tempPolicyIndexes=reshape(Policy,[l_a,N_a,N_z]); %first dim indexes the optimal choice for d and rest of dimensions a,z
%     PolicyIndexesKron=zeros(1,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(:,i1,i2));
%         end
%     end
% else
%     num_d_vars=length(n_d);
%     tempPolicyIndexes=reshape(Policy,[l_a+num_d_vars,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%     PolicyIndexesKron=zeros(2,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexes(1:num_d_vars,i1,i2));
%             PolicyIndexesKron(2,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(num_d_vars+1:num_d_vars+l_a,i1,i2));
%         end
%     end
% end

seedtemp=sub2ind_homemade([n_a,n_z],simoptions.seedpoint);
seedpoint=ind2sub_homemade([N_a,N_z],seedtemp);

SimTimeSeriesKron=SimTimeSeriesIndexes_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,pi_z, simoptions.burnin, simoptions.seedpoint, simoptions.simperiods,simoptions.parallel);

if simoptions.parallel~=2
    SimTimeSeries=zeros(l_a+l_z,simoptions.simperiods);
    for t=1:simoptions.simperiods
        temp=SimTimeSeriesKron(:,t);
        a_c_vec=ind2sub_homemade([n_a],temp(1));
        z_c_vec=ind2sub_homemade([n_z],temp(2));
        for i=1:l_a
            SimTimeSeries(i,t)=a_c_vec(i);
        end
        for i=1:l_z
            SimTimeSeries(l_a+i,t)=z_c_vec(i);
        end
    end
elseif simoptions.parallel==2
    SimTimeSeries=zeros(l_a+l_z,simoptions.simperiods,'gpuArray');
    for t=1:simoptions.simperiods
        temp=SimTimeSeriesKron(:,t);
        a_c_vec=ind2sub_homemade_gpu([n_a],temp(1));
        z_c_vec=ind2sub_homemade_gpu([n_z],temp(2));
        for i=1:l_a
            SimTimeSeries(i,t)=a_c_vec(i);
        end
        for i=1:l_z
            SimTimeSeries(l_a+i,t)=z_c_vec(i);
        end
    end
end


end



