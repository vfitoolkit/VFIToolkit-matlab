function SimLifeCycle=SimLifeCycleIndexes_FHorz_Case1(Policy,n_d,n_a,n_z,N_j,pi_z, simoptions)
% Simulates a path based on PolicyIndexes of length 'periods' beginning from point 'seedpoint' (this is not just left
% as being random since some random points may be ones that never 'exist'
% in eqm).

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

%% Check which vfoptions have been used, set all others to defaults 
if nargin<6
    %If vfoptions is not given, just use all the defaults
    simoptions.polindorval=1;
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2),1];
    simoptions.simperiods=N_j;
    simoptions.parallel=2;
    simoptions.verbose=0;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.polindorval;','fieldexists=0;')
    if fieldexists==0
        simoptions.polindorval=1;
    end
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2),1];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=N_j;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
end


l_a=length(n_a);
l_z=length(n_z);

%Policy is [l_d+l_a,n_a,n_s,n_z]
PolicyIndexesKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,simoptions);

%seedtemp=sub2ind_homemade([n_a,n_z],simoptions.seedpoint);
%seedpoint=ind2sub_homemade([N_a,N_z],seedtemp);

MoveOutputtoGPU=0;
if simoptions.parallel==2
    % Simulation on GPU is really slow. So instead, switch to CPU, and then switch
    % back. For anything but ridiculously short simulations it is more than worth the overhead.
    PolicyIndexesKron=gather(PolicyIndexesKron);
    pi_z=gather(pi_z);
    MoveOutputtoGPU=1;
end

SimLifeCycleKron=SimLifeCycleIndexes_FHorz_Case1_raw(PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z, simoptions.burnin, simoptions.seedpoint, simoptions.simperiods);

SimLifeCycle=zeros(l_a+l_z,simoptions.simperiods);
for t=1:simoptions.simperiods
    temp=SimLifeCycleKron(:,t);
    a_c_vec=ind2sub_homemade([n_a],temp(1));
    z_c_vec=ind2sub_homemade([n_z],temp(2));
    for i=1:l_a
        SimLifeCycle(i,t)=a_c_vec(i);
    end
    for i=1:l_z
        SimLifeCycle(l_a+i,t)=z_c_vec(i);
    end
end

if MoveOutputtoGPU==1
    SimLifeCycle=gpuArray(SimLifeCycle);
end


end



