function SteadyStateDist=SteadyState_Case1_Simulation(PolicyIndexes,n_d,n_a,n_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

if nargin<6
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=0;
    simoptions.verbose=0;
    simoptions.ncores=1;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    eval('fieldexists=1;simoptions.seedpoint;','fieldexists=0;')
    if fieldexists==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    eval('fieldexists=1;simoptions.simperiods;','fieldexists=0;')
    if fieldexists==0
        simoptions.simperiods=10^4;
    end
    eval('fieldexists=1;simoptions.burnin;','fieldexists=0;')
    if fieldexists==0
        simoptions.burnin=10^3;
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=0;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
    eval('fieldexists=1;simoptions.ncores;','fieldexists=0;')
    if fieldexists==0
        simoptions.ncores=1;
    end
end

%%
PolicyIndexesKron=KronPolicyIndexes_Case1(PolicyIndexes, n_d, n_a, n_z,simoptions);
    
% l_a=length(n_a);
% %PolicyIndexes is [num_d_vars+num_a_vars,n_a,n_s,n_z]
% if N_d==0
%     tempPolicyIndexes=reshape(Policy,[l_a,N_a,N_z]); %first dim indexes the optimal choice for d and rest of dimensions a,z
%     PolicyIndexesKron=zeros(1,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(:,i1,i2));
%         end
%     end
% else
%     l_d=length(n_d);
%     tempPolicyIndexes=reshape(Policy,[l_a+l_d,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%     PolicyIndexesKron=zeros(2,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexes(1:l_d,i1,i2));
%             PolicyIndexesKron(2,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(l_d+1:l_d+l_a,i1,i2));
%         end
%     end
% end

SteadyStateDistKron=SteadyState_Case1_Simulation_raw(PolicyIndexesKron,N_d,N_a,N_z,pi_z, simoptions);
    
SteadyStateDist=reshape(SteadyStateDistKron,[n_a,n_z]);

end
