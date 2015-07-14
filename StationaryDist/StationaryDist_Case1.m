function SteadyStateDist=SteadyState_Case1(SteadyStateDist,PolicyIndexes,n_d,n_a,n_z,pi_z,simoptions)
%If Nagents=0, then it will treat the agents as being on a continuum of
%weight 1.
%If Nagents is any other (integer), it will give the most likely of the
%distributions of that many agents across the various steady-states; this
%is for use with models that have a finite number of agents, rather than a
%continuum.

if nargin<8
    simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.tolerance=10^(-9);
%     Nagents=0;
else
    eval('fieldexists=1;simoptions.nagents;','fieldexists=0;')
    if fieldexists==0
        simoptions.nagents=0;
    end
    eval('fieldexists=1;simoptions.maxit;','fieldexists=0;')
    if fieldexists==0
        simoptions.maxit=5*10^4;
    end
    eval('fieldexists=1;simoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        simoptions.tolerance=10^(-9);
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyIndexesKron=KronPolicyIndexes_Case1(PolicyIndexes, n_d, n_a, n_z,simoptions);

% num_a_vars=length(n_a);
% 
% %PolicyIndexes is [num_d_vars+num_a_vars,n_a,n_s,n_z]
% if N_d==0
%     tempPolicyIndexes=reshape(PolicyIndexes,[num_a_vars,N_a,N_z]); %first dim indexes the optimal choice for d and rest of dimensions a,z
%     PolicyIndexesKron=zeros(1,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(:,i1,i2));
%         end
%     end
% else
%     num_d_vars=length(n_d);
%     tempPolicyIndexes=reshape(PolicyIndexes,[num_a_vars+num_d_vars,N_a,N_z]); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%     PolicyIndexesKron=zeros(2,N_a,N_z);
%     for i1=1:N_a
%         for i2=1:N_z
%             PolicyIndexesKron(1,i1,i2)=sub2ind_homemade([n_d],tempPolicyIndexes(1:num_d_vars,i1,i2));
%             PolicyIndexesKron(2,i1,i2)=sub2ind_homemade([n_a],tempPolicyIndexes(num_d_vars+1:num_d_vars+num_a_vars,i1,i2));
%         end
%     end
% end

SteadyStateDistKron=reshape(SteadyStateDist,[N_a*N_z,1]);

% tic;
SteadyStateDistKron=SteadyState_Case1_raw(SteadyStateDistKron,PolicyIndexesKron,N_d,N_a,N_z,pi_z,simoptions);
% toc

SteadyStateDist=reshape(SteadyStateDistKron,[n_a,n_z]);

end
