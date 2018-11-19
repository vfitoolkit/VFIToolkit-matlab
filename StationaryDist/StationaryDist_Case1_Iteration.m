function StationaryDist=StationaryDist_Case1_Iteration(StationaryDist,Policy,n_d,n_a,n_z,pi_z,simoptions)

if nargin<7
%    simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.tolerance=10^(-9);
    simoptions.parallel=2;
else
%     eval('fieldexists=1;simoptions.nagents;','fieldexists=0;')
%     if fieldexists==0
%         simoptions.nagents=0;
%     end
    eval('fieldexists=1;simoptions.maxit;','fieldexists=0;')
    if fieldexists==0
        simoptions.maxit=5*10^4;
    end
    eval('fieldexists=1;simoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        simoptions.tolerance=10^(-9);
    end
    eval('fieldexists=1;simoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        simoptions.parallel=2;
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z); %,simoptions);

StationaryDistKron=reshape(StationaryDist,[N_a*N_z,1]);

% tic;
StationaryDistKron=StationaryDist_Case1_Iteration_raw(StationaryDistKron,PolicyKron,N_d,N_a,N_z,pi_z,simoptions);
% toc

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);

end
