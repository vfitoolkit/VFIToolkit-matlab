function StationaryDist=StationaryDist_Case2_Iteration(StationaryDist,Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_z,pi_z,simoptions)

if exist('simoptions','var')==0
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.tolerance=10^(-9);
    simoptions.parallel=1+(gpuDeviceCount>0);
else
    if isfield(simoptions, 'maxit')==0
        simoptions.maxit=5*10^4;
    end
    if isfield(simoptions, 'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions, 'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

PolicyKron=KronPolicyIndexes_Case2(Policy, n_d, n_a, n_z);%,simoptions);

StationaryDistKron=reshape(StationaryDist,[N_a*N_z,1]);

StationaryDistKron=StationaryDist_Case2_Iteration_raw(StationaryDistKron,PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z,simoptions);

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);


end