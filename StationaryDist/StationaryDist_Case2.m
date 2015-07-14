function StationaryDist=StationaryDist_Case2(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_z,pi_z,simoptions)
%Note: N_d is not actually needed, it is included to make it more like Case1 code.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if nargin<8
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=0;
    simoptions.verbose=0;
    simoptions.ncores=1;
%     simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.tolerance=10^(-9);
    simoptions.iterate=1;
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
    eval('fieldexists=1;simoptions.iterate;','fieldexists=0;')
    if fieldexists==0
        simoptions.iterate=1;
    end
end

%%

PolicyKron=KronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,simoptions);

StationaryDistKron=StationaryDist_Case2_Simulation_raw(PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z, simoptions);

if simoptions.iterate==1
    StationaryDistKron=StationaryDist_Case2_Iteration_raw(StationaryDistKron,PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z,simoptions);
end

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);


end