function StationaryDist=StationaryDist_Case2(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_z,pi_z,simoptions)
%Note: N_d is not actually needed, it is included to make it more like Case1 code.

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
%     simoptions.nagents=0;
    simoptions.maxit=5*10^4; %In my experience, after a simulation, if you need more that 5*10^4 iterations to reach the steady-state it is because something has gone wrong
    simoptions.tolerance=10^(-9);
    simoptions.iterate=1;
    try
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'seedpoint')==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if isfield(simoptions,'simperiods')==0
        simoptions.simperiods=10^4;
    end
    if isfield(simoptions,'burnin')==0
        simoptions.burnin=10^3;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=2;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
    if isfield(simoptions,'ncores')==0
        if simoptions.iterate==1
            try
                PoolDetails=gcp;
                simoptions.ncores=PoolDetails.NumWorkers;
            catch
                simoptions.ncores=1;
            end
        else
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions,'maxit')==0
        simoptions.maxit=5*10^4;
    end
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
end

%%

PolicyKron=KronPolicyIndexes_Case2(Policy, n_d, n_a, n_z); %,simoptions);

StationaryDistKron=StationaryDist_Case2_Simulation_raw(PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z, simoptions);

fprintf('DEBUG of StationaryDist_Case2, before iterate: sum(sum(StationaryDistKron))=%8.2f \n', sum(sum(StationaryDistKron)))

if simoptions.iterate==1
    StationaryDistKron=StationaryDist_Case2_Iteration_raw(StationaryDistKron,PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z,simoptions);
end
fprintf('DEBUG of StationaryDist_Case2, after iterate: sum(sum(StationaryDistKron))=%8.2f \n', sum(sum(StationaryDistKron)))

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);


end