function StationaryDist=StationaryDist_Case2_Simulation(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

%Phi_aprime is (number_a_vars,d,a,z,zprime)
%PolicyIndexes is [number_d_vars,n_a,,n_z]

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if nargin<8
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=2;
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
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
        simoptions.parallel=2;
    end
    eval('fieldexists=1;simoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        simoptions.verbose=0;
    end
    eval('fieldexists=1;simoptions.ncores;','fieldexists=0;')
    if fieldexists==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
end

%%

PolicyKron=KronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,simoptions);

StationaryDistKron=StationaryDist_Case2_Simulation_raw(PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z, simoptions);

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);

end