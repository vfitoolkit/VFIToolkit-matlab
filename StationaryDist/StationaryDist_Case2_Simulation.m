function StationaryDist=StationaryDist_Case2_Simulation(Policy,Phi_aprimeKron,Case2_Type,n_d,n_a,n_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes (and Phi_aprime) of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

%Phi_aprime is (number_a_vars,d,a,z,zprime)
%PolicyIndexes is [number_d_vars,n_a,,n_z]

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=1+(gpuDeviceCount>0);
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
    if isfield(simoptions, 'seedpoint')==0
        simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    end
    if isfield(simoptions, 'simperiods')==0
        simoptions.simperiods=10^4;
    end
    if isfield(simoptions, 'burnin')==0
        simoptions.burnin=10^3;
    end
    if isfield(simoptions, 'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions, 'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end

end

%%

PolicyKron=KronPolicyIndexes_Case2(Policy, n_d, n_a, n_z);%,simoptions);

StationaryDistKron=StationaryDist_Case2_Simulation_raw(PolicyKron,Phi_aprimeKron,Case2_Type,N_d,N_a,N_z,pi_z, simoptions);

StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);

end