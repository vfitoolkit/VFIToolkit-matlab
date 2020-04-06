function StationaryDist=StationaryDist_Case1_Simulation(Policy,n_d,n_a,n_z,pi_z, simoptions)
%Simulates a path based on PolicyIndexes of length 'periods' after a burn
%in of length 'burnin' (burn-in are the initial run of points that are then
%dropped)

N_a=prod(n_a);
N_z=prod(n_z);
N_d=prod(n_d);

if exist('simoptions','var')==0
    simoptions.seedpoint=[ceil(N_a/2),ceil(N_z/2)];
    simoptions.simperiods=10^4;
    simoptions.burnin=10^3;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    if simoptions.parallel>0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
else
    %Check simoptions for missing fields, if there are some fill them with the defaults
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
    if simoptions.parallel>0 && isfield(simoptions,'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
end

%%
PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z); %,simoptions);

StationaryDistKron=StationaryDist_Case1_Simulation_raw(PolicyKron,N_d,N_a,N_z,pi_z, simoptions);
    
StationaryDist=reshape(StationaryDistKron,[n_a,n_z]);

end
