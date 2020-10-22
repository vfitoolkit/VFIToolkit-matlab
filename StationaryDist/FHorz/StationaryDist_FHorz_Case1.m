function StationaryDist=StationaryDist_FHorz_Case1(jequaloneDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
        if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
        if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
        if isfield(simoptions,'verbose')==0
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
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end

jequaloneDistKron=reshape(jequaloneDist,[N_a*N_z,1]);
if simoptions.parallel~=2
    Policy=gather(Policy);
    jequaloneDistKron=gather(jequaloneDistKron);    
    pi_z=gather(pi_z);
end

PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j);


if simoptions.iterate==0
    if simoptions.parallel==3 || simoptions.parallel==4 
        % Sparse matrix is not relevant for the simulation methods, only for iteration method
        simoptions.parallel=simoptions.parallel-3;
    end
    StationaryDistKron=StationaryDist_FHorz_Case1_Simulation_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions);
elseif simoptions.iterate==1
    StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyKron,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions);
end

StationaryDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
