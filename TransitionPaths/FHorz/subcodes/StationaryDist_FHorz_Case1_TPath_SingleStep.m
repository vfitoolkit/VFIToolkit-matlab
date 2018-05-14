function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=2;
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
        simoptions.parallel=2;
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

% PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,simoptions);
% 
% jequaloneDistKron=reshape(AgentDist,[N_a*N_z,1]);

if simoptions.iterate==0
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions);
elseif simoptions.iterate==1
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions);
end

% AgentDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
