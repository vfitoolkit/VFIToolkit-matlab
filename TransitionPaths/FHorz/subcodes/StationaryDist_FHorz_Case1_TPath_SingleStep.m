function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=0;
if isfield(simoptions,'n_e')
    N_e=prod(simoptions.n_e);
end


% PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j,simoptions);
% 
% jequaloneDistKron=reshape(AgentDist,[N_a*N_z,1]);

if N_e==0
    if simoptions.iterate==0
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions);
    elseif simoptions.iterate==1
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions);
    end
else
    if simoptions.iterate==0
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_e_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_e,N_j,pi_z,simoptions.pi_e,Parameters,simoptions);
    elseif simoptions.iterate==1
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_e_raw(AgentDist,AgeWeightParamNames,Policy,N_d,N_a,N_z,N_e,N_j,pi_z,simoptions.pi_e,Parameters,simoptions);
    end
end

% AgentDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
