function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeights,AgeWeightsOld,Policy,n_d,n_a,n_z,N_j,pi_z_J,simoptions)

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

if simoptions.fastOLG==0
    if N_e==0
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_z,N_j,pi_z_J);
    else
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_e_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J);
    end
elseif simoptions.fastOLG==1
    if N_e==1
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_z,N_j,pi_z_J);
    else
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_e_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_z,N_e,N_j,pi_z_J,simoptions.pi_e_J);
    end
end
% AgentDist=reshape(StationaryDistKron,[n_a,n_z,N_j]);

end
