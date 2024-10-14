function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_noz(AgentDist,AgeWeights,Policy,n_d,n_a,N_j,simoptions)

N_d=prod(n_d);
N_a=prod(n_a);

% PolicyKron=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a,N_j,simoptions);
% 
% jequaloneDistKron=reshape(AgentDist,[N_a,1]);

if simoptions.iterate==0
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Simulation_noz_raw(AgentDist,AgeWeights,Policy,N_d,N_a,N_j,simoptions);
elseif simoptions.iterate==1
    if simoptions.fastOLG==0
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_noz_raw(AgentDist,AgeWeights,Policy,N_d,N_a,N_j);
    elseif simoptions.fastOLG==1
        AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,Policy,N_d,N_a,N_j);
    end
end

% AgentDist=reshape(StationaryDistKron,[n_a,N_j]);

end
