function AgentDist=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist_SemiExo(AgentDist,Policy_dsemiexoPath,Policy_aprimePath,PolicyProbsPath,tt,N_a,N_z,N_e,N_j,N_probs,pi_z_J,pi_z_J_sim,pi_e_J,pi_e_J_sim,pi_semiz_J,jequalOneDist,transpathoptions,simoptions)
% Semi-exogenous state: one time-step (tt) forward of the agent distribution, calling the SemiExo single-step dist raws.
% The semiz transition (semiz->semiz') depends on the decision d2, so it is folded into the iteration (cannot be a fixed markov step).
% AgentDist is in the SemiExo layout produced by the previous step (slowOLG: [N_a*N_bothze,N_j]; fastOLG: (a,semiz,j,z) with e trailing).

n_semiz=simoptions.n_semiz;
N_semiz=prod(n_semiz);
N_asemiz=N_a*N_semiz;
N_dsemiz=simoptions.setup_semiexo.N_dsemiz;

if transpathoptions.semizpathtrivial==0
    pi_semiz_J=transpathoptions.pi_semiz_J_sim_T(:,:,:,:,tt); % standard form (semiz,semiz',d2,j); both dist raw families use the standard form regardless of fastOLG
end
if N_z>0 && transpathoptions.zpathtrivial==0
    if simoptions.fastOLG==0
        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt); % (z,z',j) standard form [fastOLG=0, so no appended row]
    else
        pi_z_J_sim=transpathoptions.pi_z_J_sim_T{tt};
    end
end
if N_e>0 && transpathoptions.epathtrivial==0
    % This must come before the pi_e_J gather / pi_e_J_sim repelem below, which act on the per-period arrays
    if simoptions.fastOLG==0
        pi_e_J=transpathoptions.pi_e_J_T(:,:,tt); % [N_e,N_j]
    else
        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt); % generic N_a-sized version, expanded to N_asemiz below
    end
end

% pi_e_J [N_e,N_j] is used by the slowOLG raws; pi_e_J_sim (with N_asemiz) by the fastOLG raws.
% The passed-in pi_e_J_sim is the generic N_a-sized version; rebuild with N_asemiz via repelem.
% (repelem by N_semiz preserves correctness because pi_e is constant within each age block.)
if N_e>0
    if simoptions.fastOLG==0
        pi_e_J=gather(pi_e_J); % [N_e,N_j]
    else
        pi_e_J_sim=gpuArray(repelem(pi_e_J_sim,N_semiz,1)); % [N_a*(N_j-1)(*N_z),N_e] -> [N_asemiz*(N_j-1)(*N_z),N_e]
    end
end

Policy_dsemiexo_tt=Policy_dsemiexoPath(:,:,tt);

if simoptions.fastOLG==0
    %% slowOLG: AgentDist panel [N_a*N_bothze,N_j]
    if simoptions.gridinterplayer==0
        Policy_aprime_tt=Policy_aprimePath(:,:,tt);
        if N_z==0 && N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist);
        elseif N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,jequalOneDist);
        elseif N_z==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,jequalOneDist);
        else
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,jequalOneDist);
        end
    else % gridinterplayer==1
        Policy_aprime_tt=Policy_aprimePath(:,:,:,tt);
        PolicyProbs_tt=PolicyProbsPath(:,:,:,tt);
        if N_z==0 && N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist);
        elseif N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,jequalOneDist);
        elseif N_z==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,jequalOneDist);
        else
            AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J,pi_e_J,jequalOneDist);
        end
    end
else
    %% fastOLG
    if simoptions.gridinterplayer==0
        Policy_aprime_tt=Policy_aprimePath(:,:,tt);
        if N_z==0 && N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist);
        elseif N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J_sim,jequalOneDist);
        elseif N_z==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J_sim,jequalOneDist);
        else
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J_sim,pi_e_J_sim,jequalOneDist);
        end
    else % gridinterplayer==1
        Policy_aprime_tt=Policy_aprimePath(:,:,:,tt);
        PolicyProbs_tt=PolicyProbsPath(:,:,:,tt);
        if N_z==0 && N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist);
        elseif N_e==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J_sim,jequalOneDist);
        elseif N_z==0
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_noz_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J_sim,jequalOneDist);
        else
            AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_e_raw(AgentDist,Policy_dsemiexo_tt,Policy_aprime_tt,PolicyProbs_tt,N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J_sim,pi_e_J_sim,jequalOneDist);
        end
    end
end

end
