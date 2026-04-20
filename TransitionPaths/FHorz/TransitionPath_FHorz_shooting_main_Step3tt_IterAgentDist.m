function [AgentDistnext]=TransitionPath_FHorz_shooting_main_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_e,N_j,pi_z_J,pi_z_J_sim,pi_e_J,pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist,transpathoptions,simoptions)

if N_z>0 && transpathoptions.zpathtrivial==0
    if simoptions.fastOLG==0
        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
    else
        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
    end
end
if N_e>0 && transpathoptions.epathtrivial==0
    if simoptions.fastOLG==0
        pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
    else
        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt); % (a,j,z)-by-e
    end
end

if N_z==0 && N_e==0
    if simoptions.fastOLG==0
        % PolicyaprimePath_slowOLG
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),N_a,N_j,II1,II2,jequalOneDist);
    else % simoptions.fastOLG==1
        % PolicyaprimejPath
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),N_a,N_j,II1,II2,jequalOneDist);
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_j,II1,jequalOneDist);
        end
    end
elseif N_z>0 && N_e==0
    if simoptions.fastOLG==0
        % PolicyaprimezPath_slowOLG
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_Iteration_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),N_a,N_z,N_j,pi_z_J,II1,II2,jequalOneDist);
    else % simoptions.fastOLG==1
        % PolicyaprimejzPath
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),N_a,N_z,N_j,pi_z_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_z,N_j,pi_z_J_sim,II1,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        end
    end
elseif N_z==0 && N_e>0
    if simoptions.fastOLG==0
        % PolicyaprimePath_slowOLG
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),N_a,N_e,N_j,pi_e_J,II1,II2,jequalOneDist);
    else % simoptions.fastOLG==1
        % PolicyaprimejPath
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),N_a,N_e,N_j,pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_e,N_j,pi_e_J_sim,II1,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        end
    end
elseif N_z>0 && N_e>0
    if simoptions.fastOLG==0
        % PolicyaprimezPath_slowOLG
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_Iteration_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,II1,II2,jequalOneDist);
    else % simoptions.fastOLG==1
        % PolicyaprimejzPath
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim, pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,II1,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        end
    end
end






end