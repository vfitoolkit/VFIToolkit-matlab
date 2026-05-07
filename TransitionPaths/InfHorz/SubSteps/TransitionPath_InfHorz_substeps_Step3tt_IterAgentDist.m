function [AgentDistnext]=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_e,N_probs,pi_z_sparse,pi_e,II1,II2,transpathoptions,simoptions)

if N_z>0 && transpathoptions.zpathtrivial==0
    pi_z_sparse=transpathoptions.pi_z_sparse_T(:,:,tt);
end
if N_e>0 && transpathoptions.epathtrivial==0
    pi_e_sparse=transpathoptions.pi_e_sparse_T(:,tt);
end

if N_z==0 && N_e==0
    error('Not yet implemented')
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_e,pi_e_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_e,pi_e_sparse);
    end
elseif N_z>0 && N_e==0
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_z,pi_z_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_z,pi_z_sparse);
    end
elseif N_z==0 && N_e>0
    error('Not yet implemented')
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_e,pi_e_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_e,pi_e_sparse);
    end
elseif N_z>0 && N_e>0
    error('Not yet implemented')
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_z,N_e,pi_z_sparse,pi_e_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_z,N_e,pi_z_sparse,pi_e_sparse);
    end
end






end
