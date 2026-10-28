function [AgentDistnext]=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist(AgentDist,PolicyPath_ForAgentDistIter,PolicyProbsPath,tt,N_a,N_z,N_e,N_probs,pi_z_sparse,pi_e,II1,II2,transpathoptions,simoptions)
% Iterates the agent dist from period tt to period tt+1.
% pi_z_T(:,:,tt) is the transition from tt to tt+1, so it is read at tt.
% pi_e_T(:,tt) is the distribution of the e realized IN period tt, so the e of period tt+1 is drawn from column
% tt+1 (same convention as pi_e_J in FHorz, and as the value fn in Step1).
% Sparse matrices can only be 2-D, so the time-varying ones are made sparse here, one period at a time,
% rather than being stored as a path. pi_e comes in as a (dense) column, and is made sparse here too.

if N_z>0 && transpathoptions.zpathtrivial==0
    pi_z_sparse=sparse(gather(transpathoptions.pi_z_T(:,:,tt)));
end
if N_e>0
    if transpathoptions.epathtrivial==0
        pi_e_sparse=sparse(gather(transpathoptions.pi_e_T(:,tt+1)));
    else
        pi_e_sparse=sparse(gather(pi_e));
    end
end

if N_z==0 && N_e==0
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a);
    end
elseif N_z>0 && N_e==0
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_z,pi_z_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_z,pi_z_sparse);
    end
elseif N_z==0 && N_e>0
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_e,pi_e_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_e,pi_e_sparse);
    end
elseif N_z>0 && N_e>0
    if N_probs==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,tt),II1,II2,N_a,N_z,N_e,pi_z_sparse,pi_e_sparse);
    else
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_e_raw(AgentDist,PolicyPath_ForAgentDistIter(:,:,tt),II1,PolicyProbsPath(:,:,tt),N_a,N_z,N_e,pi_z_sparse,pi_e_sparse);
    end
end






end
