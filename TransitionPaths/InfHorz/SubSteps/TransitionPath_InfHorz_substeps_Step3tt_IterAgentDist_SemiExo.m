function AgentDistnext=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist_SemiExo(AgentDist,Policy_dsemizPath,Policy_aprimePath,PolicyProbsPath,tt,N_a,N_z,N_e,pi_z_sparse,pi_e,transpathoptions,simoptions)
% Semi-exogenous state: iterates the agent dist from period tt to period tt+1.
% AgentDist is [N_a*N_semiz*max(N_z,1)*max(N_e,1),1] over (a,semiz,z,e), semiz fastest (cpu).
% (a,semiz,z,e) -> (a',semiz,z) via the policy, then semiz' via pi_semiz(semiz,semiz',d2(state)), then z' via pi_z, then e'
% via kron(pi_e,.). The first two are a single sparse matrix (the semiz transition depends on d2, so it cannot be a fixed
% markov step).
% Timing: pi_semiz_T(:,:,:,tt) and pi_z_T(:,:,tt) are the transitions from tt to tt+1, so are read at tt.
% pi_e_T(:,tt) is the distribution of the e realized IN period tt, so the e of period tt+1 is drawn from column tt+1.
% Policy_dsemizPath, Policy_aprimePath, PolicyProbsPath are from TransitionPath_InfHorz_substeps_Step2_AdjustPolicy_SemiExo;
% PolicyProbsPath is [] without the grid interpolation layer.
% Read from simoptions: n_semiz, setup_semiexo.N_dsemiz, setup_semiexo.pi_semiz [N_semiz,N_semiz,N_dsemiz] (cpu).

N_semiz=prod(simoptions.n_semiz);
N_dsemiz=simoptions.setup_semiexo.N_dsemiz;

if transpathoptions.semizpathtrivial==0
    pi_semiz=gather(transpathoptions.pi_semiz_T(:,:,:,tt));
else
    pi_semiz=simoptions.setup_semiexo.pi_semiz;
end
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

if isempty(PolicyProbsPath)
    if N_z==0 && N_e==0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_SemiExo_noz_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,pi_semiz);
    elseif N_z>0 && N_e==0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_SemiExo_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,N_z,pi_semiz,pi_z_sparse);
    elseif N_z==0 && N_e>0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_SemiExo_noz_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,N_e,pi_semiz,pi_e_sparse);
    elseif N_z>0 && N_e>0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_SemiExo_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,tt),N_dsemiz,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z_sparse,pi_e_sparse);
    end
else
    N_probs=size(PolicyProbsPath,2);
    if N_z==0 && N_e==0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),N_probs,N_dsemiz,N_a,N_semiz,pi_semiz);
    elseif N_z>0 && N_e==0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),N_probs,N_dsemiz,N_a,N_semiz,N_z,pi_semiz,pi_z_sparse);
    elseif N_z==0 && N_e>0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_noz_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),N_probs,N_dsemiz,N_a,N_semiz,N_e,pi_semiz,pi_e_sparse);
    elseif N_z>0 && N_e>0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_e_raw(AgentDist,Policy_dsemizPath(:,tt),Policy_aprimePath(:,:,tt),PolicyProbsPath(:,:,tt),N_probs,N_dsemiz,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z_sparse,pi_e_sparse);
    end
end

end
