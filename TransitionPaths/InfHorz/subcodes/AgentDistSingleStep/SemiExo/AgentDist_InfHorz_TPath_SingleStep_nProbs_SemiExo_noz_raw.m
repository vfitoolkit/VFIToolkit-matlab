function AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemiz,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,pi_semiz)
% One period (t to t+1) of the agent distribution on an InfHorz transition path, with semi-exogenous state semiz, no z,
% no e. The states are (a,semiz), semiz varying first.
% AgentDist is [N_a*N_semiz,1] (on cpu)
% 'nProbs': each state goes to N_probs aprime points (e.g. lower and upper grid point with the grid interpolation layer)
% Policy_dsemiz is [N_a*N_semiz,1], the joint index of the decision variable(s) d2 that determine the semiz transitions
% Policy_aprime is [N_a*N_semiz,N_probs], the aprime indexes, and PolicyProbs [N_a*N_semiz,N_probs] their probabilities
% pi_semiz is [N_semiz,N_semiz,N_dsemiz] (semiz,semiz',d2), the t->t+1 transition (on cpu)
%
% A single step, from (a,semiz) to (a',semiz'). [As StationaryDist_InfHorz_SemiExo_GI_noz_raw, one iteration.]

%% Sparsity trick on pi_semiz: N_semizshort is the maximum number of non-zeros in any row of pi_semiz
pi_semiz=gather(pi_semiz);
N_semizshort=max(max(sum((pi_semiz>0),2)));
[pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix
pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:);
idxshort=idx(:,end-N_semizshort+1:end,:);

Policy_dsemiz=gather(reshape(Policy_dsemiz,[N_a*N_semiz,1]));
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_semiz,N_probs]));
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_semiz,N_probs]));

semizindex_short=repelem((1:1:N_semiz)',N_a,1)+N_semiz*(0:1:N_semizshort-1)+(N_semiz*N_semizshort)*(Policy_dsemiz-1); % [N_a*N_semiz,N_semizshort], index for (semiz,semizprime(short),dsemiz) in pi_semiz_short and idxshort
Policy_aprimesemiz=repelem(Policy_aprime,1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_probs); % columns are (semiz' short) within each of the N_probs aprime points
semiztransitions=repelem(PolicyProbs,1,N_semizshort).*repmat(pi_semiz_short(semizindex_short),1,N_probs); % probability of each aprime point times probability of each semiz'

II2=repelem((1:1:N_a*N_semiz)',1,N_semizshort*N_probs);

Gammatranspose=sparse(Policy_aprimesemiz,II2,semiztransitions,N_a*N_semiz,N_a*N_semiz); % From (a,semiz) to (a',semiz') [sparse() accumulates at repeated indices]

AgentDist=Gammatranspose*gather(AgentDist);

AgentDist=full(AgentDist); % returned full (kron with the sparse pi_e would otherwise leave it sparse), as the AggVars multiply it into a gpuArray

end
