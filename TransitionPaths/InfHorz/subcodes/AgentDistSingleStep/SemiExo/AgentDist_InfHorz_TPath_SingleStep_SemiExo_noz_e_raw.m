function AgentDist=AgentDist_InfHorz_TPath_SingleStep_SemiExo_noz_e_raw(AgentDist,Policy_dsemiz,Policy_aprime,N_dsemiz,N_a,N_semiz,N_e,pi_semiz,pi_e_sparse)
% One period (t to t+1) of the agent distribution on an InfHorz transition path, with semi-exogenous state semiz and
% iid e, no z. The states are (a,semiz,e), semiz varying first.
% AgentDist is [N_a*N_semiz*N_e,1] (on cpu)
% Policy_dsemiz and Policy_aprime are [N_a*N_semiz*N_e,1]: the joint index of the decision variable(s) d2 that
% determine the semiz transitions, and the aprime index
% pi_semiz is [N_semiz,N_semiz,N_dsemiz] (semiz,semiz',d2), the t->t+1 transition (on cpu)
% pi_e_sparse is [N_e,1] (sparse, on cpu), the distribution of the e realized in period t+1
%
% Two steps: from (a,semiz,e) to (a',semiz') summing out e, then put e' back in (kron(pi_e,.), as e is iid).
% [As StationaryDist_InfHorz_SemiExo_noz_e_raw, one iteration.]

%% Sparsity trick on pi_semiz: N_semizshort is the maximum number of non-zeros in any row of pi_semiz
pi_semiz=gather(pi_semiz);
N_semizshort=max(max(sum((pi_semiz>0),2)));
[pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix
pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:);
idxshort=idx(:,end-N_semizshort+1:end,:);

Policy_dsemiz=gather(reshape(Policy_dsemiz,[N_a*N_semiz*N_e,1]));
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_semiz*N_e,1]));

semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1)+(N_semiz*N_semizshort)*(Policy_dsemiz-1); % [N_a*N_semiz*N_e,N_semizshort], index for (semiz,semizprime(short),dsemiz) in pi_semiz_short and idxshort
Policy_aprimesemiz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1); % add semiz' index
semiztransitions=pi_semiz_short(semizindex_short);

II2=repelem((1:1:N_a*N_semiz*N_e)',1,N_semizshort);

Gammatranspose=sparse(Policy_aprimesemiz,II2,semiztransitions,N_a*N_semiz,N_a*N_semiz*N_e); % From (a,semiz,e) to (a',semiz')

AgentDist=Gammatranspose*gather(AgentDist);
AgentDist=kron(pi_e_sparse,AgentDist); % [N_a*N_semiz*N_e,1], next period's e is independent of everything

AgentDist=full(AgentDist); % returned full (kron with the sparse pi_e would otherwise leave it sparse), as the AggVars multiply it into a gpuArray

end
