function AgentDist=AgentDist_InfHorz_TPath_SingleStep_SemiExo_e_raw(AgentDist,Policy_dsemiz,Policy_aprime,N_dsemiz,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z_sparse,pi_e_sparse)
% One period (t to t+1) of the agent distribution on an InfHorz transition path, with semi-exogenous state semiz,
% markov z and iid e. The states are (a,semiz,z,e), semiz varying first, then z, then e.
% AgentDist is [N_a*N_semiz*N_z*N_e,1] (on cpu)
% Policy_dsemiz and Policy_aprime are [N_a*N_semiz*N_z*N_e,1]: the joint index of the decision variable(s) d2 that
% determine the semiz transitions, and the aprime index
% pi_semiz is [N_semiz,N_semiz,N_dsemiz] (semiz,semiz',d2), the t->t+1 transition (on cpu)
% pi_z_sparse is [N_z,N_zprime] (sparse, on cpu), the t->t+1 transition
% pi_e_sparse is [N_e,1] (sparse, on cpu), the distribution of the e realized in period t+1
%
% Tan improvement: the first step maps (a,semiz,z,e) to (a',semiz',z), folding in semiz->semiz' (which depends on d2)
% and summing out e, the second is the standard z to z', and the third puts e' back in (kron(pi_e,.), as e is iid).
% [As StationaryDist_InfHorz_SemiExo_e_raw, one iteration.]

%% Sparsity trick on pi_semiz: N_semizshort is the maximum number of non-zeros in any row of pi_semiz
pi_semiz=gather(pi_semiz);
N_semizshort=max(max(sum((pi_semiz>0),2)));
[pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix
pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:);
idxshort=idx(:,end-N_semizshort+1:end,:);

N_bothz=N_semiz*N_z;
Policy_dsemiz=gather(reshape(Policy_dsemiz,[N_a*N_bothz*N_e,1]));
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_bothz*N_e,1]));

semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1)+(N_semiz*N_semizshort)*(Policy_dsemiz-1); % [N_a*N_bothz*N_e,N_semizshort], index for (semiz,semizprime(short),dsemiz) in pi_semiz_short and idxshort
zoffset=repmat(repelem(N_a*N_semiz*(0:1:N_z-1)',N_a*N_semiz,1),N_e,1); % [Tan improvement, z stays where it is in the first step, e goes]
Policy_aprimesemizz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+zoffset; % add semiz' index, and z index for Tan improvement
semiztransitions=pi_semiz_short(semizindex_short);

II2=repelem((1:1:N_a*N_bothz*N_e)',1,N_semizshort);

Gammatranspose=sparse(Policy_aprimesemizz,II2,semiztransitions,N_a*N_bothz,N_a*N_bothz*N_e); % From (a,semiz,z,e) to (a',semiz',z)

% Three steps of the Tan improvement
AgentDist=reshape(Gammatranspose*gather(AgentDist),[N_a*N_semiz,N_z]);
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_bothz,1]);
AgentDist=kron(pi_e_sparse,AgentDist); % [N_a*N_bothz*N_e,1], next period's e is independent of everything

AgentDist=full(AgentDist); % returned full (kron with the sparse pi_e would otherwise leave it sparse), as the AggVars multiply it into a gpuArray

end
