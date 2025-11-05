function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_e_raw(AgentDist,Policy_aprime,N_a,N_e,N_j,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist)
% Parallelizes over age jj (age weights are handled elsewhere, here all are normalized to one)
% AgentDist is [N_a*N_j,N_e] % To be able to do Step 2 of Tan improvement it needs to be this form (note N_j then N_z,N_e)
% pi_e_J_sim is [N_a*(N_j-1),N_e]
% Policy_aprime is [1,N_a*(N_j-1)*N_e]

% Get AgentDist for periods 1:N_j-1
% exceptlastj=kron(ones(1,(N_j-1)),1:1:N_a)+kron(N_a*(0:1:N_j-2),ones(1,N_a)); % Note: there is one use of N_j which is because we want to index AgentDist
AgentDist_tt=sparse(gather(reshape(AgentDist(exceptlastj),[N_a*(N_j-1)*N_e,1]))); % avoiding those that correspond to jj=N_j

firststep=Policy_aprime+N_a*repmat(repelem((0:1:N_j-2),1,N_a),1,N_e);
Gammatranspose=sparse(firststep,1:1:N_a*(N_j-1)*N_e,ones(N_a*(N_j-1)*N_e,1),N_a*(N_j-1),N_a*(N_j-1)*N_e);
% Note: N_j-1, not N_j
% Note that Gamma goes from (a,j,e) to (a',j) [Gammatranspose is has these reversed]

AgentDist_tt=reshape(Gammatranspose*AgentDist_tt,[N_a*(N_j-1),1]);

AgentDist_tt=gpuArray(full(AgentDist_tt)).*pi_e_J_sim; % put e' in

% exceptfirstj=kron(ones(1,(N_j-1)),1:1:N_a)+kron(N_a*(1:1:N_j-1),ones(1,N_a)); % Note: there is one use of N_j which is because we want to index AgentDist
AgentDist(exceptfirstj)=AgentDist_tt; % N_a+1 is avoiding those that correspond to jj=1

AgentDist(justfirstj)=jequalOneDist; % age j=1 dist


end
