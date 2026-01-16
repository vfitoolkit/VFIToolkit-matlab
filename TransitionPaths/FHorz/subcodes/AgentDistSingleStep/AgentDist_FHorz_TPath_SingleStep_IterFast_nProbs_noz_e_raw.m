function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_e_raw(AgentDist,Policy_aprimej,PolicyProbs,N_a,N_e,N_j,pi_e_J_sim,II,exceptlastj,exceptfirstj,justfirstj,jequalOneDist)
% Parallelizes over age jj (age weights are handled elsewhere, here all are normalized to one)
% AgentDist is [N_a*N_j,N_e] % To be able to do Step 2 of Tan improvement it needs to be this form (note N_j then N_z,N_e)
% pi_e_J_sim is [N_a*(N_j-1),N_e]
% Policy_aprime is [N_a*(N_j-1)*N_e,N_probs], already except last j
% PolicyProbs is [N_a*(N_j-1)*N_e,N_probs], already except last j

% precomputed
% II=repelem((1:1:N_a*(N_j-1)*N_e)',1,N_probs);
% policyexceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e*N_probs,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e*N_probs,1)+repelem(N_a*N_j*(0:1:N_e*N_probs-1)',N_a*(N_j-1),1);
% exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
% exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
% justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);

% Get AgentDist for periods 1:N_j-1
AgentDist_tt=sparse(gather(reshape(AgentDist(exceptlastj),[N_a*(N_j-1)*N_e,1]))); % avoiding those that correspond to jj=N_j

Gammatranspose=sparse(Policy_aprimej,II,PolicyProbs,N_a*(N_j-1),N_a*(N_j-1)*N_e);
% Note: N_j-1, not N_j
% Note that Gamma goes from (a,j,e) to (a',j) [Gammatranspose is has these reversed]

AgentDist_tt=reshape(Gammatranspose*AgentDist_tt,[N_a*(N_j-1),1]);

AgentDist_tt=gpuArray(full(AgentDist_tt)).*pi_e_J_sim; % put e' in

AgentDist(exceptfirstj)=AgentDist_tt; % N_a+1 is avoiding those that correspond to jj=1

AgentDist(justfirstj)=jequalOneDist; % age j=1 dist


end
