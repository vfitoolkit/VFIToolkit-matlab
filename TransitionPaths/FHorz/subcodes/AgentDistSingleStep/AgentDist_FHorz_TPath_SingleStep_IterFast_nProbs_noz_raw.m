function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_raw(AgentDist,Policy_aprimej,PolicyProbs,N_a,N_j,II,jequalOneDist)
% Parallelizes over age jj (age weights are handled elsewhere, here all are normalized to one)
% AgentDist is [N_a*N_j,1]
% Policy_aprime is [N_a*(N_j-1),N_probs], already except last j
% PolicyProbs is [N_a*(N_j-1),N_probs], already except last j

% precomputed
% II=repelem((1:1:N_a*(N_j-1))',1,N_probs);
% policyexceptlastj=repmat((1:1:N_a)',(N_j-1)*N_probs,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_probs,1)+repelem(N_a*N_j*(0:1:N_probs-1)',N_a*(N_j-1),1);

AgentDist_tt=sparse(gather(reshape(AgentDist(1:end-N_a),[N_a*(N_j-1),1]))); % end-N_a is avoiding those that correspond to jj=N_j

Gammatranspose=sparse(Policy_aprimej,II,PolicyProbs,N_a*(N_j-1),N_a*(N_j-1));  % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]
% Note: N_j-1, not N_j

AgentDist_tt=Gammatranspose*AgentDist_tt;

AgentDist(N_a+1:end)=gpuArray(full(AgentDist_tt)); % N_a+1 is avoiding those that correspond to jj=1
AgentDist(1:N_a)=jequalOneDist; % age j=1 dist


end
