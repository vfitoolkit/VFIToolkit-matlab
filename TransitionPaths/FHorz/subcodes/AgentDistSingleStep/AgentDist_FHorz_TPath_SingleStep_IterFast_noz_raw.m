function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_raw(AgentDist,Policy_aprimej,N_a,N_j,II1,II2,jequalOneDist)
% Parallelizes over age jj (age weights are handled elsewhere, here all are normalized to one)
% AgentDist is [N_a*N_j,1]
% Policy_aprimej is [N_a*(N_j-1),1], already except last j

% precomputed:
% II1=1:1:N_a*(N_j-1);
% II2=ones(N_a*(N_j-1),1);
% exceptlastj=repmat((1:1:N_a)',N_j-1,1)+repelem(N_a*(0:1:N_j-2)',N_a,1);

AgentDist_tt=sparse(gather(reshape(AgentDist(1:end-N_a),[N_a*(N_j-1),1]))); % end-N_a is avoiding those that correspond to jj=N_j

Gammatranspose=sparse(Policy_aprimej,II1,II2,N_a*(N_j-1),N_a*(N_j-1));
% Note: N_j-1, not N_j

AgentDist_tt=Gammatranspose*AgentDist_tt;

AgentDist(N_a+1:end)=gpuArray(full(AgentDist_tt)); % N_a+1 is avoiding those that correspond to jj=1
AgentDist(1:N_a)=jequalOneDist; % age j=1 dist


end
