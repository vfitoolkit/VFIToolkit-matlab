function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_SemiExo_noz_raw(AgentDist,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist)
% fastOLG: parallelizes over age jj. One time-step of the agent distribution, semi-exogenous state (no z, no e), grid interpolation layer (N_probs points).
% age weights are handled elsewhere, here all are normalized to one
% AgentDist is [N_a*N_semiz*N_j,1], ordered (a,semiz,j) with a fastest
% Policy_dsemiexo is [N_a*N_semiz,N_j]; Policy_aprime and PolicyProbs are [N_a*N_semiz,N_probs,N_j]

N_AS=N_a*N_semiz;

%% Sparsity trick on pi_semiz_J
N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2);
pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:));
idxshort=gather(idx(:,end-N_semizshort+1:end,:,:));

% Restrict policy to ages j=1,...,N_j-1
Policy_dsemiexo=gather(reshape(Policy_dsemiexo(:,1:N_j-1),[N_AS*(N_j-1),1]));
Policy_aprime=permute(reshape(Policy_aprime,[N_AS,N_probs,N_j]),[1,3,2]); % [N_AS,N_j,N_probs]
Policy_aprime=gather(reshape(Policy_aprime(:,1:N_j-1,:),[N_AS*(N_j-1),N_probs]));
PolicyProbs=permute(reshape(PolicyProbs,[N_AS,N_probs,N_j]),[1,3,2]); % [N_AS,N_j,N_probs]
PolicyProbs=gather(reshape(PolicyProbs(:,1:N_j-1,:),[N_AS*(N_j-1),N_probs]));

semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_j-1,1)+N_semiz*(0:1:N_semizshort-1);
jtermsemiz=(N_semiz*N_semizshort*N_dsemiz)*repelem((0:1:N_j-2)',N_AS,1);
destjblock=repelem(N_AS*(0:1:N_j-2)',N_AS,1);

semizindex_short=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo-1)+jtermsemiz;
Policy_aprimesemiz=repelem(Policy_aprime,1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_probs)+destjblock; % destjblock broadcasts over columns
PolicyProbs_comb=repelem(PolicyProbs,1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short),1,N_probs);

II2=repelem((1:1:N_AS*(N_j-1))',1,N_semizshort*N_probs);
Gammatranspose=sparse(Policy_aprimesemiz,II2,PolicyProbs_comb,N_AS*(N_j-1),N_AS*(N_j-1)); % sparse() accumulates at repeated indices

exceptlastj=(1:1:N_AS*(N_j-1))';
exceptfirstj=N_AS+(1:1:N_AS*(N_j-1))';
justfirstj=(1:1:N_AS)';

AgentDist_tt=sparse(gather(AgentDist(exceptlastj)));
AgentDist_tt=Gammatranspose*AgentDist_tt;

AgentDist(exceptfirstj)=gpuArray(full(AgentDist_tt));
AgentDist(justfirstj)=jequalOneDist; % age j=1 dist

end
