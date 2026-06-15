function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_noz_raw(AgentDist,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,jequalOneDist)
% fastOLG: parallelizes over age jj. One time-step of the agent distribution, with semi-exogenous state (no z, no e).
% age weights are handled elsewhere, here all are normalized to one
% AgentDist is [N_a*N_semiz*N_j,1], ordered (a,semiz,j) with a fastest
% Policy_dsemiexo and Policy_aprime are [N_a*N_semiz,N_j] (the d2 index, and the bare aprime index in 1..N_a)
% The semiz->semiz' transition (depends on d2) is folded into Gammatranspose

N_asemiz=N_a*N_semiz;

%% Sparsity trick on pi_semiz_J
N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2);
pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:));
idxshort=gather(idx(:,end-N_semizshort+1:end,:,:));

% Restrict policy to ages j=1,...,N_j-1 and flatten in (a,semiz,j) order
Policy_dsemiexo=gather(reshape(Policy_dsemiexo(:,1:N_j-1),[N_asemiz*(N_j-1),1]));
Policy_aprime=gather(reshape(Policy_aprime(:,1:N_j-1),[N_asemiz*(N_j-1),1]));

% semizindex_short indexes pi_semiz_J_short and idxshort, which are [N_semiz,N_semizshort,N_dsemiz,N_j]
semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_j-1,1)+N_semiz*(0:1:N_semizshort-1); % [N_asemiz*(N_j-1),N_semizshort]
jtermsemiz=(N_semiz*N_semizshort*N_dsemiz)*repelem((0:1:N_j-2)',N_asemiz,1); % age term for indexing pi_semiz [N_asemiz*(N_j-1),1]
destjblock=repelem(N_asemiz*(0:1:N_j-2)',N_asemiz,1); % dest age-block offset (j preserved; the j->j+1 shift is done by writing to exceptfirstj)

semizindex_short=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo-1)+jtermsemiz;
Policy_aprimesemiz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+repmat(destjblock,1,N_semizshort);
semiztransitions=pi_semiz_J_short(semizindex_short);

II2=repelem((1:1:N_asemiz*(N_j-1))',1,N_semizshort);
Gammatranspose=sparse(Policy_aprimesemiz,II2,semiztransitions,N_asemiz*(N_j-1),N_asemiz*(N_j-1)); % From (a,semiz,j) to (a',semiz',j)

% Index sets for the fastOLG age-shift
exceptlastj=(1:1:N_asemiz*(N_j-1))'; % (a,semiz,j=1..N_j-1) are the first N_asemiz*(N_j-1) entries
exceptfirstj=N_asemiz+(1:1:N_asemiz*(N_j-1))'; % (a,semiz,j=2..N_j)
justfirstj=(1:1:N_asemiz)'; % (a,semiz,j=1)

AgentDist_tt=sparse(gather(AgentDist(exceptlastj)));
AgentDist_tt=Gammatranspose*AgentDist_tt;

AgentDist(exceptfirstj)=gpuArray(full(AgentDist_tt));
AgentDist(justfirstj)=jequalOneDist; % age j=1 dist

end
