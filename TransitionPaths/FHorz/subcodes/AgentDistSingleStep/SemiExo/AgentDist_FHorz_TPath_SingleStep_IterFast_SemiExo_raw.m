function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_raw(AgentDist,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J_sim,jequalOneDist)
% fastOLG: parallelizes over age jj. One time-step of the agent distribution, with semi-exogenous state and markov z (no e).
% age weights are handled elsewhere, here all are normalized to one
% AgentDist is [N_a*N_semiz*N_j*N_z,1], ordered (a,semiz,j,z) with a fastest
% Policy_dsemiexo and Policy_aprime are [N_a*N_semiz*N_z,N_j] in (a,semiz,z,j) order (semiz fastest, then z)
% pi_z_J_sim is [(N_j-1)*N_z,(N_j-1)*N_z] block-diagonal (j,z,j',z'), as built for the standard fastOLG z raw
% semiz->semiz' (depends on d2) is folded into Gammatranspose; z->z' is the Tan-improvement second step

N_asemiz=N_a*N_semiz;

%% Sparsity trick on pi_semiz_J
N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2);
pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:));
idxshort=gather(idx(:,end-N_semizshort+1:end,:,:));

% Restrict policy to ages j=1,...,N_j-1 and put in (a,semiz,j,z) order
Policy_dsemiexo=reshape(Policy_dsemiexo,[N_asemiz,N_z,N_j]);
Policy_dsemiexo=gather(reshape(permute(Policy_dsemiexo(:,:,1:N_j-1),[1,3,2]),[N_asemiz*(N_j-1)*N_z,1]));
Policy_aprime=reshape(Policy_aprime,[N_asemiz,N_z,N_j]);
Policy_aprime=gather(reshape(permute(Policy_aprime(:,:,1:N_j-1),[1,3,2]),[N_asemiz*(N_j-1)*N_z,1]));

% semizindex_short indexes pi_semiz_J_short and idxshort, which are [N_semiz,N_semizshort,N_dsemiz,N_j]
semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),(N_j-1)*N_z,1)+N_semiz*(0:1:N_semizshort-1); % [N_asemiz*(N_j-1)*N_z,N_semizshort]
jtermsemiz=(N_semiz*N_semizshort*N_dsemiz)*repmat(repelem((0:1:N_j-2)',N_asemiz,1),N_z,1); % age term for indexing pi_semiz
destoffset=repmat(repelem(N_asemiz*(0:1:N_j-2)',N_asemiz,1),N_z,1)+repelem(N_asemiz*(N_j-1)*(0:1:N_z-1)',N_asemiz*(N_j-1),1); % dest (j,z) block offset

semizindex_short=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo-1)+jtermsemiz;
Policy_aprimesemizz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+repmat(destoffset,1,N_semizshort);
semiztransitions=pi_semiz_J_short(semizindex_short);

II2=repelem((1:1:N_asemiz*(N_j-1)*N_z)',1,N_semizshort);
Gammatranspose=sparse(Policy_aprimesemizz,II2,semiztransitions,N_asemiz*(N_j-1)*N_z,N_asemiz*(N_j-1)*N_z); % From (a,semiz,j,z) to (a',semiz',j,z)

% Index sets for the fastOLG age-shift (a,semiz,j,z layout)
exceptlastj=repmat((1:1:N_asemiz)',(N_j-1)*N_z,1)+repmat(repelem(N_asemiz*(0:1:N_j-2)',N_asemiz,1),N_z,1)+repelem(N_asemiz*N_j*(0:1:N_z-1)',N_asemiz*(N_j-1),1);
exceptfirstj=repmat((1:1:N_asemiz)',(N_j-1)*N_z,1)+repmat(repelem(N_asemiz*(1:1:N_j-1)',N_asemiz,1),N_z,1)+repelem(N_asemiz*N_j*(0:1:N_z-1)',N_asemiz*(N_j-1),1);
justfirstj=repmat((1:1:N_asemiz)',N_z,1)+repelem(N_asemiz*N_j*(0:1:N_z-1)',N_asemiz,1);

% Tan improvement Step 1
AgentDist_tt=sparse(gather(AgentDist(exceptlastj)));
AgentDist_tt=reshape(Gammatranspose*AgentDist_tt,[N_asemiz,(N_j-1)*N_z]);
% Tan improvement Step 2
AgentDist_tt=reshape(AgentDist_tt*pi_z_J_sim,[N_asemiz*(N_j-1)*N_z,1]);

AgentDist(exceptfirstj)=gpuArray(full(AgentDist_tt));
AgentDist(justfirstj)=jequalOneDist; % age j=1 dist

end
