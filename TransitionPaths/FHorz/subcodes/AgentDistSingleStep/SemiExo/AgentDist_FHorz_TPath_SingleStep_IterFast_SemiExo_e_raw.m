function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_SemiExo_e_raw(AgentDist,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_z,N_e,N_j,pi_semiz_J,pi_z_J_sim,pi_e_J_sim,jequalOneDist)
% fastOLG: parallelizes over age jj. One time-step of the agent distribution, with semi-exogenous state, markov z, and iid e.
% age weights are handled elsewhere, here all are normalized to one
% AgentDist is [N_a*N_semiz*N_j*N_z,N_e], ordered (a,semiz,j,z) with a fastest, e in columns
% Policy_dsemiexo and Policy_aprime are [N_a*N_semiz*N_z*N_e,N_j] in (a,semiz,z,e,j) order
% pi_z_J_sim is [(N_j-1)*N_z,(N_j-1)*N_z] block-diagonal (j,z,j',z'); pi_e_J_sim is [N_a*N_semiz*(N_j-1)*N_z,N_e]
% semiz->semiz' (depends on d2) is folded into Gammatranspose (which also sums out e); z->z' is the Tan second step; e' is added via the pi_e_J_sim multiply

N_AS=N_a*N_semiz;

%% Sparsity trick on pi_semiz_J
N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2);
pi_semiz_J_short=gather(pi_semiz_J_short(:,end-N_semizshort+1:end,:,:));
idxshort=gather(idx(:,end-N_semizshort+1:end,:,:));

% Restrict policy to ages j=1,...,N_j-1 and put in (a,semiz,j,z,e) order
Policy_dsemiexo=reshape(Policy_dsemiexo,[N_AS,N_z,N_e,N_j]);
Policy_dsemiexo=gather(reshape(permute(Policy_dsemiexo(:,:,:,1:N_j-1),[1,4,2,3]),[N_AS*(N_j-1)*N_z*N_e,1]));
Policy_aprime=reshape(Policy_aprime,[N_AS,N_z,N_e,N_j]);
Policy_aprime=gather(reshape(permute(Policy_aprime(:,:,:,1:N_j-1),[1,4,2,3]),[N_AS*(N_j-1)*N_z*N_e,1]));

% semizindex_short indexes pi_semiz_J_short and idxshort, which are [N_semiz,N_semizshort,N_dsemiz,N_j]
semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),(N_j-1)*N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1);
jtermsemiz=(N_semiz*N_semizshort*N_dsemiz)*repmat(repmat(repelem((0:1:N_j-2)',N_AS,1),N_z,1),N_e,1);
destoffset=repmat(repmat(repelem(N_AS*(0:1:N_j-2)',N_AS,1),N_z,1)+repelem(N_AS*(N_j-1)*(0:1:N_z-1)',N_AS*(N_j-1),1),N_e,1); % dest (j,z) block offset (e summed out)

semizindex_short=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo-1)+jtermsemiz;
Policy_aprimesemizz=repelem(Policy_aprime,1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+repmat(destoffset,1,N_semizshort);
semiztransitions=pi_semiz_J_short(semizindex_short);

II2=repelem((1:1:N_AS*(N_j-1)*N_z*N_e)',1,N_semizshort);
Gammatranspose=sparse(Policy_aprimesemizz,II2,semiztransitions,N_AS*(N_j-1)*N_z,N_AS*(N_j-1)*N_z*N_e); % From (a,semiz,j,z,e) to (a',semiz',j,z)

% Index sets for the fastOLG age-shift (a,semiz,j,z layout, e in columns)
exceptlastj=repmat((1:1:N_AS)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_AS*(0:1:N_j-2)',N_AS,1),N_z*N_e,1)+repelem(N_AS*N_j*(0:1:N_z*N_e-1)',N_AS*(N_j-1),1);
exceptfirstj=repmat((1:1:N_AS)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_AS*(1:1:N_j-1)',N_AS,1),N_z*N_e,1)+repelem(N_AS*N_j*(0:1:N_z*N_e-1)',N_AS*(N_j-1),1);
justfirstj=repmat((1:1:N_AS)',N_z*N_e,1)+repelem(N_AS*N_j*(0:1:N_z*N_e-1)',N_AS,1);

% Tan improvement Step 1 (e summed out by Gammatranspose)
AgentDist_tt=sparse(gather(reshape(AgentDist(exceptlastj),[N_AS*(N_j-1)*N_z*N_e,1])));
AgentDist_tt=reshape(Gammatranspose*AgentDist_tt,[N_AS,(N_j-1)*N_z]);
% Tan improvement Step 2
AgentDist_tt=reshape(AgentDist_tt*pi_z_J_sim,[N_AS*(N_j-1)*N_z,1]);

AgentDist_tt=gpuArray(full(AgentDist_tt)).*pi_e_J_sim; % put e' in

AgentDist(exceptfirstj)=AgentDist_tt;
AgentDist(justfirstj)=jequalOneDist; % age j=1 dist

end
