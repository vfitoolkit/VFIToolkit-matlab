function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_raw(AgentDist,Policy_aprimejz,N_a,N_z,N_j,pi_z_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist)
% Parallelizes over age jj (age weights are handled elsewhere, here all are normalized to one)
% AgentDist is [N_a*N_j*N_z,1] % To be able to do Step 2 of Tan improvement it needs to be this form (note N_j then N_z)
% pi_z_J_sim is [(N_j-1)*N_z,(N_j-1)*N_z] (j,z,j',z')
% Policy_aprimejz is [N_a*(N_j-1)*N_z,1], already except last j

% precomputed:
% II1=1:1:N_a*(N_j-1)*N_z;
% II2=ones(N_a*(N_j-1)*N_z,1);
% exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
% exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
% justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);

% Get AgentDist for periods 1:N_j-1
AgentDist_tt=sparse(gather(reshape(AgentDist(exceptlastj),[N_a*(N_j-1)*N_z,1]))); % end-N_a*N_z is avoiding those that correspond to jj=N_j

% Tan improvement Step 1
Gammatranspose=sparse(Policy_aprimejz,II1,II2,N_a*(N_j-1)*N_z,N_a*(N_j-1)*N_z);
% Note: N_j-1, not N_j

AgentDist_tt=reshape(Gammatranspose*AgentDist_tt,[N_a,(N_j-1)*N_z]);

% Tan improvement Step 2

% NOTE: Following four lines are precomputed and pi_z_J_sim is then passed as input
% pi_z_J_sim=gather(reshape(permute(pi_z_J(:,:,1:end-1),[3,1,2]),[(N_j-1)*N_z,N_z]));
% II3=repmat(1:1:(N_j-1)*N_z,1,N_z);
% II4=repmat(1:1:(N_j-1),1,N_z*N_z)+repelem((N_j-1)*(0:1:N_z-1),1,N_z*(N_j-1));
% pi_z_J_sim=sparse(II3,II4,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);  

% Note, we just construct a block-diagonal, the blocks are the (z,z'). The diagonal is j. 
% Because we are going from an agent dist on j=1,...,N_j-1 to an agent dist
% on j=2,...,N_j the diagonal for j actually corresponds to deterministic ageing
AgentDist_tt=reshape(AgentDist_tt*pi_z_J_sim,[N_a*(N_j-1)*N_z,1]);

AgentDist(exceptfirstj)=gpuArray(full(AgentDist_tt)); % N_a*N_z+1 is avoiding those that correspond to jj=1

AgentDist(justfirstj)=jequalOneDist; % Age j=1 dist


end
