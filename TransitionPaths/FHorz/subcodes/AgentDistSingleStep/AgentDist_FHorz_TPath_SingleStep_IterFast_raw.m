function AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_raw(AgentDist,Policy_aprime,N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist)
% Parallelizes over age jj (age weights are handled elsewhere, here all are normalized to one)
% AgentDist is [N_a*N_j*N_z,1] % To be able to do Step 2 of Tan improvement it needs to be this form (note N_j then N_z)
% pi_z_J_sim is [(N_j-1)*N_z,N_z] (j,z,z')
% Policy_aprime is [1,N_a*(N_j-1)*N_z]

% Get AgentDist for periods 1:N_j-1
% exceptlastj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
AgentDist_tt=sparse(gather(reshape(AgentDist(exceptlastj),[N_a*(N_j-1)*N_z,1]))); % end-N_a*N_z is avoiding those that correspond to jj=N_j

% Tan improvement Step 1
firststep=Policy_aprime+kron(N_a*(0:1:(N_j-1)*N_z-1),ones(1,N_a));
Gammatranspose=sparse(firststep,1:1:N_a*(N_j-1)*N_z,ones(N_a*(N_j-1)*N_z,1),N_a*(N_j-1)*N_z,N_a*(N_j-1)*N_z);
% Note: N_j-1, not N_j

AgentDist_tt=reshape(Gammatranspose*AgentDist_tt,[N_a,(N_j-1)*N_z]);

% Tan improvement Step 2

% NOTE: Following four lines are precomputed and passed as inputs
% pi_z_J_sim=gather(reshape(permute(pi_z_J(:,:,1:end-1),[3,1,2]),[(N_j-1)*N_z,N_z]));
% II1=repmat(1:1:(N_j-1)*N_z,1,N_z);
% II2=repmat(1:1:(N_j-1),1,N_z*N_z)+repelem((N_j-1)*(0:1:N_z-1),1,N_z*(N_j-1));
% pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);  

% Note, we just construct a block-diagonal, the blocks are the (z,z'). The diagonal is j. 
% Because we are going from an agent dist on j=1,...,N_j-1 to an agent dist
% on j=2,...,N_j the diagonal for j actually corresponds to deterministic ageing
AgentDist_tt=reshape(AgentDist_tt*pi_z_J_sim,[N_a*(N_j-1)*N_z,1]);

% exceptfirstj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
AgentDist(exceptfirstj)=gpuArray(full(AgentDist_tt)); % N_a*N_z+1 is avoiding those that correspond to jj=1

AgentDist(justfirstj)=jequalOneDist; % Age j=1 dist


end
