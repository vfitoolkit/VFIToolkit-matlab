function AgentDist=AgentDist_InfHorz_TPath_SingleStep_e_raw(AgentDist,Policy_aprimez,II1,II2,N_a,N_z,N_e,pi_z_sparse,pi_e_sparse)
% II1 and II2 are inputs just so they can be precomputed and then reused each period of transition
% II1=1:1:N_a*N_z*N_e;
% II2=ones(N_a*N_z*N_e,1);
% AgentDist is [N_a*N_z*N_e,1]
% Policy_aprimez is [N_a*N_z*N_e,1], the (aprime,z) index aprime+N_a*(z-1)
% pi_z_sparse is [N_z,N_zprime], pi_e_sparse is [N_e,1] the distribution of next period's e

% AgentDist is already sparse and on cpu

% Next period's a does not depend on next period's e, so the first step maps (a,z,e) into (aprime,z)
Gammatranspose=sparse(gather(Policy_aprimez),II1,II2,N_a*N_z,N_a*N_z*N_e);

% Three steps of the Tan improvement
AgentDist=reshape(Gammatranspose*AgentDist,[N_a,N_z]);
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_z,1]);
% full(): kron with the sparse pi_e returns a sparse matrix, and the AggVars multiply the dist into a gpuArray, which
% then gives a sparse gpuArray that isnan() cannot take. (Without e the dist stays full: Gammatranspose times a full
% vector is full.)
AgentDist=full(kron(pi_e_sparse,AgentDist)); % [N_a*N_z*N_e,1], next period's e is independent of everything

end
