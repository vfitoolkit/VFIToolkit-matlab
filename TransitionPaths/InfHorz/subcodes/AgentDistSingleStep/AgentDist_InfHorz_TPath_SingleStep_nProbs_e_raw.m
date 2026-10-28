function AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_e_raw(AgentDist,Policy_aprimez,II1,PolicyProbs,N_a,N_z,N_e,pi_z_sparse,pi_e_sparse)
% 'nProbs' refers to n probabilities.
% Policy_aprimez has shape [N_a*N_z*N_e,N_probs] and contains only the (kron) (aprime,z) indexes, no d indexes.
% PolicyProbs are the corresponding probabilities of each of these N_probs.
% II1=repelem((1:1:N_a*N_z*N_e)',1,N_probs);
% pi_z_sparse is [N_z,N_zprime], pi_e_sparse is [N_e,1] the distribution of next period's e

% AgentDist is already sparse and on cpu

Gammatranspose=sparse(gather(Policy_aprimez),II1,gather(PolicyProbs),N_a*N_z,N_a*N_z*N_e); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

% Three steps of the Tan improvement
AgentDist=reshape(Gammatranspose*AgentDist,[N_a,N_z]);
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_z,1]);
% full(): kron with the sparse pi_e returns a sparse matrix, and the AggVars multiply the dist into a gpuArray, which
% then gives a sparse gpuArray that isnan() cannot take. (Without e the dist stays full: Gammatranspose times a full
% vector is full.)
AgentDist=full(kron(pi_e_sparse,AgentDist)); % [N_a*N_z*N_e,1]


end
