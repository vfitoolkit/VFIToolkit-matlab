function AgentDist=AgentDist_InfHorz_TPath_SingleStep_noz_e_raw(AgentDist,Policy_aprime,II1,II2,N_a,N_e,pi_e_sparse)
% II1 and II2 are inputs just so they can be precomputed and then reused each period of transition
% II1=1:1:N_a*N_e;
% II2=ones(N_a*N_e,1);
% AgentDist is [N_a*N_e,1]
% Policy_aprime is [N_a*N_e,1], the aprime index
% pi_e_sparse is [N_e,1], the distribution of next period's e

% AgentDist is already sparse and on cpu

% Next period's a does not depend on next period's e, so the first step maps (a,e) into aprime only
Gammatranspose=sparse(gather(Policy_aprime),II1,II2,N_a,N_a*N_e);

% Two steps of the Tan improvement
AgentDist=Gammatranspose*AgentDist; % [N_a,1]
% full(): kron with the sparse pi_e returns a sparse matrix, and the AggVars multiply the dist into a gpuArray, which
% then gives a sparse gpuArray that isnan() cannot take. (Without e the dist stays full: Gammatranspose times a full
% vector is full.)
AgentDist=full(kron(pi_e_sparse,AgentDist)); % [N_a*N_e,1], next period's e is independent of everything

end
