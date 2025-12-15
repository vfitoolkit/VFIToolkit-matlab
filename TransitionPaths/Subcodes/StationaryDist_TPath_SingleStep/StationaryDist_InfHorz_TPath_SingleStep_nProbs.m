function AgentDist=StationaryDist_InfHorz_TPath_SingleStep_nProbs(AgentDist,Policy_aprimez,II2,PolicyProbs,N_a,N_z,pi_z_sparse)
% 'nProbs' refers to n probabilities.
% Policy_aprimez has shape [N_a*N_z,N_probs] and contains only the (kron) aprime indexes, no d indexes. 
% PolicyProbs are the corresponding probabilities of each of these N_probs.

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimez,II2,PolicyProbs,N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

% Two steps of the Tan improvement
AgentDist=reshape(Gammatranspose*AgentDist,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_z,1]);

end