function AgentDist=StationaryDist_InfHorz_TPath_SingleStep_TwoProbs(AgentDist,Policy_aprimez,II2,PolicyProbs,N_a,N_z,pi_z_sparse)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimez,II2,PolicyProbs,N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

% Two steps of the Tan improvement
AgentDist=reshape(Gammatranspose*AgentDist,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_z,1]);

end