function AgentDist=AgentDist_InfHorz_TPath_SingleStep_nProbs_noz_raw(AgentDist,Policy_aprime,II1,PolicyProbs,N_a)
% 'nProbs' refers to n probabilities.
% Policy_aprime has shape [N_a,N_probs] and contains only the (kron) aprime indexes, no d indexes.
% PolicyProbs are the corresponding probabilities of each of these N_probs.
% II1=repelem((1:1:N_a)',1,N_probs);

% AgentDist is already sparse and on cpu

Gammatranspose=sparse(gather(Policy_aprime),II1,gather(PolicyProbs),N_a,N_a); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

% No exogenous shocks, so the Tan improvement is just the one step
AgentDist=Gammatranspose*AgentDist;


end
