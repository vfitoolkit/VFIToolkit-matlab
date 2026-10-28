function AgentDist=AgentDist_InfHorz_TPath_SingleStep_noz_raw(AgentDist,Policy_aprime,II1,II2,N_a)
% II1 and II2 are inputs just so they can be precomputed and then reused each period of transition
% II1=1:1:N_a;
% II2=ones(N_a,1);
% AgentDist is [N_a,1]
% Policy_aprime is [N_a,1], the aprime index

% AgentDist is already sparse and on cpu

Gammatranspose=sparse(gather(Policy_aprime),II1,II2,N_a,N_a);

% No exogenous shocks, so the Tan improvement is just the one step
AgentDist=Gammatranspose*AgentDist;

end
