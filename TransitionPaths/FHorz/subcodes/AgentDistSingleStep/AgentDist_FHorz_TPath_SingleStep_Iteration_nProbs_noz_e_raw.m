function AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_noz_e_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_e,N_j,pi_e_J,II1,jequaloneDist)
% age weights are handled elsewhere, here all are normalized to one
% AgentDist=reshape(AgentDist,[N_a*N_e*N_probs,N_j]);
% Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,N_j,N_probs]));
% PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z*N_e,N_j,N_probs]));

% precomputed:
% II1=repelem((1:1:N_a*N_e)',1,N_probs);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(gather(AgentDist(:,jj)));

    Gammatranspose=sparse(Policy_aprime(:,jj,:),II1,PolicyProbs(:,jj,:),N_a,N_a*N_e);

    % Two steps of the Tan improvement
    AgentDist_jj=Gammatranspose*AgentDist_jj;

    pi_e=sparse(gather(pi_e_J(:,jj)));
    AgentDist_jj=kron(pi_e,AgentDist_jj);

    AgentDist(:,jj+1)=full(AgentDist_jj);
end
% Move result to gpu
AgentDist=gpuArray(AgentDist);
% Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.

AgentDist(:,1)=jequaloneDist; % age j=1 dist


end
