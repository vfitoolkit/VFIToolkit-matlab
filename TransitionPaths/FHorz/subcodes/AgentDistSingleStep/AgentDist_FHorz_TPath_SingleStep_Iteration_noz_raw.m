function AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_raw(AgentDist,Policy_aprime,N_a,N_j,jequalOneDist)
% age weights are handled elsewhere, here all are normalized to one
% AgentDist is [N_a,N_j]
% Policy_aprime=gather(reshape(Policy_aprime,[1,N_a,N_j]));

AgentDist=gather(AgentDist);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(AgentDist(:,jj));

    Gammatranspose=sparse(Policy_aprime(1,:,jj),1:1:N_a,ones(N_a,1),N_a,N_a);

    AgentDist_jj=Gammatranspose*AgentDist_jj;

    AgentDist(:,jj+1)=full(AgentDist_jj);
end

% Move the solution to the gpu
AgentDist=gpuArray(AgentDist);
AgentDist(:,1)=jequalOneDist; % age j=1 dist

end
