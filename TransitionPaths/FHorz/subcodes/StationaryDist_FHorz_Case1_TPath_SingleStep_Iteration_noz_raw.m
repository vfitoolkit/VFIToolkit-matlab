function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,PolicyIndexesKron,N_d,N_a,N_j)
% AgentDist is [N_a,N_j]
% AgeWeights is [1,N_j]

if N_d==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a,N_j]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a,N_j]);
end

optaprime=gather(optaprime);

% Remove the existing age weights, then impose the new age weights at the end
AgentDist=AgentDist./AgeWeightsOld;

AgentDist=gather(AgentDist);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(AgentDist(:,jj));

    Gammatranspose=sparse(optaprime(1,:,jj),1:1:N_a,ones(N_a,1),N_a,N_a);

    AgentDist_jj=Gammatranspose*AgentDist_jj;

    AgentDist(:,jj+1)=full(AgentDist_jj);
end

% Move the solution to the gpu
AgentDist=gpuArray(AgentDist);

% Need to remove the old age weights, and impose the new ones
% Already removed the old age weights earlier, so now just impose the new ones.
% AgeWeights is a row vector
AgentDist=AgentDist.*AgeWeights;

end
