function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,PolicyIndexesKron,N_d,N_a,N_j)
% Parallelizes over age jj
% AgentDist is [N_a*N_j,1]
% AgeWeights is [N_a*N_j,1] (obviously just repeats same numbers over the N_a)

if N_d==0
    optaprime=gather(reshape(PolicyIndexesKron(:,1:end-1),[1,N_a*(N_j-1)]));
else
    optaprime=gather(reshape(PolicyIndexesKron(2,:,1:end-1),[1,N_a*(N_j-1)]));
end

% Remove the existing age weights, then impose the new age weights at the end
AgentDist=AgentDist./AgeWeightsOld;

AgentDist_tt=sparse(gather(reshape(AgentDist(1:end-N_a),[N_a*(N_j-1),1]))); % end-N_a is avoiding those that correspond to jj=N_j

Gammatranspose=sparse(optaprime+N_a*repelem((0:1:N_j-2),1,N_a),1:1:N_a*(N_j-1),ones(N_a*(N_j-1),1),N_a*(N_j-1),N_a*(N_j-1));
% Note: N_j-1, not N_j

AgentDist_tt=Gammatranspose*AgentDist_tt;

AgentDist(N_a+1:end)=gpuArray(full(AgentDist_tt)); % N_a+1 is avoiding those that correspond to jj=1

% Need to remove the old age weights, and impose the new ones
% Already removed the old age weights earlier, so now just impose the new ones.
% AgeWeights is a row vector
AgentDist=AgentDist.*AgeWeights;

end
