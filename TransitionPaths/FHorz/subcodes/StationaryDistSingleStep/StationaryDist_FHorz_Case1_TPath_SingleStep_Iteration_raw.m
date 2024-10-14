function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_z,N_j,pi_z_J)
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% AgentDist=reshape(AgentDist,[N_a*N_z,N_j]);

% Remove the existing age weights, then impose the new age weights at the end
AgentDist=AgentDist./AgeWeightsOld;

optaprime=gather(reshape(optaprime,[1,N_a*N_z,N_j]));

pi_z_J=gather(pi_z_J);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(gather(AgentDist(:,jj)));
    pi_z=sparse(pi_z_J(:,:,jj));

    optaprime_jj=optaprime(1,:,jj);

    % Tan improvement
    firststep=optaprime_jj+kron(N_a*(0:1:N_z-1),ones(1,N_a));
    Gammatranspose=sparse(firststep,1:1:N_a*N_z,ones(N_a*N_z,1),N_a*N_z,N_a*N_z);

     % Two steps of the Tan improvement
    AgentDist_jj=reshape(Gammatranspose*AgentDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    AgentDist_jj=reshape(AgentDist_jj*pi_z,[N_a*N_z,1]);

    AgentDist(:,jj+1)=full(AgentDist_jj);
end
% Move result to gpu
AgentDist=gpuArray(AgentDist);
% Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.

% Need to remove the old age weights, and impose the new ones
% Already removed the old age weights earlier, so now just impose the new ones.
% AgeWeights is a column vector
AgentDist=AgentDist.*AgeWeights;

end
