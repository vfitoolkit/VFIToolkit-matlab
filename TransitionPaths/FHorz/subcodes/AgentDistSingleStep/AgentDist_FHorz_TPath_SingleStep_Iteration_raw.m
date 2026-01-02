function AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_raw(AgentDist,Policy_aprime,N_a,N_z,N_j,pi_z_J,jequalOneDist)
% age weights are handled elsewhere, here all are normalized to one

% AgentDist=reshape(AgentDist,[N_a*N_z,N_j]);
% Policy_aprime=gather(reshape(Policy_aprime,[1,N_a*N_z,N_j]));

pi_z_J=gather(pi_z_J);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(gather(AgentDist(:,jj)));
    pi_z=sparse(pi_z_J(:,:,jj));

    optaprime_jj=Policy_aprime(1,:,jj);

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
AgentDist(:,1)=jequalOneDist;


end
