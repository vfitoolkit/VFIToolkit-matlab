function AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_e_raw(AgentDist,Policy_aprimez,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,II1,II2,jequaloneDist)
% age weights are handled elsewhere, here all are normalized to one

% AgentDist=reshape(AgentDist,[N_a*N_z*N_e,N_j]);
% Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z*N_e,N_j]));

% precompute:
% II1=1:1:N_a*N_z*N_e;
% II2=ones(N_a*N_z*N_e,1);

for jjr=1:(N_j-1)
    jj=N_j-jjr; % It is important that this is in reverse order (due to just overwriting AgentDist)
    AgentDist_jj=sparse(gather(AgentDist(:,jj)));

    Gammatranspose=sparse(Policy_aprimez(:,jj),II1,II2,N_a*N_z,N_a*N_z*N_e);

    pi_z=sparse(gather(pi_z_J(:,:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).
    pi_e=sparse(gather(pi_e_J(:,jj)));

    % Two steps of the Tan improvement
    AgentDist_jj=reshape(Gammatranspose*AgentDist_jj,[N_a,N_z]);
    AgentDist_jj=reshape(AgentDist_jj*pi_z,[N_a*N_z,1]);

    AgentDist_jj=kron(pi_e,AgentDist_jj);

    AgentDist(:,jj+1)=full(AgentDist_jj);
end
% Move result to gpu
AgentDist=gpuArray(AgentDist);
% Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.

AgentDist(:,1)=jequaloneDist; % age j=1 dist

end
