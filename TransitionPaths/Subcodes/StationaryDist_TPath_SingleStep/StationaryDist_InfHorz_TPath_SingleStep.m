function AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,Policy_aprimez,II1,IIones,N_a,N_z,pi_z_sparse)
% II1 and IIones are inputs just so they can be precomputed and then reused each period of transition
% II1=gpuArray(1:1:N_a*N_z);
% IIones=ones(N_a*N_z,1,'gpuArray');

Gammatranspose=sparse(Policy_aprimez,II1,IIones,N_a*N_z,N_a*N_z);

% Two steps of the Tan improvement
AgentDist=reshape(Gammatranspose*AgentDist,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_z,1]);

end