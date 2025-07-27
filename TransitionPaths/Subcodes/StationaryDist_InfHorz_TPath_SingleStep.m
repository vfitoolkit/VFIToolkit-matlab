function AgentDist=StationaryDist_InfHorz_TPath_SingleStep(AgentDist,Policy,N_d,N_a,N_z,pi_z_sparse)

% Setup for Tan improvement
if N_d==0
    optaprime=gather(reshape(Policy,[1,N_a*N_z]));
else
    optaprime=gather(reshape(shiftdim(Policy(2,:,:),-1),[1,N_a*N_z])); % This shifting of dimensions is probably not necessary
end

firststep=optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a));
Gammatranspose=sparse(firststep,1:1:N_a*N_z,ones(N_a*N_z,1),N_a*N_z,N_a*N_z);

% Two steps of the Tan improvement
AgentDist=reshape(Gammatranspose*AgentDist,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
AgentDist=reshape(AgentDist*pi_z_sparse,[N_a*N_z,1]);




end