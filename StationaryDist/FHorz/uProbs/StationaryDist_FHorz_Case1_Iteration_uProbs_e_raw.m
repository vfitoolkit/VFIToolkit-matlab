function StationaryDist=StationaryDist_FHorz_Case1_Iteration_uProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_e,N_u,N_j,pi_z_J,pi_e_J,Parameters)
% 'uProbs' refers to n_u probabilities.
% Policy_aprime has an additional final dimensions of length n_u and 2 which is
% the n_u points and the lower/upper grid point (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these.

Policy_aprimez=Policy_aprime+repmat(N_a*gpuArray(0:1:N_z-1),1,N_e);
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z*N_e,N_u*2,N_j])); % (a,z,e,u,2,j)
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z*N_e,N_u*2,N_j])); % (a,z,e,u,2,j)

%% Use Tan improvement
% Cannot reshape() with sparse gpuArrays. [And not obvious how to do Tan improvement without reshape()]
% Using full gpuArrays is marginally slower than just spare cpu arrays, so no point doing that.
% Hence, just force sparse cpu arrays.

% Seems loop over e vs parallel over e is essentially break-even.
StationaryDist=zeros(N_a*N_z*N_e,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros

% Precompute
II2=repelem((1:1:N_a*N_z*N_e)',1,N_u*2);  % Index for this period (a,z,e), note the n_u*2 copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z*N_e); % Note: sparse() will accumulate at repeated indices

    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.

    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    pi_e=sparse(gather(pi_e_J(:,jj)));

    % Second step of Tan improvement
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

    StationaryDist_jj=kron(pi_e,StationaryDist_jj);

    StationaryDist(:,jj+1)=gpuArray(full(StationaryDist_jj));
end


% Reweight the different ages based on 'AgeWeightParamNames'. (it is assumed there is only one Age Weight Parameter (name))
try
    AgeWeights=Parameters.(AgeWeightParamNames{1});
catch
    error('Unable to find the AgeWeightParamNames in the parameter structure')
end
% I assume AgeWeights is a row vector
if size(AgeWeights,2)==1 % If it seems to be a column vector, then transpose it
    AgeWeights=AgeWeights';
end

StationaryDist=StationaryDist.*AgeWeights;

end
