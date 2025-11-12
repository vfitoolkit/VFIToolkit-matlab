function StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprimez has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these N_probs.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz&z together into the 1st dim, semiz'&nprobs into the 2nd dim.

% Policy_aprime is currently [N_a,N_semiz*N_z,N_probs,N_j]
Policy_aprimesemizz=repelem(reshape(Policy_aprime,[N_a*N_semiz*N_z,N_probs,N_j]),1,N_semiz)+repmat(N_a*gpuArray(0:1:N_semiz-1),1,N_probs)+repelem(N_a*N_semiz*gpuArray(0:1:N_z-1)',N_a*N_semiz,1); % Note: add semiz' index following the semiz' dimension, add z' index following the z dimension for Tan improvement
Policy_aprimesemizz=gather(Policy_aprimesemizz); % [N_a*N_semiz*N_z,N_semiz*N_probs,N_j]
% % Previous two lines gave out of memory order, so the following line just does gather() earlier.
% Policy_aprimesemizz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_z,N_probs,N_j]),1,N_semiz)+repmat(N_a*(0:1:N_semiz-1),1,N_probs)+repelem(N_a*N_semiz*(0:1:N_z-1)',N_a*N_semiz,1); % Note: add semiz' index following the semiz' dimension, add z' index following the z dimension for Tan improvement

Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_z,1,N_j]);
% precompute
semizindex=repmat(repelem(gpuArray(1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*gpuArray(0:1:N_semiz-1)+(N_semiz*N_semiz)*(Policy_dsemiexo-1); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
% % Previous line gave out of memory order, so the following line just does gather() earlier.
% semizindex=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semiz-1)+gather((N_semiz*N_semiz)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
% semizindex is [N_a*N_semiz*N_z,N_semiz,N_j]


PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_z,N_probs,N_j]);
PolicyProbs=repelem(PolicyProbs,1,N_semiz).*repmat(pi_semiz_J(semizindex),1,N_probs);
PolicyProbs=gather(PolicyProbs);
% % Previous two lines gave out of memory order, so the following line just does gather() earlier.
% pi_semiz_J=gather(pi_semiz_J);
% PolicyProbs=repelem(gather(PolicyProbs),1,N_semiz).*repmat(pi_semiz_J(semizindex),1,N_probs);

N_bothz=N_semiz*N_z;

%% Use Tan improvement

StationaryDist=zeros(N_a*N_semiz*N_z,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(jequaloneDistKron); % use sparse matrix

% Precompute
II2=repelem(gpuArray(1:1:N_a*N_semiz*N_z)',1,N_semiz*N_probs); % Index for this period (a,semiz), note the N_probs-copies

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprimesemizz(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_bothz,N_a*N_bothz); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a*N_semiz,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_bothz,1]);

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
