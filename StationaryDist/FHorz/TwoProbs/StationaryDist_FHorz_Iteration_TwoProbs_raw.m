function StationaryDist=StationaryDist_FHorz_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z_J,Parameters)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1); % Note: add z index following the z dimension
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z,2,N_j])); % (a,z,2,j)
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,2,N_j])); % (a,z,2,j)

%% Use Tan improvement
% Cannot reshape() with sparse gpuArrays. [And not obvious how to do Tan improvement without reshape()]
% Using full gpuArrays is marginally slower than just spare cpu arrays, so no point doing that.
% Hence, just force sparse cpu arrays.

StationaryDist=zeros(N_a*N_z,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % use sparse cpu matrix

% Precompute
II2=[1:1:N_a*N_z; 1:1:N_a*N_z]'; % Index for this period (a,z), note the 2 copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

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
