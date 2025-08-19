function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_e,N_j,pi_e_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these N_probs.

Policy_aprime=reshape(Policy_aprime,[N_a*N_e,N_probs,N_j]);
PolicyProbs=reshape(PolicyProbs,[N_a*N_e,N_probs,N_j]);

%% Use Tan improvement

StationaryDist=zeros(N_a*N_e,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(jequaloneDistKron); % sparse() creates a matrix of zeros

% Precompute
II2=repmat(gpuArray((1:1:N_a*N_z)'),1,N_probs); %  Index for this period (a,e), note the N_probs-copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprime(:,:,jj),II2,PolicyProbs(:,:,jj),N_a,N_a*N_e); % Note: sparse() will accumulate at repeated indices

    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

    pi_e=sparse(pi_e_J(:,jj);

    StationaryDist_jj=kron(pi_e,StationaryDist_jj);

    StationaryDist(:,jj+1)=full(StationaryDist_jj);
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
