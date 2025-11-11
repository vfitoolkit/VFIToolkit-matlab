function StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these N_probs.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz&e together into the 1st dim, semiz'&nprobs into the 2nd dim.

% Policy_aprime is currently [N_a,N_semiz*N_e,N_probs,N_j]
Policy_aprimesemiz=repelem(reshape(Policy_aprime,[N_a*N_semiz*N_e,N_probs,N_j]),1,N_semiz)+repmat(N_a*gpuArray(0:1:N_semiz-1),1,N_probs); % Note: add semiz' index following the semiz' dimension
Policy_aprimesemiz=gather(Policy_aprimesemiz); % [N_a*N_semiz*N_e,N_semiz*N_probs,N_j]

Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1,N_j]);
% precompute
semizindex=repmat(repelem(gpuArray(1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*gpuArray(0:1:N_semiz-1)+(N_semiz*N_semiz)*(Policy_dsemiexo-1); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
% semizindex is [N_a*N_semiz*N_e,N_semiz,N_j]

PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_e,N_probs,N_j]);
PolicyProbs=repelem(PolicyProbs,1,N_semiz).*repmat(pi_semiz_J(semizindex),1,N_probs);
PolicyProbs=gather(PolicyProbs);

%% Use Tan improvement

StationaryDist=zeros(N_a*N_semiz*N_e,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(jequaloneDistKron); % sparse() creates a matrix of zeros

% Precompute
II2=repelem(gpuArray(1:1:N_a*N_semiz*N_e)',1,N_semiz*N_probs); % Index for this period (a,semiz), note the N_probs-copies

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprimesemiz(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_semiz,N_a*N_semiz*N_e); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

    % No z, so just a simple iteration
    StationaryDist_jj=Gammatranspose*StationaryDist_jj;
    
    % Put e back into dist
    pi_e=sparse(gather(pi_e_J(:,jj)));
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
