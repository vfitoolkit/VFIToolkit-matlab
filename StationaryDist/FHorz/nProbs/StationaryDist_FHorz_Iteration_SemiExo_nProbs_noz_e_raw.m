function StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_nProbs_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_probs,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz&e together into the 1st dim, semiz'&nprobs into the 2nd dim.

% It is likely that most of the elements in pi_semiz_J are zero, we can
% take advantage of this to speed things up. Ignore for a moment the
% dependence on d and j, and pretend it is just a N_semiz-by-N_semiz
% matrix. Then we can calculate N_semizshort=max(sum((pi_semiz>0),2)), the
% maximum number of non-zeros in any row of pi_semiz. And we then use this
% in place of N_semiz as the second dimension.

N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
% Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
idxshort=idx(:,end-N_semizshort+1:end,:,:);

Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1,N_j]);
semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1))+(N_semiz*N_semizshort*N_dsemiz)*shiftdim((0:1:N_j-1),-1); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
pi_semiz_J_short=gather(pi_semiz_J_short);
% semizindex_short is [N_a*N_semiz*N_e,N_semizshort,N_j]
% used to index pi_semiz_J_short which is [N_semiz,N_semizshort,N_dsemiz,N_j]
% and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz,N_j]

% Policy_aprime is currently [N_a,N_semiz*N_e,N_probs,N_j]
Policy_aprimesemiz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_e,N_probs,N_j]),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_probs); % Note: add semiz' index following the semiz' dimension, add z' index following the z dimension for Tan improvement
% Policy_aprimesemizz is currently [N_a,N_semiz*N_z,N_probs*N_semizshort,N_j]

PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_e,N_probs,N_j]);
PolicyProbs=repelem(gather(PolicyProbs),1,N_semizshort,1).*repmat(pi_semiz_J_short(semizindex_short),1,N_probs,1); % is of size [N_a*N_semiz*N_e,N_probs*N_semiz,N_j]
% WHY DONT I CREATE PolicyProbs ON GPU, THEN gather()? UNLESS IT RUNS OUT OF GPU MEMORY THIS SHOULD BE FASTER?

%% Use Tan improvement

StationaryDist=zeros(N_a*N_semiz*N_e,N_j,'gpuArray'); % StationaryDist cannot be sparse
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(jequaloneDistKron); % use sparse matrix

% Precompute; II2 used only for sparse matrix creation...best done on CPU
II2=repelem((1:1:N_a*N_semiz*N_e)',1,N_semizshort*N_probs); % Index for this period (a,semiz), note the N_semizshort*N_probs-copies

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprimesemiz(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_semiz,N_a*N_semiz*N_e); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

    % No z, so just a simple iteration
    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

    % Put e back into dist
    StationaryDist_jj=kron(pi_e_J(:,jj),StationaryDist_jj);

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
