function StationaryDist=StationaryDist_FHorz_Iteration_TwoProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_j,Parameters)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

Policy_aprime=gather(Policy_aprime); %  (a,2,j)
PolicyProbs=gather(reshape(PolicyProbs,[N_a,2,N_j])); % (a,z,2,j)

%% Use Tan improvement
% Cannot reshape() with sparse gpuArrays. [And not obvious how to do Tan improvement without reshape()]
% Using full gpuArrays is marginally slower than just spare cpu arrays, so no point doing that.
% Hence, just force sparse cpu arrays.

StationaryDist=zeros(N_a,N_j);
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros

% Precompute
II2=[1:1:N_a;1:1:N_a]'; % Note the 2-copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprime(:,:,jj),II2,PolicyProbs(:,:,jj),N_a,N_a);  % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

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
