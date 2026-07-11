function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_j,Parameters)
% 'nProbs' refers to four probabilities.
% Policy_aprime has an additional dimension of length 4 which is the four points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these four

% Policy_aprime and PolicyProbs are currently [N_a,N_probs,N_j]
Policy_aprime=gather(Policy_aprime);
PolicyProbs=gather(PolicyProbs);

%% Use Tan improvement

StationaryDist=zeros(N_a,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros
epsilon=5e-4; % A suitably small value

% Precompute
II2=repmat((1:1:N_a)',1,N_probs); % Note the N_probs-copies

for jj=1:(N_j-1)
    % First, get Gamma
    Gammatranspose=sparse(Policy_aprime(:,:,jj),II2,PolicyProbs(:,:,jj),N_a,N_a);  % Note: sparse() will accumulate at repeated indices

    % No z, so just a single step
    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

    % Clean up Gaussian diffusion from Gamma step
    nnz_gamma=nnz(StationaryDist_jj);
    if nnz_gamma>6
        [epsilons, e_idx] = mink(nonzeros(StationaryDist_jj), nnz_gamma-6);
        e_idx=e_idx(epsilons<epsilon);
        epsilons=epsilons(epsilons<epsilon);
        if nnz(epsilons)>0
            nonzero_idx=find(StationaryDist_jj);
            % zero out likely error artifacts
            StationaryDist_jj(nonzero_idx(e_idx))=0;
            keep_nonzero=true(size(nonzero_idx));
            keep_nonzero(e_idx)=false;
            % redistribute values zeroed out equally among remaining nonzero terms
            % QUESTION: should we redistribute pro-rata instead of equally?
            StationaryDist_jj(nonzero_idx(keep_nonzero))=StationaryDist_jj(nonzero_idx(keep_nonzero))+sum(epsilons)/(nnz_gamma-length(epsilons));
        end
    end
    StationaryDist(:,jj+1)=gather(full(StationaryDist_jj));
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
