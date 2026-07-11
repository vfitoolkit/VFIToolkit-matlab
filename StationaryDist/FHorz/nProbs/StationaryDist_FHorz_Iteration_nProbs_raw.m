function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,N_a,N_z,N_j,pi_z_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

% Policy_aprime and PolicyProbs are currently [N_a,N_z,N_probs,N_j]
Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index following the z dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z,N_probs,N_j])); % sparse() requires inputs to be 2-D
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,N_probs,N_j])); % sparse() requires inputs to be 2-D

%% Use Tan improvement

StationaryDist=zeros(N_a*N_z,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % use sparse matrix
epsilon_z=1e-3*full(sum(reshape(StationaryDist_jj,[N_a,N_z]),1)); % A suitably small value, scaled by the probability sum of this z slice

% Precompute
II2=repmat((1:1:N_a*N_z)',1,N_probs); %  Index for this period (a,z), note the N_probs-copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices

    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]);

    % Clean up Gaussian diffusion from Gamma step
    nnz_gamma=sum(StationaryDist_jj~=0,1);
    for z_c=1:length(nnz_gamma)
        if nnz_gamma(z_c)>6
            [epsilons, e_idx] = mink(nonzeros(StationaryDist_jj(:,z_c)), nnz_gamma(z_c)-6);
            e_idx=e_idx(epsilons<epsilon_z(z_c));
            epsilons=epsilons(epsilons<epsilon_z(z_c));
            if nnz(epsilons)>0
                nonzero_idx=find(StationaryDist_jj(:,z_c));
                % zero out likely error artifacts
                StationaryDist_jj(nonzero_idx(e_idx),z_c)=0;
                keep_nonzero=true(size(nonzero_idx));
                keep_nonzero(e_idx)=false;
                % redistribute values zeroed out equally among remaining nonzero terms
                % QUESTION: should we redistribute pro-rata instead of equally?
                StationaryDist_jj(nonzero_idx(keep_nonzero),z_c)=StationaryDist_jj(nonzero_idx(keep_nonzero),z_c)+sum(epsilons)/(nnz_gamma(z_c)-length(epsilons));
            end
        end
    end

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
