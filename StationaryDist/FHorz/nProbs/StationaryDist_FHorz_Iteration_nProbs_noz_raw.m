function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_j,Parameters,simoptions)
% 'nProbs' refers to four probabilities.
% Policy_aprime has an additional dimension of length 4 which is the four points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these four

if exist('simoptions','var')==0
    simoptions=struct();
    simoptions.optimize_nProbs=0;
    simoptions.verbosed=0;
else
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'optimize_nProbs')
        simoptions.optimize_nProbs=0;
    end
end

epsilon=1e-7;
total_zeros_created=0;
jj_at_max_a2=Inf;

% Policy_aprime and PolicyProbs are currently [N_a,N_probs,N_j]
N_a1=prod(n_a1);
N_a2=prod(n_a2);
if N_a2==0
    N_a=N_a1;
elseif N_a1==0
    N_a1=1;
    N_a=N_a2;
else
    N_a=N_a1*N_a2;
end
Policy_aprime=gather(Policy_aprime);
needs_rounding=(PolicyProbs<epsilon | PolicyProbs>1-epsilon);
needs_rounding(PolicyProbs==0)=0;
needs_rounding(PolicyProbs==1)=0;
PolicyProbs(needs_rounding)=round(PolicyProbs(needs_rounding));
PolicyProbs=gather(PolicyProbs);

%% Use Tan improvement

StationaryDist=zeros(N_a,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros

% Precompute
II2=repmat((1:1:N_a)',1,N_probs); % Note the N_probs-copies

for jj=1:(N_j-1)
    % First, get Gamma
    Gammatranspose=sparse(Policy_aprime(:,:,jj),II2,PolicyProbs(:,:,jj),N_a,N_a);  % Note: sparse() will accumulate at repeated indices

    % No z, so just a single step
    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

    if simoptions.optimize_nProbs==1
        [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj, N_a1,N_a2,0,jj, epsilon,total_zeros_created,jj_at_max_a2,simoptions);
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

if isfinite(jj_at_max_a2)
    if N_a2>0
        warning("Max ExpAsset index %3d first reached at age %3d \n", N_a2, jj_at_max_a2);
    else
        warning("Max grid-interpolated asset index %3d first reached at age %3d \n", N_a1, jj_at_max_a2);
    end
end

if simoptions.verbose
    if total_zeros_created>0
        fprintf("With epsilon = %.2e, total zeros created = %d \n", epsilon, total_zeros_created);
        if ~isfinite(jj_at_max_a2)
            max_a=nan;
            if N_a2==0
                temp=reshape(StationaryDist,[N_a1,N_j]);
                [a1,age_j]=ind2sub(size(temp),find(temp~=0));
                max_a=max(a1);
                jj_at_max_a2=min(age_j(a1==max_a));
            else
                temp=reshape(StationaryDist,[N_a1,N_a2,N_j]);
                [~,a2,age_j]=ind2sub(size(temp),find(temp~=0));
                max_a=max(a2);
                jj_at_max_a2=min(age_j(a2==max_a));
            end
            if N_a2>0
                fprintf("Max ExpAsset index reached: %3d (of %3d) at age %3d \n", max_a, N_a2, jj_at_max_a2);
            else
                fprintf("Max grid-interpolated asset index reached: %3d (of %3d) at age %3d \n", max_a, N_a1, jj_at_max_a2);
            end
        end
    end
end

end
