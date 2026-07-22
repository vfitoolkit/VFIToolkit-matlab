function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_j,Parameters,simoptions)
% 'nProbs' refers to four probabilities.
% Policy_aprime has an additional dimension of length 4 which is the four points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these four

if exist('simpoptions','var')==0
    simoptions=struct();
    simoptions.optimize_nProbs=0;
elseif ~isfield(simoptions,'optimize_nProbs')
    simoptions.optimize_nProbs=0;
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
II1=(1:1:N_a)';
II2=repmat((1:1:N_a)',1,N_probs); % Note the N_probs-copies

for jj=1:(N_j-1)
    % First, get Gamma
    Gammatranspose=sparse(Policy_aprime(:,:,jj),II2,PolicyProbs(:,:,jj),N_a,N_a);  % Note: sparse() will accumulate at repeated indices
    Gammatranspose_lower=sparse(Policy_aprime(:,1,jj),II1,PolicyProbs(:,1,jj),N_a,N_a);
    Gammatranspose_upper=sparse(Policy_aprime(:,2,jj),II1,PolicyProbs(:,2,jj),N_a,N_a);

    % No z, so just a single step
    StationaryDist_lower_jj=Gammatranspose_lower*StationaryDist_jj;
    StationaryDist_upper_jj=Gammatranspose_upper*StationaryDist_jj;
    StationaryDist_jj=Gammatranspose*StationaryDist_jj; % =StationaryDist_lower_jj+StationaryDist_upper_jj;

    if simoptions.optimize_nProbs==1
        [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj,StationaryDist_lower_jj,StationaryDist_upper_jj, N_a1,N_a2,0,0,jj, epsilon,total_zeros_created,jj_at_max_a2);
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

if total_zeros_created>0
    fprintf("With epsilon = %.2e, total zeros created = %d \n", epsilon, total_zeros_created);
    if isfinite(jj_at_max_a2)
        fprintf("Max ExpAsset value first observed at age %3d \n", jj_at_max_a2);
    else
        temp=reshape(StationaryDist,[N_a1,N_a2,N_j]);
        [a1,a2,age_j]=ind2sub(size(temp),find(temp~=0));
        max_a2=max(a2);
        jj_at_max_a2=min(age_j(find(a2==max_a2)));
        fprintf("Max ExpAsset index reached = %3d (of %3d) at age %3d \n", max_a2, N_a2, jj_at_max_a2);
    end
end

end
