function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_z,N_j,pi_z_J,Parameters)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

epsilon=1e-7;
total_zeros_created=0;
jj_at_max_a2=Inf;

% Policy_aprime and PolicyProbs are currently [N_a,N_z,N_probs,N_j]
N_a1=prod(n_a1);
N_a2=prod(n_a2);
if N_a2==0
    N_a=N_a1;
elseif N_a1==0
    N_a=N_a2;
else
    N_a=N_a1*N_a2;
end
Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index following the z dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z,N_probs,N_j])); % sparse() requires inputs to be 2-D
needs_rounding=(PolicyProbs<epsilon | PolicyProbs>1-epsilon);
needs_rounding(PolicyProbs==0)=0;
needs_rounding(PolicyProbs==1)=0;
PolicyProbs(needs_rounding)=round(PolicyProbs(needs_rounding));
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,N_probs,N_j])); % sparse() requires inputs to be 2-D

%% Use Tan improvement

StationaryDist=zeros(N_a*N_z,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % use sparse matrix

% Precompute
II1=(1:1:N_a*N_z)';
II2=repmat((1:1:N_a*N_z)',1,N_probs); %  Index for this period (a,z), note the N_probs-copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
    Gammatranspose_lower=sparse(Policy_aprimez(:,1,jj),II1,PolicyProbs(:,1,jj),N_a*N_z,N_a*N_z);
    Gammatranspose_upper=sparse(Policy_aprimez(:,2,jj),II1,PolicyProbs(:,2,jj),N_a*N_z,N_a*N_z);

    % First step of Tan improvement
    needs_rounding=full(StationaryDist_jj<epsilon | StationaryDist_jj>1-epsilon);
    needs_rounding(StationaryDist_jj==0)=0;
    needs_rounding(StationaryDist_jj==1)=0;
    StationaryDist_jj(needs_rounding)=round(StationaryDist_jj(needs_rounding));
    StationaryDist_lower_jj=reshape(Gammatranspose_lower*StationaryDist_jj,[N_a,N_z]);
    StationaryDist_upper_jj=reshape(Gammatranspose_upper*StationaryDist_jj,[N_a,N_z]);
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    StationaryDist_jj=StationaryDist_jj*pi_z;
    StationaryDist_lower_jj=StationaryDist_lower_jj*pi_z;
    StationaryDist_upper_jj=StationaryDist_upper_jj*pi_z;

    [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj,StationaryDist_lower_jj,StationaryDist_upper_jj, N_a1,N_a2,N_z,jj, epsilon,total_zeros_created,jj_at_max_a2);

    StationaryDist_jj=reshape(StationaryDist_jj,[N_a*N_z,1]);
    assert(all(StationaryDist_jj>=0));
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

fprintf("With epsilon = %.2e, total zeros created = %d \n", epsilon, total_zeros_created);
if isfinite(jj_at_max_a2)
    fprintf("Max ExpAsset value first observed at age %3d \n", jj_at_max_a2);
else
    temp=reshape(StationaryDist,[N_a1,N_a2,N_z,N_j]);
    [a1,a2,z_c,age_j]=ind2sub(size(temp),find(temp~=0));
    max_a2=max(a2);
    jj_at_max_a2=min(age_j(find(a2==max_a2)));
    fprintf("Max ExpAsset index reached = %3d (of %3d) at age %3d \n", max_a2, N_a2, jj_at_max_a2);
end

end
