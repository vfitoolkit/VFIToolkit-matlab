function StationaryDist=StationaryDist_FHorz_Iteration_nProbs_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_z,N_e,N_j,pi_z_J,pi_e_J,Parameters,simoptions)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

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
% Policy_aprime and PolicyProbs are currently [N_a,N_z*N_e,N_probs,N_j]
Policy_aprimez=Policy_aprime+repmat(N_a*(0:1:N_z-1),1,N_e);  % Note: add z' index following the z dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z*N_e,N_probs,N_j])); % sparse() requires inputs to be 2-D
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z*N_e,N_probs,N_j])); % sparse() requires inputs to be 2-D

%% Use Tan improvement

StationaryDist=zeros(N_a*N_z*N_e,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros

% Precompute
II2=repmat((1:1:N_a*N_z*N_e)',1,N_probs); %  Index for this period (a,z), note the N_probs-copies

for jj=1:(N_j-1)

    % First, get Gamma
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),II2,PolicyProbs(:,:,jj),N_a*N_z,N_a*N_z*N_e); % Note: sparse() will accumulate at repeated indices

    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

    if simoptions.optimize_nProbs==1
        [StationaryDist_jj,total_zeros_created,jj_at_max_a2]=StationaryDist_FHorz_Optimize_nProbs_raw(StationaryDist_jj, N_a1,N_a2,N_z,N_e,jj, epsilon,total_zeros_created,jj_at_max_a2,simoptions);
    end

    % Put e back into dist
    pi_e=sparse(gather(pi_e_J(:,jj)));
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
                temp=reshape(StationaryDist,[N_a1,N_z*N_e,N_j]);
                [a1,~,age_j]=ind2sub(size(temp),find(temp~=0));
                max_a=max(a1);
                jj_at_max_a2=min(age_j(a1==max_a));
            else
                temp=reshape(StationaryDist,[N_a1,N_a2,N_z*N_e,N_j]);
                [~,a2,~,age_j]=ind2sub(size(temp),find(temp~=0));
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
