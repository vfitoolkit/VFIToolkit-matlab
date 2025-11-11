function StationaryDistKron=StationaryDist_FHorz_Iteration_SemiExo_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_a,N_semiz,N_j,pi_semiz_J,Parameters)
% Will treat the agents as being on a continuum of mass 1.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz together into the 1st dim.

% Policy_aprime is currently [N_a,N_semiz,N_j]
Policy_aprimesemiz=reshape(Policy_aprime,[N_a*N_semiz,1,N_j])+N_a*gpuArray(0:1:N_semiz-1); % Note: add semiz' index following the semiz' dimension
Policy_aprimesemiz=gather(Policy_aprimesemiz); % [N_a*N_semiz,N_semiz,N_j]

Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz,1,N_j]);
% precompute
semizindex=repelem(gpuArray(1:1:N_semiz)',N_a,1)+N_semiz*gpuArray(0:1:N_semiz-1)+(N_semiz*N_semiz)*(Policy_dsemiexo-1); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
% semizindex is [N_a*N_semiz,N_semiz,N_j]


%% No z, so no Tan improvement

StationaryDistKron=zeros(N_a*N_semiz,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;
StationaryDistKron_jj=sparse(gather(jequaloneDistKron));

II2=repelem((1:1:N_a*N_semiz)',1,N_semiz);

%%
for jj=1:(N_j-1)

    semiztransitions=gather(pi_semiz_J(semizindex(:,:,jj)));
    Gammatranspose=sparse(Policy_aprimesemiz(:,:,jj),II2,semiztransitions,N_a*N_semiz,N_a*N_semiz); % From (a,semiz) to (a',semiz')

    % No z, so just one-step for iteration
    StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;

    StationaryDistKron(:,jj+1)=gpuArray(full(StationaryDistKron_jj));
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

StationaryDistKron=StationaryDistKron.*AgeWeights;

end
