function StationaryDistKron=StationaryDist_FHorz_Iteration_SemiExo_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,PolicyProbs,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

N_bothz=N_semiz*N_z;

Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_bothz,N_j])); % (a,z,j)
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_bothz,2,N_j])); % (a,z,2,j)
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_bothz,2,N_j])); % (a,z,2,j)

% precompute
semizindexcorrespondingtod2_c=repelem(repmat((1:1:N_semiz)',N_z,1),N_a,1);

%% Use Tan improvement
% Cannot reshape() with sparse gpuArrays. [And not obvious how to do Tan improvement without reshape()]
% Using full gpuArrays is marginally slower than just spare cpu arrays, so no point doing that.
% Hence, just force sparse cpu arrays.

StationaryDistKron=zeros(N_a*N_bothz,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;
StationaryDistKron_jj=sparse(gather(jequaloneDistKron)); % sparse() creates a matrix of zeros

% Precompute
II2=repelem((1:1:N_a*N_bothz),2*N_semiz,1); % Index for this period (a,semiz,z), note the 2 copies
% Note: repelem((1:1:N_a*N_bothz),2*N_semiz,1) is just a simpler way to write repelem((1:1:N_a*N_bothz)',1,2*N_semiz)'

for jj=1:(N_j-1)

    firststep=repmat(Policy_aprime(:,:,jj),1,N_semiz)+kron(N_a*N_semiz*(0:1:N_z-1)',ones(N_a*N_semiz,1))+N_a*repelem(0:1:N_semiz-1,1,2); % (a',semiz',z')-by-(2,semiz)
    % Note: optaprime and the z are columns, while semiz is a row that adds every semiz
    
    % Get the semiz transition probabilities into needed form
    pi_semiz=pi_semiz_J(:,:,:,jj);
    % Get the right part of pi_semiz_J
    % d2 depends on (a,semiz,z), and pi_semiz is going to be about (semiz,semiz'), so I need to put it all together as (a,semiz,z,semiz').
    % semizindexcorrespondingtod2_c=repelem(repmat((1:1:N_semiz)',N_z,1),N_a,1); % precomputed
    fullindex=semizindexcorrespondingtod2_c+N_semiz*(0:1:N_semiz-1)+(N_semiz*N_semiz)*(Policy_dsemiexo(:,jj)-1);
    semiztransitions=pi_semiz(fullindex); % (a,z,semiz -by- semiz')

    % First, get Gamma
    Gammatranspose=sparse(firststep',II2,(repmat(PolicyProbs(:,:,jj),1,N_semiz).*repelem(semiztransitions,1,2))',N_a*N_bothz,N_a*N_bothz); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]

    % First step of Tan improvement
    StationaryDistKron_jj=reshape(Gammatranspose*StationaryDistKron_jj,[N_a*N_semiz,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj)));
    StationaryDistKron_jj=reshape(StationaryDistKron_jj*pi_z,[N_a*N_bothz,1]);

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
