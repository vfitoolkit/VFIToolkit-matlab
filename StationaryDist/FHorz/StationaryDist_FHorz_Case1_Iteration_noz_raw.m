function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_noz_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_j,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

if N_d==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a,N_j]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a,N_j]);
end

optaprime=gather(optaprime);
jequaloneDistKron=gather(jequaloneDistKron);

StationaryDistKron=zeros(N_a,N_j);
StationaryDistKron(:,1)=jequaloneDistKron;

StationaryDistKron_jj=sparse(jequaloneDistKron);

for jj=1:(N_j-1)

    if simoptions.verbose==1
        fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
    end

    Gammatranspose=sparse(optaprime(1,:,jj),1:1:N_a,ones(N_a,1),N_a,N_a);

    StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;
    StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
end

if simoptions.parallel==2 % Move the solution to the gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
end

%%

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
