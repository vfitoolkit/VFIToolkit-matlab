function StationaryDistKron=StationaryDist_FHorz_Iteration_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,N_a,N_j,Parameters)
% Will treat the agents as being on a continuum of mass 1.

Policy_aprime=gather(Policy_aprime);

StationaryDistKron=zeros(N_a,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;

StationaryDistKron_jj=sparse(gather(jequaloneDistKron));

IIind=(1:1:N_a)';
JJind=ones(N_a,1);

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprime(:,jj),IIind,JJind,N_a,N_a);

    StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;
    StationaryDistKron(:,jj+1)=gpuArray(full(StationaryDistKron_jj));
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
