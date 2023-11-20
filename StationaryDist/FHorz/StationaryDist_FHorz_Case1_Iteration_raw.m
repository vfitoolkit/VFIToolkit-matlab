function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z_J,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

StationaryDistKron=zeros(N_a*N_z,N_j);
StationaryDistKron(:,1)=gather(jequaloneDistKron);

StationaryDist_jj=sparse(gather(jequaloneDistKron));

if N_d==0
    PolicyIndexesKron=gather(reshape(PolicyIndexesKron,[1,N_a*N_z,N_j]));
else
    PolicyIndexesKron=gather(reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_z,N_j]));
end

for jj=1:(N_j-1)
    if simoptions.verbose==1
        fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
    end

    optaprime=PolicyIndexesKron(1,:,jj);

    firststep=optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a));
    Gammatranspose=sparse(firststep,1:1:N_a*N_z,ones(N_a*N_z,1),N_a*N_z,N_a*N_z);

    pi_z=sparse(gather(pi_z_J(:,:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).

    % Two steps of the Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

    StationaryDistKron(:,jj+1)=full(StationaryDist_jj);
end
if simoptions.parallel==2 % Move result to gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
    % Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.
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
