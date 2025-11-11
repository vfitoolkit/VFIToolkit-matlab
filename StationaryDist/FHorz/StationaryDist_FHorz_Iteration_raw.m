function StationaryDistKron=StationaryDist_FHorz_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,N_a,N_z,N_j,pi_z_J,Parameters)
% Will treat the agents as being on a continuum of mass 1.

Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1); % Note: add z' index following the z dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(Policy_aprimez);

StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;

StationaryDist_jj=sparse(gather(jequaloneDistKron));

IIind=1:1:N_a*N_z;
JJind=ones(N_a,N_z);

for jj=1:(N_j-1)
    
    Gammatranspose=sparse(Policy_aprimez(:,:,jj),IIind,JJind,N_a*N_z,N_a*N_z);

    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

    StationaryDistKron(:,jj+1)=full(StationaryDist_jj);
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
