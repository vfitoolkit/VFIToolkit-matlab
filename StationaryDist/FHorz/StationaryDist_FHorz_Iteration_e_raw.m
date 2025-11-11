function StationaryDistKron=StationaryDist_FHorz_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,Parameters)
% Will treat the agents as being on a continuum of mass 1.

% Ran a bunch of runtime tests. Tan improvement is always faster.
% Seems loop over e vs parallel over e is essentially break-even [see old version of this code on github, loop was removed in late2025].

Policy_aprimez=Policy_aprime+N_a*repmat(gpuArray(0:1:N_z-1),1,N_e,1); % Note: add z' index following the z&e dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(Policy_aprimez);

StationaryDistKron=zeros(N_a*N_z*N_e,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;

StationaryDist_jj=sparse(gather(jequaloneDistKron));

IIind=(1:1:N_a*N_z*N_e)';
JJind=ones(N_a,N_z*N_e);

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprimez(:,:,jj),IIind,JJind,N_a*N_z,N_a*N_z*N_e);


    % First step of Tan improvement
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(gather(pi_z_J(:,:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

    % Put e back into dist
    pi_e=sparse(gather(pi_e_J(:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).
    StationaryDist_jj=kron(pi_e,StationaryDist_jj);

    StationaryDistKron(:,jj+1)=gpuArray(full(StationaryDist_jj));
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
