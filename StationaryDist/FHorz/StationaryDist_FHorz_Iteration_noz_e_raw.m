function StationaryDistKron=StationaryDist_FHorz_Iteration_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,N_a,N_e,N_j,pi_e_J,Parameters)
% When this command was first created, I did a bunch of runtime tests to
% compare loop over e vs parallel over e. Runtimes were essentially
% break-even. Nowadays only the parallel is coded (you can find the loop
% version using github to access the version pre-2025).

Policy_aprime=gather(Policy_aprime);

StationaryDistKron=zeros(N_a*N_e,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;

StationaryDist_jj=sparse(gather(jequaloneDistKron));

IIind=1:1:N_a*N_e;
JJind=ones(N_a,N_e);

for jj=1:(N_j-1)
    
    Gammatranspose=sparse(Policy_aprime(:,:,jj),IIind,JJind,N_a,N_a*N_e);

    % Tan improvement not really relevant for without z shock
    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

    % Add e back into the distribution
    pi_e=sparse(gather(pi_e_J(:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).

    StationaryDist_jj=kron(pi_e,StationaryDist_jj);

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
