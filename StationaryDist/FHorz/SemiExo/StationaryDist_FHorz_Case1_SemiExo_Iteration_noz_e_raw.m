function StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d1,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

if N_d1==0
    optd2prime=gather(reshape(PolicyIndexesKron(1,:,:,:,:),[N_a*N_semiz*N_e,N_j])); % Note: column vector (conditional on jj)
else
    optd2prime=gather(reshape(ceil(PolicyIndexesKron(1,:,:,:,:)/N_d1),[N_a*N_semiz*N_e,N_j]));   
end
optaprime=gather(reshape(PolicyIndexesKron(2,:,:,:,:),[N_a*N_semiz*N_e,N_j])); % Note: column vector (conditional on jj)

%% No z, so no Tan improvement

StationaryDistKron=zeros(N_a*N_semiz*N_e,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;
StationaryDistKron_jj=sparse(gather(jequaloneDistKron));

II2=repelem((1:1:N_a*N_semiz*N_e),N_semiz,1);
% Note: repelem((1:1:N_a*N_semiz*N_e),N_semiz,1) is just a simpler way to write repelem((1:1:N_a*N_semiz*N_e)',1,N_semiz)'

semizindexcorrespondingtod2_c=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1); % precompute

for jj=1:(N_j-1)
    firststep=optaprime(:,jj)+N_a*(0:1:N_semiz-1); % (a',semiz',e')-by-semiz
    % Note: optaprime is column, while semiz is a row that adds every semiz

    % Get the semiz transition probabilities into needed form
    pi_semiz_jj=pi_semiz_J(:,:,:,jj);
    % Get the right part of pi_semiz_J 
    % d2 depends on (a,z,semiz), and pi_semiz is going to be about (semiz,semiz'), so I need to put it all together as (a,z,semiz,semiz').
    % semizindexcorrespondingtod2_c=repelem((1:1:N_semiz)',N_a,1); % precomputed
    fullindex=semizindexcorrespondingtod2_c+N_semiz*(0:1:N_semiz-1)+(N_semiz*N_semiz)*(optd2prime(:,jj)-1);
    semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,e-by-semiz')

    Gammatranspose=sparse(firststep',II2,semiztransitions',N_a*N_semiz,N_a*N_semiz*N_e); % From (a,semiz,e) to (a',semiz')

    StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;

    % Add e back into the distribution
    pi_e=sparse(gather(pi_e_J(:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)). 
    
    StationaryDistKron_jj=kron(pi_e,StationaryDistKron_jj);

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
