function StationaryDistKron=StationaryDist_FHorz_SemiExo_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d1,N_a,N_z,N_semiz,N_j,pi_z_J,pi_semiz_J,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

N_bothz=N_z*N_semiz;

if N_d1==0
    optd2prime=gather(reshape(PolicyIndexesKron(1,:,:,:),[N_a*N_bothz,N_j])); % Note: column vector (conditional on jj)
else
    optd2prime=gather(reshape(ceil(PolicyIndexesKron(1,:,:,:)/N_d1),[N_a*N_bothz,N_j]));   
end
optaprime=gather(reshape(PolicyIndexesKron(2,:,:,:),[N_a*N_bothz,N_j])); % Note: column vector (conditional on jj)

%% Tan improvement verion

% To do Tan improvement with semiz shocks we treat the first step as
% (a,semiz,z) to (a',semiz',z) and then the second is the standard just
% updating z to z'.
StationaryDistKron=zeros(N_a*N_bothz,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;
StationaryDistKron_jj=sparse(gather(jequaloneDistKron));

II2=repelem((1:1:N_a*N_bothz),N_semiz,1);
% Note: repelem((1:1:N_a*N_bothz),N_semiz,1) is just a simpler way to write repelem((1:1:N_a*N_bothz)',1,N_semiz)'

semizindexcorrespondingtod2_c=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1); % precompute

for jj=1:(N_j-1)
    firststep=optaprime(:,jj)+kron(N_a*N_semiz*(0:1:N_z-1)',ones(N_a*N_semiz,1))+N_a*(0:1:N_semiz-1); % (a',semiz',z')-by-semiz
    % Note: optaprime and the z are columns, while semiz is a row that adds every semiz

    % Get the semiz transition probabilities into needed form
    pi_semiz_jj=pi_semiz_J(:,:,:,jj);
    % Get the right part of pi_semiz_J 
    % d2 depends on (a,z,semiz), and pi_semiz is going to be about (semiz,semiz'), so I need to put it all together as (a,z,semiz,semiz').
    % semizindexcorrespondingtod2_c=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1); % precomputed
    fullindex=semizindexcorrespondingtod2_c+N_semiz*(0:1:N_semiz-1)+(N_semiz*N_semiz)*(optd2prime(:,jj)-1);
    semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

    Gammatranspose=sparse(firststep',II2,semiztransitions',N_a*N_bothz,N_a*N_bothz); % From (a,semiz,z) to (a',semiz',z)

    % First step of Tan improvment
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
