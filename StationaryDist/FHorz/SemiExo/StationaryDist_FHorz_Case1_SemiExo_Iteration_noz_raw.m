function StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_noz_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d1,N_d2,N_a,N_semiz,N_j,pi_semiz_J,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

% simoptions.tanimprovement=1

optdprime=gather(reshape(PolicyIndexesKron(1,:,:,:),[N_a*N_semiz,N_j])); % Note: column vector (conditional on jj)
optaprime=gather(reshape(PolicyIndexesKron(2,:,:,:),[N_a*N_semiz,N_j])); % Note: column vector (conditional on jj)

%% Tan improvement verion

% To do Tan improvement with semiz shocks we treat the first step as
% (a,semiz,z) to (a',semiz',z) and then the second is the standard just
% updating z to z'.
StationaryDistKron=zeros(N_a*N_semiz,N_j,'gpuArray');
StationaryDistKron(:,1)=jequaloneDistKron;
StationaryDistKron_jj=sparse(gather(jequaloneDistKron));

for jj=1:(N_j-1)
    firststep=optaprime(:,jj)+N_a*(0:1:N_semiz-1); % (a',semiz')-by-semiz
    % Note: optaprime is column, while semiz is a row that adds every semiz

    % Get the semiz transition probabilities into needed form
    pi_semiz_jj=pi_semiz_J(:,:,:,jj);
    % z transitions based on semiz
    dsub=ind2sub_vec_homemade([N_d1,N_d2],optdprime(:,jj));
    d2_c=dsub(:,2); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
    % Get the right part of pi_semiz_J 
    % d2 depends on (a,z,semiz), and pi_semiz is going to be about (semiz,semiz'), so I need to put it all together as (a,z,semiz,semiz').
    semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a,1));
    fullindex=semizindexcorrespondingtod2_c+N_semiz*(0:1:N_semiz-1)+(N_semiz*N_semiz)*(d2_c-1);
    semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

    Gammatranspose=sparse(firststep',repelem((1:1:N_a*N_semiz),N_semiz,1),semiztransitions',N_a*N_semiz,N_a*N_semiz); % From (a,semiz) to (a',semiz')
    % Note: repelem((1:1:N_a*N_semiz),N_semiz,1) is just a simpler way to write repelem((1:1:N_a*N_semiz)',1,N_semiz)'

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
