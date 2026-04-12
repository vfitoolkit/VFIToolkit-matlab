function StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_z,N_j,pi_semiz_J,pi_z_J,Parameters)
% Will treat the agents as being on a continuum of mass 1.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz&z together into the 1st dim.

%%
% It is likely that most of the elements in pi_semiz_J are zero, we can
% take advantage of this to speed things up. Ignore for a moment the
% dependence on d and j, and pretend it is just a N_semiz-by-N_semiz
% matrix. Then we can calculate N_semizshort=max(sum((pi_semiz>0),2)), the
% maximum number of non-zeros in any row of pi_semiz. And we then use this
% in place of N_semiz as the second dimension.

N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
% Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
[pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
idxshort=idx(:,end-N_semizshort+1:end,:,:);

Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_z,1,N_j]);
semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1))+(N_semiz*N_semizshort*N_dsemiz)*shiftdim((0:1:N_j-1),-1); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
pi_semiz_J_short=gather(pi_semiz_J_short);
% semizindex_short is [N_a*N_semiz*N_z,N_semizshort,N_j]
% used to index pi_semiz_J_short which is [N_semiz,N_semizshort,N_dsemiz,N_j]
% and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz,N_j]

% Policy_aprime is currently [N_a,N_semiz*N_z,1,N_j]
Policy_aprimesemizz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_z,1,N_j]),1,N_semizshort)+N_a*(idxshort(semizindex_short)-1)+repelem(N_a*N_semiz*(0:1:N_z-1)',N_a*N_semiz,1); % Note: add semiz' index following the semiz' dimension, add z' index following the z dimension for Tan improvement
% Policy_aprimesemizz is currently [N_a,N_semiz*N_z,N_semizshort,N_j]

semiztransitions=gather(pi_semiz_J_short(semizindex_short));

pi_z_J=gather(pi_z_J);
N_bothz=N_semiz*N_z;

%% Tan improvement verion
% To do Tan improvement with semiz shocks we treat the first step as
% (a,semiz,z) to (a',semiz',z) and then the second is the standard just
% updating z to z'.

StationaryDist=zeros(N_a*N_bothz,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron));

II2=repelem((1:1:N_a*N_bothz)',1,N_semizshort);

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprimesemizz(:,:,jj),II2,semiztransitions(:,:,jj),N_a*N_bothz,N_a*N_bothz); % From (a,semiz,z) to (a',semiz',z)

    % First step of Tan improvment
    StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a*N_semiz,N_z]);

    % Second step of Tan improvement
    pi_z=sparse(pi_z_J(:,:,jj));
    StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_bothz,1]);

    StationaryDist(:,jj+1)=gpuArray(full(StationaryDist_jj));
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

StationaryDist=StationaryDist.*AgeWeights;


end
