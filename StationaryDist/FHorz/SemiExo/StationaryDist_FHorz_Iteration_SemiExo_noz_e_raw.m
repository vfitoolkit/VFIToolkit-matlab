function StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_noz_e_raw(jequaloneDistKron, AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_e,N_j,pi_semiz_J,pi_e_J,Parameters)
% Will treat the agents as being on a continuum of mass 1.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz&e together into the 1st dim.

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

Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1,N_j]);
semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1))+(N_semiz*N_semizshort*N_dsemiz)*shiftdim((0:1:N_j-1),-1); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
pi_semiz_J_short=gather(pi_semiz_J_short);
% semizindex_short is [N_a*N_semiz*N_e,N_semizshort,N_j]
% used to index pi_semiz_J_short which is [N_semiz,N_semizshort,N_dsemiz,N_j]
% and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz,N_j]

% Policy_aprime is currently [N_a,N_semiz*N_e,1,N_j]
Policy_aprimesemiz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_e,1,N_j]),1,N_semizshort)+N_a*(idxshort(semizindex_short)-1); % Note: add semiz' index following the semiz' dimension
% Policy_aprimesemiz is currently [N_a,N_semiz*N_e,N_semizshort,N_j]

semiztransitions=gather(pi_semiz_J_short(semizindex_short));

%% No z, so no Tan improvement

StationaryDist=zeros(N_a*N_semiz*N_e,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDist_jj=sparse(gather(jequaloneDistKron));

II2=repelem((1:1:N_a*N_semiz*N_e)',1,N_semizshort);

for jj=1:(N_j-1)

    Gammatranspose=sparse(Policy_aprimesemiz(:,:,jj),II2,semiztransitions(:,:,jj),N_a*N_semiz,N_a*N_semiz*N_e); % From (a,semiz,e) to (a',semiz')

    % No z, so just one-step for iteration
    StationaryDist_jj=Gammatranspose*StationaryDist_jj;

    % Add e back into the distribution
    pi_e=sparse(gather(pi_e_J(:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).
    StationaryDist_jj=kron(pi_e,StationaryDist_jj);

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
