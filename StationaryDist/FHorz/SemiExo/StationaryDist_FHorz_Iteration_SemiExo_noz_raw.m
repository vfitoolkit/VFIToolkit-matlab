function StationaryDist=StationaryDist_FHorz_Iteration_SemiExo_noz_raw(jequaloneDistKron,AgeWeightParamNames,Policy_dsemiexo,Policy_aprime,N_dsemiz,N_a,N_semiz,N_j,pi_semiz_J,Parameters)
% Will treat the agents as being on a continuum of mass 1.

% When we use semiz, we need to use a different shape for Policy_aprime.
% sparse() limits us to 2-D, and we need to get in a semiz' dimension. So I
% put a&semiz together into the 1st dim.

% Note: Tried doing creation of semiztransitions, etc., in parallel over jj
% before the loop. Having it in the loop massively reduces the memory-use which
% was a bottleneck when parallel over jj, and the runtime is actually if
% anything faster in the loop version that it was parallel over jj.

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

Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz,1,N_j]));
Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz,1,N_j]);
pi_semiz_J_short=gather(pi_semiz_J_short);
idxshort=gather(idxshort);
semizindexbase=repelem((1:1:N_semiz)',N_a,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
% semizindex_short_jj (built per age below) is [N_a*N_semiz,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j]

%% No z, so no Tan improvement

StationaryDist=zeros(N_a*N_semiz,N_j,'gpuArray');
StationaryDist(:,1)=jequaloneDistKron;
StationaryDistKron_jj=sparse(gather(jequaloneDistKron));

II2=repelem((1:1:N_a*N_semiz)',1,N_semizshort);

for jj=1:(N_j-1)
    semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1);
    Policy_aprimesemiz_jj=repelem(Policy_aprime(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1); % Note: add semiz' index following the semiz' dimension
    semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

    Gammatranspose=sparse(Policy_aprimesemiz_jj,II2,semiztransitions_jj,N_a*N_semiz,N_a*N_semiz); % From (a,semiz) to (a',semiz')

    % No z, so just one-step for iteration
    StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;

    StationaryDist(:,jj+1)=gpuArray(full(StationaryDistKron_jj));
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
