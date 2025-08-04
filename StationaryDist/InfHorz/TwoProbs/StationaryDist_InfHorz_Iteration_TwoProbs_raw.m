function StationaryDistKron=StationaryDist_InfHorz_Iteration_TwoProbs_raw(StationaryDistKron,Policy_aprime,PolicyProbs,N_a,N_z,pi_z,simoptions)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1); % Note: add z index following the z dimension
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z,2])); % (a,z,2)
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,2])); % (a,z,2)

%% Use Tan improvement
% Cannot do max on sparse gpu matrix in Matlab yet, so this is on cpu

StationaryDistKron=sparse(gather(StationaryDistKron)); % use sparse matrix

% Precompute
II2=[1:1:N_a*N_z; 1:1:N_a*N_z]'; % Index for this period (a,z), note the 2 copies

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimez,II2,PolicyProbs,N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices [only relevant at grid end points]
% pi_z for second step of Tan improvement
pi_z=sparse(gather(pi_z));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    % First step of Tan improvement
    StationaryDistKron=reshape(Gammatranspose*StationaryDistKron,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    % Second step of Tan improvement
    StationaryDistKron=reshape(StationaryDistKron*pi_z,[N_a*N_z,1]);

    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDistKron;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(StationaryDistKron-StationaryDistKronOld));
    end

    counter=counter+1;

    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, currdist/simoptions.tolerance)
        end
    end

end

% Convert back to full matrix for output
StationaryDistKron=full(StationaryDistKron);

end