function StationaryDist=StationaryDist_InfHorz_Iteration_nProbs_noz_e_raw(StationaryDist,Policy_aprime,PolicyProbs,N_probs,N_a,N_e,pi_e,simoptions)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these N_probs.

% Policy_aprime and PolicyProbs are currently [N_a,N_e,N_probs]
Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,N_probs])); % sparse() requires inputs to be 2-D
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_e,N_probs])); % sparse() requires inputs to be 2-D

%% Use Tan improvement
% Cannot do max on sparse gpu matrix in Matlab yet, so this is on cpu

StationaryDist=sparse(gather(StationaryDist)); % use sparse matrix

% Precompute
II2=repmat((1:1:N_a*N_e)',1,N_probs); %  Index for this period (a,z), note the N_probs-copies

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprime,II2,PolicyProbs,N_a,N_a*N_e); % Note: sparse() will accumulate at repeated indices

% pi_e
pi_e=sparse(gather(pi_e));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    % First step of Tan improvement
    StationaryDist=Gammatranspose*StationaryDist; %No point checking distance every single iteration. Do 100, then check.

    % Put e back into dist
    StationaryDist=kron(pi_e,StationaryDist);

    if rem(counter,simoptions.multiiter)==0
        StationaryDistKronOld=StationaryDist;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(StationaryDist-StationaryDistKronOld));
    end

    counter=counter+1;

    if simoptions.verbose==1
        if rem(counter,50)==0
            fprintf('StationaryDist_Case1: after %i iterations the current distance ratio is %8.6f (currdist/tolerance, convergence when reaches 1) \n', counter, currdist/simoptions.tolerance)
        end
    end

end

% Convert back to full matrix for output
StationaryDist=gpuArray(full(StationaryDist));

end