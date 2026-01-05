function StationaryDist=StationaryDist_InfHorz_Iteration_zprime_nProbs_raw(StationaryDist,Policy_aprime,PolicyProbs,N_probs,N_a,N_z,pi_z,simoptions)
% 'zprime' refers to Policy_aprime depending on zprime: it is of size [N_a,N_z,N_zprime,N_probs]
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these N_probs.

N_zprime=N_z; % Just to make the code easier to read

% Policy_aprime and PolicyProbs are currently [N_a,N_z,N_zprime,N_probs]
Policy_aprimezprime=Policy_aprime+N_a*shiftdim(gpuArray(0:1:N_z-1),-1);  % Note: add z' index following the z' dimension
Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,N_zprime*N_probs])); % sparse() requires inputs to be 2-D
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,N_zprime*N_probs])); % sparse() requires inputs to be 2-D

%% Because Policy depends on zprime, I don't think Tan improvement can be used (I've not thought hard, but fairly confident it won't)

StationaryDist=sparse(gather(StationaryDist)); % use sparse matrix

% Precompute
II2=repmat((1:1:N_a*N_z)',1,N_zprime*N_probs); %  Index for this period (a,z), note the N_zprime*N_probs-copies

pi_z=sparse(gather(repelem(repmat(pi_z,1,N_probs),N_a,1)));

% Transition matrix
TranstionMatrixtranspose=sparse(Policy_aprimezprime,II2,PolicyProbs.*pi_z,N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    % Iterate on agent distribution
    StationaryDist=TranstionMatrixtranspose*StationaryDist;

    if rem(counter,simoptions.multiiter)==0
        StationaryDistOld=StationaryDist;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(StationaryDist-StationaryDistOld));
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