function StationaryDist=StationaryDist_InfHorz_Iteration_nProbs_raw(StationaryDist,Policy_aprime,PolicyProbs,N_probs,n_a1,n_a2,N_z,pi_z,simoptions)
% 'nProbs' refers to N_probs probabilities.
% Policy_aprime has an additional dimension of length N_probs which is the N_probs points (and contains only the aprime indexes, no d indexes as would usually be the case).
% PolicyProbs are the corresponding probabilities of each of these N_probs.

epsilon=1e-7;
total_zeros_created=0;
counter_at_max_a2=Inf;

% For grid interpolation, N_a2 arrives as zero.  We simplify the implementation
% by treating this N_a1x1 grid as an 1xN_a1 grid (with N_a1 spelled N_a2 below).
N_a1=prod(n_a1);
N_a2=prod(n_a2);
if N_a2==0
    N_a=N_a1;
elseif N_a1==0
    N_a=N_a2;
else
    N_a=N_a1*N_a2;
end

% Policy_aprime and PolicyProbs are currently [N_a,N_z,N_probs]
Policy_aprimez=Policy_aprime+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index following the z dimension [Tan improvement, z stays where it is]
Policy_aprimez=gather(reshape(Policy_aprimez,[N_a*N_z,N_probs])); % sparse() requires inputs to be 2-D
PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,N_probs])); % sparse() requires inputs to be 2-D

%% Use Tan improvement
% Cannot do max on sparse gpu matrix in Matlab yet, so this is on cpu

StationaryDist=sparse(gather(StationaryDist)); % use sparse matrix

% Precompute
II2=repmat((1:1:N_a*N_z)',1,N_probs); %  Index for this period (a,z), note the N_probs-copies

% Gamma for first step of Tan improvement
Gammatranspose=sparse(Policy_aprimez,II2,PolicyProbs,N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices

% pi_z for second step of Tan improvement
pi_z=sparse(gather(pi_z));

currdist=Inf;
counter=0;
while currdist>simoptions.tolerance && counter<simoptions.maxit

    % First step of Tan improvement
    StationaryDist=reshape(Gammatranspose*StationaryDist,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
    % Second step of Tan improvement
    StationaryDist=reshape(StationaryDist*pi_z,[N_a*N_z,1]);

    if simoptions.optimize_nProbs==1
        [StationaryDist,total_zeros_created,counter_at_max_a2]=StationaryDist_InfHorz_Optimize_nProbs_raw(StationaryDist,n_a1,n_a2,N_z, counter,epsilon,total_zeros_created,counter_at_max_a2,simoptions);
    end

    if rem(counter,simoptions.multiiter)==0
        StationaryDistOld=StationaryDist;
    elseif rem(counter,simoptions.multiiter)==10
        currdist=max(abs(full(StationaryDist)-full(StationaryDistOld)));
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


if isfinite(counter_at_max_a2)
    if N_a2>0
        warning("Max ExpAsset index %3d first reached at counter %3d \n", N_a2, counter_at_max_a2);
    else
        warning("Max grid-interpolated asset index %3d first reached at counter %3d \n", N_a1, counter_at_max_a2);
    end
end

if simoptions.verbose
    if total_zeros_created>0
        fprintf("With epsilon = %.2e, total zeros created = %d \n", epsilon, total_zeros_created);
        if ~isfinite(counter_at_max_a2)
            max_a=nan;
            if N_a2==0
                temp=reshape(StationaryDist,[N_a1,N_z]);
                [a1,~]=ind2sub(size(temp),find(temp~=0));
                max_a=max(a1);
            else
                if N_a1>0
                    temp=reshape(full(StationaryDist),[N_a1,N_a2,N_z]);
                    [~,a2,~]=ind2sub(size(temp),find(temp~=0));
                else
                    temp=reshape(StationaryDist,[1,N_a2,N_z]);
                    [~,a2,~]=ind2sub(size(temp),find(temp~=0));
                end
                max_a=max(a2);
            end
            if N_a2>0
                fprintf("Max InfHorz ExpAsset index reached: %3d (of %3d); counter %d \n", max_a, N_a2, counter);
            else
                fprintf("Max InfHorz grid-interpolated asset index reached: %3d (of %3d); counter %3d \n", max_a, N_a1, counter);
            end
        end
    end
end

end