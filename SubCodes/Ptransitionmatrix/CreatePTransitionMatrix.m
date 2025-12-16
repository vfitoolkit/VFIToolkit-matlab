function P=CreatePTransitionMatrix(Policy,l_d,l_a,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,simoptions)

if N_semiz==0 && N_z>0 && N_e==0
    Policy=reshape(Policy, [size(Policy,1),N_a,N_z]);

    if simoptions.gridinterplayer==0
        Policy_aprime=zeros(N_a,N_z,1,'gpuArray');
    elseif simoptions.gridinterplayer==1
        Policy_aprime=zeros(N_a,N_z,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
    end

    if l_a==1
        Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
    elseif l_a==2
        Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
    elseif l_a==3
        Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
    elseif l_a==4
        Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
    else
        error('EvalFnOnAgentDist_CorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
    end

    % Constructing Gammatranspose is done differently if using grid interpolation layers
    if simoptions.gridinterplayer==0
        Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z,1])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
        Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,N_z])); % sparse() requires inputs to be 2-D
        % Precompute
        II2=repmat((1:1:N_a*N_z)',1,N_z); %  Index for this period (a,z)

        % P: full transition matrix on (a,z)
        P=sparse(II2,Policy_aprimezprime,repelem(pi_z,N_a,1),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices

    elseif simoptions.gridinterplayer==1
        Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

        % First, get Gamma
        PolicyProbs=zeros(N_a,N_z,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

        PolicyProbs(:,:,2)=shiftdim((Policy(end,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point
        PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

        % Policy_aprime and PolicyProbs are currently [N_a,N_z,2]
        Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*2,1])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
        Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,2*N_z])); % sparse() requires inputs to be 2-D
        PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,2])); % sparse() requires inputs to be 2-D

        % Precompute
        II2=repmat((1:1:N_a*N_z)',1,2*N_z); %  Index for this period (a,z), note the N_probs-copies

        % P: full transition matrix on (a,z)
        P=sparse(II2,Policy_aprimezprime,repmat(PolicyProbs,1,N_z).*repelem(pi_z,N_a,2),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
    end   

end








end