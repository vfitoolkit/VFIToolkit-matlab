function P=CreatePTransitionMatrix(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,Parameters,simoptions)

if simoptions.experienceasset>=1
    %% Experience asset
    error('Not yet implemented')
elseif simoptions.inheritanceasset==1
    %% Inheritance asset
    P=CreatePTransitionMatrix_interitanceasset(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,Parameters,simoptions);
    return
else
    %% Standard Endogenous States
    if N_semiz==0 && N_z==0 && N_e==0
        % no exogenous states
        Policy=reshape(Policy, [size(Policy,1),N_a]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        if l_a==1
            Policy_aprime(:,1)=shiftdim(Policy(l_d+1,:),1);
        elseif l_a==2
            Policy_aprime(:,1)=shiftdim(Policy(l_d+1,:)+n_a(1)*(Policy(l_d+2,:)-1),1);
        elseif l_a==3
            Policy_aprime(:,1)=shiftdim(Policy(l_d+1,:)+n_a(1)*(Policy(l_d+2,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:)-1),1);
        elseif l_a==4
            Policy_aprime(:,1)=shiftdim(Policy(l_d+1,:)+n_a(1)*(Policy(l_d+2,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:)-1),1);
        else
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            Policy_aprime=gather(Policy_aprime);
            % Precompute
            II2=(1:1:N_a)'; %  Index for this period (a)

            % P: full transition matrix on (a)
            P=sparse(II2,Policy_aprime,ones(N_a,1),N_a,N_a); % Note: sparse() will accumulate at repeated indices

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,2)=Policy_aprime(:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,2)=shiftdim((Policy(end-1,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,1)=1-PolicyProbs(:,2); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,2]
            Policy_aprime=gather(reshape(Policy_aprime,[N_a,2])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a,2])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a)',1,2); %  Index for this period (a), note the N_probs-copies

            % P: full transition matrix on (a)
            P=sparse(II2,Policy_aprime,PolicyProbs,N_a,N_a); % Note: sparse() will accumulate at repeated indices
        end

    elseif N_semiz==0 && N_z==0 && N_e>0
        % e
        Policy=reshape(Policy, [size(Policy,1),N_a,N_e]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_e,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_e,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
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
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0

            Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,1]));
            % Precompute
            II2=(1:1:N_a*N_e)'; %  Index for this period (a,e)

            % P: full transition matrix on (a)
            P=sparse(II2,Policy_aprime,ones(N_a*N_e,1),N_a*N_e,N_a); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order
            % Note: runtime test shows it is faster to add pi_e in at the end rather than creating Policy_aprimeeprime and building P with pi_e in it.

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_e,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,N_e,2]
            Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,2])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_e,2])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_e)',1,2); %  Index for this period (a), note the N_probs-copies

            % P: full transition matrix on (a)
            P=sparse(II2,Policy_aprime,PolicyProbs,N_a*N_e,N_a); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order
        end

    elseif N_semiz==0 && N_z>0 && N_e==0
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
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
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

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
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

    elseif N_semiz==0 && N_z>0 && N_e>=0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_z*N_e]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_z*N_e,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_z*N_e,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
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
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*N_e,1])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z*N_e,N_z])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_z*N_e)',1,N_z); %  Index for this period (a,z)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimezprime,repmat(repelem(pi_z,N_a,1),N_e,1),N_a*N_z*N_e,N_a*N_z); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_z*N_e,2,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,N_z*N_e,2]
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*N_e*2,1])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z*N_e,2*N_z])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z*N_e,2])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_z*N_e)',1,2*N_z); %  Index for this period (a,z), note the N_probs-copies

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimezprime,repmat(PolicyProbs,1,N_z).*repmat(repelem(pi_z,N_a,2),N_e,1),N_a*N_z*N_e,N_a*N_z); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order
        end

    elseif N_semiz>0 && N_z==0 && N_e==0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
        elseif l_a==2
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
        elseif l_a==3
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
        elseif l_a==4
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
        else
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0

            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz,1]);
            semizindex_short=repelem((1:1:N_semiz)',N_a,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz,1]
            Policy_aprimesemiz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz,1]),1,N_semizshort)+N_a*(idxshort(semizindex_short)-1); % Note: add semiz' index following the semiz' dimension
            % Policy_aprimesemiz is currently [N_a,N_semiz,N_semizshort]

            semiztransitions=gather(pi_semiz_short(semizindex_short));

            % Precompute
            II2=repmat((1:1:N_a*N_semiz)',1,N_semizshort); %  Index for this period (a,z)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemiz,semiztransitions,N_a*N_semiz,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz,2,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz,1]);
            semizindex_short=repelem((1:1:N_semiz)',N_a,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz,N_probs]
            Policy_aprimesemiz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz,N_probs]),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_probs,1); % Note: add semiz' index following the semiz' dimension
            % Policy_aprimesemiz is currently [N_a,N_semiz,N_probs*N_semizshort]

            PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz,N_probs]);
            PolicyProbs=repelem(gather(PolicyProbs),1,N_semizshort,1).*repmat(pi_semiz_short(semizindex_short),1,N_probs,1); % is of size [N_a*N_semiz,N_probs*N_semiz]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz)',1,N_semizshort*N_probs); % Index for this period (a,semiz), note the N_semizshort*N_probs-copies

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemiz,PolicyProbs,N_a*N_semiz,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
        end

    elseif N_semiz>0 && N_z==0 && N_e>0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz*N_e]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz*N_e,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz*N_e,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
        elseif l_a==2
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
        elseif l_a==3
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
        elseif l_a==4
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
        else
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            N_semizshort=max(max(max(sum((pi_semiz>0),2))));

            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1]);
            semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz*N_e,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz*N_e,1]
            Policy_aprimesemiz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_e,1]),1,N_semizshort)+N_a*(idxshort(semizindex_short)-1); % Note: add semiz' index following the semiz' dimension
            % Policy_aprimesemiz is currently [N_a,N_semiz*N_e,N_semizshort]

            semiztransitions=gather(pi_semiz_short(semizindex_short));

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_e)',1,N_semizshort); %  Index for this period (a,z)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemiz,semiztransitions,N_a*N_semiz*N_e,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz*N_e,2,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point
            
            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1]);
            semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz*N_e,N_semizshort]
            % used to index pi_semiz_J_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz*N_e,N_probs]
            Policy_aprimesemiz=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_e,N_probs]),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_probs); % Note: add semiz' index following the semiz' dimension, add z' index following the z dimension for Tan improvement
            % Policy_aprimesemizz is currently [N_a,N_semiz*N_z,N_semizshort*N_probs]

            PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_e,N_probs]);
            PolicyProbs=repelem(gather(PolicyProbs),1,N_semizshort,1).*repmat(pi_semiz_short(semizindex_short),1,N_probs,1); % is of size [N_a*N_semiz*N_e,N_probs*N_semiz]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_e)',1,N_semizshort*N_probs); % Index for this period (a,semiz), note the N_semizshort*N_probs-copies

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemiz,PolicyProbs,N_a*N_semiz*N_e,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order
        end

    elseif N_semiz>0 && N_z>0 && N_e==0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz*N_z]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz*N_z,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz*N_z,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
        elseif l_a==2
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
        elseif l_a==3
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
        elseif l_a==4
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
        else
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0

            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_z,1]);
            semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz*N_z,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz*N_z,1]
            Policy_aprimesemizzprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_z,1])+N_a*repmat((idxshort(semizindex_short)-1),1,N_z)+repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort); % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
            % Policy_aprimesemizz is currently [N_a,N_semiz*N_z,N_semizshort]

            semiztransitions=gather(pi_semiz_short(semizindex_short));

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z)',1,N_semizshort*N_z); %  Index for this period (a,z)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemizzprime,repmat(semiztransitions,1,N_z).*repelem(pi_z,N_a*N_semiz,N_semizshort),N_a*N_semiz*N_z,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz*N_z,2,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_z,1]);
            semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz*N_z,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz*N_z,N_probs]
            Policy_aprimesemizzprime=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_z,N_probs]),1,N_semiz*N_z)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_z*N_probs)+repmat(repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort),1,N_probs); % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
            % Policy_aprimesemizz is currently [N_a,N_semiz*N_z,N_semizshort*N_probs]

            PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_z,N_probs]);
            PolicyProbs=repelem(PolicyProbs,1,N_semizshort*N_z).*repelem(repmat(pi_semiz_short(semizindex_short),1,N_z),1,N_probs).*repelem(repmat(pi_z,1,N_probs),N_a*N_semizshort,N_semiz); % is of size [N_a*N_semiz*N_z,N_semiz*N_z*N_probs]
            PolicyProbs=gather(PolicyProbs);

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z)',1,2*N_semiz*N_z); %  Index for this period (a,z), note the N_probs-copies

            size(II2)
            size(Policy_aprimesemizzprime)
            size(PolicyProbs)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemizzprime,PolicyProbs,N_a*N_semiz*N_z,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
        end

    elseif N_semiz>0 && N_z>0 && N_e>0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz*N_z*N_e]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz*N_z*N_e,1,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz*N_z*N_e,2,'gpuArray'); % Policy_aprime has an additional final dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:),1);
        elseif l_a==2
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),1);
        elseif l_a==3
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),1);
        elseif l_a==4
            Policy_aprime(:,:,1)=shiftdim(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),1);
        else
            error('EvalFnOnAgentDist_AutoCorrTransProbs_InfHorz cannot handle length(n_a)>4, contact me if you need this')
        end
        
        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_z*N_e,1]);
            semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz*N_z*N_e,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz*N_z*N_e,1]
            Policy_aprimesemizzprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_z*N_e,1])+N_a*repmat((idxshort(semizindex_short)-1),1,N_z)+repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort); % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
            % Policy_aprimesemizz is currently [N_a,N_semiz*N_z*N_e,N_semizshort]

            semiztransitions=gather(pi_semiz_short(semizindex_short));

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z*N_e)',1,N_semizshort*N_z); %  Index for this period (a,z)

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemizzprime,repmat(semiztransitions,1,N_z).*repmat(repelem(pi_z,N_a*N_semiz,N_semizshort),N_e,1),N_a*N_semiz*N_z*N_e,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2)=Policy_aprime(:,:,1)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz*N_z*N_e,2,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2)=shiftdim((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),1); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1)=1-PolicyProbs(:,:,2); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz>0),2))));
            % Create smaller version of pi_semiz that eliminates as many non-zeros as possible
            [pi_semiz_short, idx] = sort(pi_semiz,2); % puts all the zeros on the left of the matrix

            pi_semiz_short=pi_semiz_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            Policy_dsemiexo=reshape(Policy_dsemiexo,[N_a*N_semiz*N_z*N_e,1]);
            semizindex_short=repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1)+gather((N_semiz*N_semizshort)*(Policy_dsemiexo-1)); % index for semiz, plus that for semiz' (in the semiz' dim) and dsemiexo; their indexes in pi_semiz_J
            pi_semiz_short=gather(pi_semiz_short);
            % semizindex_short is [N_a*N_semiz*N_z*N_e,N_semizshort]
            % used to index pi_semiz_short which is [N_semiz,N_semizshort,N_dsemiz]
            % and also to index the corresponding idxshort which is [N_semiz,N_semizshort,N_dsemiz]

            % Policy_aprime is currently [N_a,N_semiz*N_z*N_e,N_probs]
            Policy_aprimesemizzprime=repelem(reshape(gather(Policy_aprime),[N_a*N_semiz*N_z*N_e,N_probs]),1,N_semiz*N_z)+repmat(N_a*(idxshort(semizindex_short)-1),1,N_z*N_probs)+repmat(repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort),1,N_probs); % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
            % Policy_aprimesemizz is currently [N_a,N_semiz*N_z*N_e,N_semizshort,N_probs]

            PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_z*N_e,N_probs]);
            PolicyProbs=repelem(PolicyProbs,1,N_semizshort*N_z).*repelem(repmat(pi_semiz_short(semizindex_short),1,N_z),1,N_probs).*repelem(repmat(pi_z,N_e,N_probs),N_a*N_semizshort,N_semiz); % is of size [N_a*N_semiz*N_z,N_semiz*N_z*N_probs]
            PolicyProbs=gather(PolicyProbs);

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z*N_e)',1,2*N_semiz*N_z); %  Index for this period (a,z), note the N_probs-copies

            % P: full transition matrix on (a,z)
            P=sparse(II2,Policy_aprimesemizzprime,PolicyProbs,N_a*N_semiz*N_z*N_e,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
            % Now put the pi_e shocks into next period
            P=kron(sparse(gather(pi_e')),P); % note, reverse order
        end
        
    end
end









end