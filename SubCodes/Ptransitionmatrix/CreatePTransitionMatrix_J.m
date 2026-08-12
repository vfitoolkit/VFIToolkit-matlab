function P_cell=CreatePTransitionMatrix_J(Policy,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,pi_semiz_J,pi_z_J,pi_e_J,Parameters,simoptions)
% Age-stacked version of CreatePTransitionMatrix. Does the index arithmetic
% for every age at once (parallel over j, single gather), then just builds
% the per-age sparse matrices in a thin loop [sparse() is 2-D so the per-age
% sparse() calls cannot be avoided]. Exception: with semiz the semiz index
% is built per age inside the loop (as in the StationaryDist SemiExo raws),
% which keeps the temporaries a factor N_j smaller.
%
% Output: P_cell is cell(N_j-1,1); P_cell{jj} is the transition matrix for
% the transition from age jj to age jj+1, identical to calling
% CreatePTransitionMatrix age-by-age.
%
% Policy has a trailing age dimension: (size1,N_a,N_j) when there are no
% exogenous shocks, otherwise (size1,N_a,N_semiz*N_z*N_e,N_j).
% pi_semiz_J is (N_semiz,N_semiz,N_dsemiz,N_j-1), pi_z_J is (N_z,N_z,N_j-1),
% pi_e_J is (N_e,N_j); they are [] for shocks that are not used. Slice jj of
% pi_semiz_J/pi_z_J is the transition from age jj to age jj+1, so only
% slices 1:N_j-1 exist (nothing is ever read from a final-period slice).
% Note: e is iid at the start of next period, so the transition from age jj
% to jj+1 uses pi_e_J(:,jj+1).

% Get N_j from the trailing dimension of Policy
if N_semiz==0 && N_z==0 && N_e==0
    N_j=size(Policy,3);
else
    N_j=size(Policy,4);
end

P_cell=cell(N_j-1,1);

if simoptions.experienceasset>=1
    %% Experience asset
    error('Not yet implemented')
elseif simoptions.inheritanceasset==1
    %% Inheritance asset: no parallelization over j, just call the age-by-age routine
    for jj=1:N_j-1
        if N_semiz==0 && N_z==0 && N_e==0
            Policy_jj=Policy(:,:,jj);
        else
            Policy_jj=Policy(:,:,:,jj);
        end
        if N_semiz>0
            pi_semiz=pi_semiz_J(:,:,:,jj);
        else
            pi_semiz=[];
        end
        if N_z>0
            pi_z=pi_z_J(:,:,jj);
        else
            pi_z=[];
        end
        if N_e>0
            pi_e=pi_e_J(:,jj+1);
        else
            pi_e=[];
        end
        P_cell{jj}=CreatePTransitionMatrix(Policy_jj,l_d,l_a,n_d,n_a,n_z,N_a,N_semiz,N_z,N_e,pi_semiz,pi_z,pi_e,Parameters,simoptions);
    end
    return
else
    %% Standard Endogenous States
    if N_semiz==0 && N_z==0 && N_e==0
        % no exogenous states
        Policy=reshape(Policy, [size(Policy,1),N_a,N_j]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        if l_a==1
            Policy_aprime(:,1,:)=reshape(Policy(l_d+1,:,:),[N_a,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,1,:)=reshape(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1),[N_a,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,1,:)=reshape(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1),[N_a,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,1,:)=reshape(Policy(l_d+1,:,:)+n_a(1)*(Policy(l_d+2,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:)-1),[N_a,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            Policy_aprime=gather(Policy_aprime);
            % Precompute
            II2=(1:1:N_a)'; %  Index for this period (a)

            for jj=1:N_j-1
                % P: full transition matrix on (a)
                P_cell{jj}=sparse(II2,Policy_aprime(:,:,jj),ones(N_a,1),N_a,N_a); % Note: sparse() will accumulate at repeated indices
            end

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,2,:)=Policy_aprime(:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,2,N_j,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,2,:)=reshape((Policy(end-1,:,:)-1)/(simoptions.ngridinterp+1),[N_a,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,1,:)=1-PolicyProbs(:,2,:); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,2,N_j]
            Policy_aprime=gather(Policy_aprime); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(PolicyProbs); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a)',1,2); %  Index for this period (a), note the N_probs-copies

            for jj=1:N_j-1
                % P: full transition matrix on (a)
                P_cell{jj}=sparse(II2,Policy_aprime(:,:,jj),PolicyProbs(:,:,jj),N_a,N_a); % Note: sparse() will accumulate at repeated indices
            end
        end

    elseif N_semiz==0 && N_z==0 && N_e>0
        % e
        Policy=reshape(Policy, [size(Policy,1),N_a,N_e,N_j]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_e,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_e,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_e,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_e,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_e,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_e,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0

            Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,1,N_j]));
            % Precompute
            II2=(1:1:N_a*N_e)'; %  Index for this period (a,e)

            for jj=1:N_j-1
                % P: full transition matrix on (a)
                P=sparse(II2,Policy_aprime(:,:,jj),ones(N_a*N_e,1),N_a*N_e,N_a); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
                % Note: runtime test shows it is faster to add pi_e in at the end rather than creating Policy_aprimeeprime and building P with pi_e in it.
            end

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_e,2,N_j,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_e,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,N_e,2,N_j]
            Policy_aprime=gather(reshape(Policy_aprime,[N_a*N_e,2,N_j])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_e,2,N_j])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_e)',1,2); %  Index for this period (a), note the N_probs-copies

            for jj=1:N_j-1
                % P: full transition matrix on (a)
                P=sparse(II2,Policy_aprime(:,:,jj),PolicyProbs(:,:,jj),N_a*N_e,N_a); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end
        end

    elseif N_semiz==0 && N_z>0 && N_e==0
        Policy=reshape(Policy, [size(Policy,1),N_a,N_z,N_j]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_z,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_z,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_z,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_z,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_z,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_z,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z,1,N_j])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,N_z,N_j])); % sparse() requires inputs to be 2-D
            % Precompute
            II2=repmat((1:1:N_a*N_z)',1,N_z); %  Index for this period (a,z)

            for jj=1:N_j-1
                % P: full transition matrix on (a,z)
                P_cell{jj}=sparse(II2,Policy_aprimezprime(:,:,jj),repelem(pi_z_J(:,:,jj),N_a,1),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
            end

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_z,2,N_j,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_z,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,N_z,2,N_j]
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*2,1,N_j])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z,2*N_z,N_j])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z,2,N_j])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_z)',1,2*N_z); %  Index for this period (a,z), note the N_probs-copies

            for jj=1:N_j-1
                % P: full transition matrix on (a,z)
                P_cell{jj}=sparse(II2,Policy_aprimezprime(:,:,jj),repmat(PolicyProbs(:,:,jj),1,N_z).*repelem(pi_z_J(:,:,jj),N_a,2),N_a*N_z,N_a*N_z); % Note: sparse() will accumulate at repeated indices
            end
        end

    elseif N_semiz==0 && N_z>0 && N_e>=0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_z*N_e,N_j]);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_z*N_e,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_z*N_e,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_z*N_e,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_z*N_e,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_z*N_e,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_z*N_e,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*N_e,1,N_j])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z*N_e,N_z,N_j])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_z*N_e)',1,N_z); %  Index for this period (a,z)

            for jj=1:N_j-1
                % P: full transition matrix on (a,z)
                P=sparse(II2,Policy_aprimezprime(:,:,jj),repmat(repelem(pi_z_J(:,:,jj),N_a,1),N_e,1),N_a*N_z*N_e,N_a*N_z); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end

        elseif simoptions.gridinterplayer==1
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_z*N_e,2,N_j,'gpuArray');% PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_z*N_e,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            % Policy_aprime and PolicyProbs are currently [N_a,N_z*N_e,2,N_j]
            Policy_aprimezprime=reshape(Policy_aprime,[N_a*N_z*N_e*2,1,N_j])+N_a*gpuArray(0:1:N_z-1);  % Note: add z' index in a new dimension, so all possible zprime
            Policy_aprimezprime=gather(reshape(Policy_aprimezprime,[N_a*N_z*N_e,2*N_z,N_j])); % sparse() requires inputs to be 2-D
            PolicyProbs=gather(reshape(PolicyProbs,[N_a*N_z*N_e,2,N_j])); % sparse() requires inputs to be 2-D

            % Precompute
            II2=repmat((1:1:N_a*N_z*N_e)',1,2*N_z); %  Index for this period (a,z), note the N_probs-copies

            for jj=1:N_j-1
                % P: full transition matrix on (a,z)
                P=sparse(II2,Policy_aprimezprime(:,:,jj),repmat(PolicyProbs(:,:,jj),1,N_z).*repmat(repelem(pi_z_J(:,:,jj),N_a,2),N_e,1),N_a*N_z*N_e,N_a*N_z); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end
        end

    elseif N_semiz>0 && N_z==0 && N_e==0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz,N_j]);
        N_dsemiz=size(pi_semiz_J,3);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_semiz,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_semiz,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_semiz,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_semiz,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0

            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz,1,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz,1,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repelem((1:1:N_semiz)',N_a,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            % semizindex_short_jj (built per age below) is [N_a*N_semiz,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz)',1,N_semizshort); %  Index for this period (a,z)

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemiz_jj=repelem(Policy_aprime(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1); % Note: add semiz' index following the semiz' dimension
                semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

                % P: full transition matrix on (a,z)
                P_cell{jj}=sparse(II2,Policy_aprimesemiz_jj,semiztransitions_jj,N_a*N_semiz,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
            end

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz,2,N_j,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_semiz,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz,N_probs,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz,N_probs,N_j]);
            PolicyProbs=reshape(gather(PolicyProbs),[N_a*N_semiz,N_probs,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repelem((1:1:N_semiz)',N_a,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            % semizindex_short_jj (built per age below) is [N_a*N_semiz,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz)',1,N_semizshort*N_probs); % Index for this period (a,semiz), note the N_semizshort*N_probs-copies

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemiz_jj=repelem(Policy_aprime(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,N_probs); % Note: add semiz' index following the semiz' dimension
                PolicyProbs_jj=repelem(PolicyProbs(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,N_probs); % is of size [N_a*N_semiz,N_probs*N_semizshort]

                % P: full transition matrix on (a,z)
                P_cell{jj}=sparse(II2,Policy_aprimesemiz_jj,PolicyProbs_jj,N_a*N_semiz,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
            end
        end

    elseif N_semiz>0 && N_z==0 && N_e>0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz*N_e,N_j]);
        N_dsemiz=size(pi_semiz_J,3);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz*N_e,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz*N_e,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_semiz*N_e,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_semiz*N_e,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_semiz*N_e,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_semiz*N_e,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));

            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz*N_e,1,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_e,1,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            % semizindex_short_jj (built per age below) is [N_a*N_semiz*N_e,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_e)',1,N_semizshort); %  Index for this period (a,z)

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemiz_jj=repelem(Policy_aprime(:,1,jj),1,N_semizshort)+N_a*(idxshort(semizindex_short_jj)-1); % Note: add semiz' index following the semiz' dimension
                semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

                % P: full transition matrix on (a,z)
                P=sparse(II2,Policy_aprimesemiz_jj,semiztransitions_jj,N_a*N_semiz*N_e,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz*N_e,2,N_j,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_semiz*N_e,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz*N_e,N_probs,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz*N_e,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_e,N_probs,N_j]);
            PolicyProbs=reshape(gather(PolicyProbs),[N_a*N_semiz*N_e,N_probs,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_e,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            % semizindex_short_jj (built per age below) is [N_a*N_semiz*N_e,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_e)',1,N_semizshort*N_probs); % Index for this period (a,semiz), note the N_semizshort*N_probs-copies

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemiz_jj=repelem(Policy_aprime(:,:,jj),1,N_semizshort)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,N_probs); % Note: add semiz' index following the semiz' dimension
                PolicyProbs_jj=repelem(PolicyProbs(:,:,jj),1,N_semizshort).*repmat(pi_semiz_J_short(semizindex_short_jj),1,N_probs); % is of size [N_a*N_semiz*N_e,N_probs*N_semizshort]

                % P: full transition matrix on (a,z)
                P=sparse(II2,Policy_aprimesemiz_jj,PolicyProbs_jj,N_a*N_semiz*N_e,N_a*N_semiz); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end
        end

    elseif N_semiz>0 && N_z>0 && N_e==0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz*N_z,N_j]);
        N_dsemiz=size(pi_semiz_J,3);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz*N_z,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz*N_z,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_semiz*N_z,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_semiz*N_z,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_semiz*N_z,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_semiz*N_z,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0

            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz*N_z,1,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz*N_z,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_z,1,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            zprimeoffset=repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort); % z' index following the z' dimension
            % semizindex_short_jj (built per age below) is [N_a*N_semiz*N_z,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z)',1,N_semizshort*N_z); %  Index for this period (a,z)

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemizzprime_jj=Policy_aprime(:,1,jj)+N_a*repmat((idxshort(semizindex_short_jj)-1),1,N_z)+zprimeoffset; % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
                semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

                % P: full transition matrix on (a,z)
                P_cell{jj}=sparse(II2,Policy_aprimesemizzprime_jj,repmat(semiztransitions_jj,1,N_z).*repelem(pi_z_J(:,:,jj),N_a*N_semiz,N_semizshort),N_a*N_semiz*N_z,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
            end

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz*N_z,2,N_j,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_semiz*N_z,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz*N_z,N_probs,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz*N_z,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_z,N_probs,N_j]);
            PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_z,N_probs,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_z,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            zprimeoffset=repmat(repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort),1,N_probs); % z' index following the z' dimension
            % semizindex_short_jj (built per age below) is [N_a*N_semiz*N_z,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z)',1,N_semizshort*N_z*N_probs); %  Index for this period (a,z), note the N_probs-copies
            % Columns are ordered (semiz',z',prob): semiz' fastest, then z', then the two grid-interp points

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemizzprime_jj=repelem(Policy_aprime(:,:,jj),1,N_semizshort*N_z)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,N_z*N_probs)+zprimeoffset; % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
                semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

                PolicyProbs_jj=repelem(PolicyProbs(:,:,jj),1,N_semizshort*N_z).*repmat(semiztransitions_jj,1,N_z*N_probs).*repmat(repelem(pi_z_J(:,:,jj),N_a*N_semiz,N_semizshort),1,N_probs); % is of size [N_a*N_semiz*N_z,N_semizshort*N_z*N_probs]
                PolicyProbs_jj=gather(PolicyProbs_jj);

                % P: full transition matrix on (a,z)
                P_cell{jj}=sparse(II2,Policy_aprimesemizzprime_jj,PolicyProbs_jj,N_a*N_semiz*N_z,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
            end
        end

    elseif N_semiz>0 && N_z>0 && N_e>0

        Policy=reshape(Policy, [size(Policy,1),N_a,N_semiz*N_z*N_e,N_j]);
        N_dsemiz=size(pi_semiz_J,3);

        if simoptions.gridinterplayer==0
            Policy_aprime=zeros(N_a,N_semiz*N_z*N_e,1,N_j,'gpuArray');
        elseif simoptions.gridinterplayer==1
            Policy_aprime=zeros(N_a,N_semiz*N_z*N_e,2,N_j,'gpuArray'); % Policy_aprime has an additional dimension of length 2 which is the two points (and contains only the aprime indexes, no d indexes as would usually be the case).
        end

        Policy_dsemiexo=shiftdim(Policy(l_d-simoptions.l_dsemiz+1:l_d,:,:,:),1);
        if l_a==1
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:),[N_a,N_semiz*N_z*N_e,1,N_j]);
        elseif l_a==2
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1),[N_a,N_semiz*N_z*N_e,1,N_j]);
        elseif l_a==3
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1),[N_a,N_semiz*N_z*N_e,1,N_j]);
        elseif l_a==4
            Policy_aprime(:,:,1,:)=reshape(Policy(l_d+1,:,:,:)+n_a(1)*(Policy(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(Policy(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(Policy(l_d+4,:,:,:)-1),[N_a,N_semiz*N_z*N_e,1,N_j]);
        else
            error('CreatePTransitionMatrix_J cannot handle length(n_a)>4, contact me if you need this')
        end

        % Constructing Gammatranspose is done differently if using grid interpolation layers
        if simoptions.gridinterplayer==0
            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz*N_z*N_e,1,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz*N_z*N_e,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_z*N_e,1,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            zprimeoffset=repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort); % z' index following the z' dimension
            % semizindex_short_jj (built per age below) is [N_a*N_semiz*N_z*N_e,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z*N_e)',1,N_semizshort*N_z); %  Index for this period (a,z)

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemizzprime_jj=Policy_aprime(:,1,jj)+N_a*repmat((idxshort(semizindex_short_jj)-1),1,N_z)+zprimeoffset; % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
                semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

                % P: full transition matrix on (a,z)
                P=sparse(II2,Policy_aprimesemizzprime_jj,repmat(semiztransitions_jj,1,N_z).*repmat(repelem(pi_z_J(:,:,jj),N_a*N_semiz,N_semizshort),N_e,1),N_a*N_semiz*N_z*N_e,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end

        elseif simoptions.gridinterplayer==1
            N_probs=2;
            Policy_aprime(:,:,2,:)=Policy_aprime(:,:,1,:)+1; % upper index, add one to index for a1

            % First, get Gamma
            PolicyProbs=zeros(N_a,N_semiz*N_z*N_e,2,N_j,'gpuArray'); % PolicyProbs are the corresponding probabilities of each of these two.

            PolicyProbs(:,:,2,:)=reshape((Policy(end-1,:,:,:)-1)/(simoptions.ngridinterp+1),[N_a,N_semiz*N_z*N_e,1,N_j]); % probability of upper grid point (end-1 because end is now L2flag)
            PolicyProbs(:,:,1,:)=1-PolicyProbs(:,:,2,:); % probability of lower grid point

            N_semizshort=max(max(max(sum((pi_semiz_J>0),2))));
            % Create smaller version of pi_semiz_J that eliminates as many non-zeros as possible
            [pi_semiz_J_short, idx] = sort(pi_semiz_J,2); % puts all the zeros on the left of the matrix

            pi_semiz_J_short=pi_semiz_J_short(:,end-N_semizshort+1:end,:,:);
            idxshort=idx(:,end-N_semizshort+1:end,:,:);

            % Policy_aprime is currently [N_a,N_semiz*N_z*N_e,N_probs,N_j]
            Policy_dsemiexo=gather(reshape(Policy_dsemiexo,[N_a*N_semiz*N_z*N_e,1,N_j]));
            Policy_aprime=reshape(gather(Policy_aprime),[N_a*N_semiz*N_z*N_e,N_probs,N_j]);
            PolicyProbs=reshape(PolicyProbs,[N_a*N_semiz*N_z*N_e,N_probs,N_j]);
            pi_semiz_J_short=gather(pi_semiz_J_short);
            idxshort=gather(idxshort);
            semizindexbase=repmat(repelem((1:1:N_semiz)',N_a,1),N_z*N_e,1)+N_semiz*(0:1:N_semizshort-1); % age-independent part of semizindex_short
            zprimeoffset=repmat(repelem(N_a*N_semiz*(0:1:N_z-1),1,N_semizshort),1,N_probs); % z' index following the z' dimension
            % semizindex_short_jj (built per age below) is [N_a*N_semiz*N_z*N_e,N_semizshort], used to index pi_semiz_J_short and idxshort which are [N_semiz,N_semizshort,N_dsemiz,N_j-1]

            % Precompute
            II2=repmat((1:1:N_a*N_semiz*N_z*N_e)',1,N_semizshort*N_z*N_probs); %  Index for this period (a,z), note the N_probs-copies
            % Columns are ordered (semiz',z',prob): semiz' fastest, then z', then the two grid-interp points

            for jj=1:N_j-1
                semizindex_short_jj=semizindexbase+(N_semiz*N_semizshort)*(Policy_dsemiexo(:,1,jj)-1)+(N_semiz*N_semizshort*N_dsemiz)*(jj-1); % index for semiz, plus that for semiz' (in the semiz' dim), dsemiexo and j; their indexes in pi_semiz_J
                Policy_aprimesemizzprime_jj=repelem(Policy_aprime(:,:,jj),1,N_semizshort*N_z)+repmat(N_a*(idxshort(semizindex_short_jj)-1),1,N_z*N_probs)+zprimeoffset; % Note: add semiz' index following the semiz' dimension, add z' index following the z' dimension
                semiztransitions_jj=pi_semiz_J_short(semizindex_short_jj);

                PolicyProbs_jj=repelem(PolicyProbs(:,:,jj),1,N_semizshort*N_z).*repmat(semiztransitions_jj,1,N_z*N_probs).*repmat(repelem(pi_z_J(:,:,jj),N_a*N_semiz,N_semizshort),N_e,N_probs); % is of size [N_a*N_semiz*N_z*N_e,N_semizshort*N_z*N_probs]
                PolicyProbs_jj=gather(PolicyProbs_jj);

                % P: full transition matrix on (a,z)
                P=sparse(II2,Policy_aprimesemizzprime_jj,PolicyProbs_jj,N_a*N_semiz*N_z*N_e,N_a*N_semiz*N_z); % Note: sparse() will accumulate at repeated indices
                % Now put the pi_e shocks into next period
                P_cell{jj}=kron(sparse(gather(pi_e_J(:,jj+1)')),P); % note, reverse order
            end
        end

    end
end


end
