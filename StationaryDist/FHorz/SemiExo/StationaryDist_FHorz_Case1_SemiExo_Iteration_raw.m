function StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d1,n_d2,N_a,N_z,N_semiz,N_j,pi_z_J,pi_semiz_J,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

% Gammatranspose=sparse(firststep,1:1:N_a*N_bothz,ones(N_a*N_bothz,1),N_a*N_bothz,N_a*N_bothz);


N_bothz=N_z*N_semiz;

optdprime=reshape(PolicyIndexesKron(1,:,:,:),[N_a*N_bothz,N_j]); % Note: column vector (conditional on jj)
optaprime=reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_bothz,N_j]); % Note: row vector (conditional on jj)
if simoptions.parallel<2 || simoptions.parallel==3
    optaprime=gather(optaprime);
    optdprime=gather(optdprime);
end


if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_bothz,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime(:,jj));
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        Ptranspose=zeros(N_a,N_a*N_bothz);
        Ptranspose(optaprime(1,:,jj)+N_a*(gpuArray(0:1:N_a*N_bothz-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z_J(:,:,jj)'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(ones(N_bothz,1),Ptranspose); % Note: without semiz, normally semiztransitions is just kron(pi_z',ones(N_a,N_a))
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_bothz,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end
        
        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime(:,jj));
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        Ptranspose=zeros(N_a,N_a*N_bothz,'gpuArray');
        Ptranspose(optaprime(1,:,jj)+N_a*(gpuArray(0:1:N_a*N_bothz-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz,'gpuArray'),pi_z_J(:,:,jj)'),ones(N_a,N_a,'gpuArray')).*kron(semiztransitions',ones(N_a*N_z,1,'gpuArray')).*kron(ones(N_bothz,1,'gpuArray'),Ptranspose);

        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
   
    StationaryDistKron=zeros(N_a*N_bothz,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    StationaryDistKron_jj=sparse(jequaloneDistKron);
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime(:,jj));
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        Gammatranspose=sparse(optaprime(1,:,jj),1:1:N_a*N_bothz,ones(N_a*N_bothz,1),N_a,N_a*N_bothz);

        pi_z=sparse(pi_z_J(:,:,jj));
        semiztransitions=sparse(semiztransitions);
        
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(ones(N_bothz,1),Gammatranspose);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            semiztransitions=gather(semiztransitions); % The indexing used can only be done on cpu
            Ptranspose=kron(ones(N_bothz,1),Gammatranspose);
            for ii=1:N_bothz
                Ptranspose(:,(1:1:N_a)+N_a*(ii-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(kron(ones(N_semiz,N_semiz),pi_z_J'),ones(N_a,N_a));
            end
        end


        StationaryDistKron_jj=Ptranspose*StationaryDistKron_jj;

        StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse? (Sparse gpu is very limited functionality, so since we return the gpuArray we want to change to full)

    if gpuDeviceCount>0 % Move the solution to the gpu if there is one
        StationaryDistKron=gpuArray(StationaryDistKron);
    end
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on cpu
    % StationaryDistKron is a gpuArray
    % StationaryDistKron_jj and Ptranspose are treated as sparse gpu arrays.    
    
    StationaryDistKron=zeros(N_a*N_bothz,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    StationaryDistKron_jj=sparse(jequaloneDistKron);
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime(:,jj));
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        Gammatranspose=sparse(gather(optaprime(1,:,jj)),1:1:N_a*N_bothz,ones(N_a*N_bothz,1),N_a,N_a*N_bothz);
        
        pi_z=sparse(pi_z_J(:,:,jj));
        semiztransitions=sparse(semiztransitions);

        Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z,1)).*repmat(Gammatranspose,N_bothz,1);

        Ptranspose=gpuArray(Ptranspose);
        
%         StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj); % Cannot index sparse gpuArray, so have to use StationaryDistKron_jj instead
        StationaryDistKron_jj=Ptranspose*StationaryDistKron_jj;
        StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
    end
    
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

StationaryDistKron=StationaryDistKron.*AgeWeights;

end
