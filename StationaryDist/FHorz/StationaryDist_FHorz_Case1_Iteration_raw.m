function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z_J,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

if simoptions.tanimprovement==1
    StationaryDistKron=zeros(N_a*N_z,N_j);
    StationaryDistKron(:,1)=gather(jequaloneDistKron);

    StationaryDist_jj=sparse(gather(jequaloneDistKron));

    if N_d==0
        PolicyIndexesKron=gather(reshape(PolicyIndexesKron,[1,N_a*N_z,N_j]));
    else
        PolicyIndexesKron=gather(reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_z,N_j]));
    end
    
    for jj=1:(N_j-1)
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        optaprime=PolicyIndexesKron(1,:,jj);

        firststep=optaprime+kron(N_a*(0:1:N_z-1),ones(1,N_a));
        Gammatranspose=sparse(firststep,1:1:N_a*N_z,ones(N_a*N_z,1),N_a*N_z,N_a*N_z);

        pi_z=sparse(gather(pi_z_J(:,:,jj))); % Note: this cannot be moved outside the for-loop as Matlab only allows sparse for 2-D arrays (so cannot, e.g., do sparse(pi_z_J)).

        % Two steps of the Tan improvement
        StationaryDist_jj=reshape(Gammatranspose*StationaryDist_jj,[N_a,N_z]); %No point checking distance every single iteration. Do 100, then check.
        StationaryDist_jj=reshape(StationaryDist_jj*pi_z,[N_a*N_z,1]);

        StationaryDistKron(:,jj+1)=full(StationaryDist_jj);
    end
    if simoptions.parallel==2 % Move result to gpu
        StationaryDistKron=gpuArray(StationaryDistKron);
        % Note: sparse gpu matrices do exist in matlab, but cannot index nor reshape() them. So cannot do Tan improvement with them.
    end

%% EVERYTHING AFTER THIS IS REALLY JUST LEGACY CODE. IS JUST LEFT HERE FOR RUNTIME DEMOS.
% It is how iteration used to be done before the Tan improvement was
% implemented. It is same accuracy, but slower and more memory intensive.
% It does not create sparse matrices as well as I now know how to so that
% slows it down even more.
elseif simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    if N_d==0
        PolicyIndexesKron=reshape(PolicyIndexesKron,[1,N_a*N_z,N_j]);
    else
        PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_z,N_j]);
    end

    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        optaprime=PolicyIndexesKron(1,:,jj);

%         %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
%         P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%         for a_c=1:N_a
%             for z_c=1:N_z
%                 if N_d==0 %length(n_d)==1 && n_d(1)==0
%                     optaprime=PolicyIndexesKron(a_c,z_c,jj);
%                 else
%                     optaprime=PolicyIndexesKron(2,a_c,z_c,jj);
%                 end
%                 for zprime_c=1:N_z
%                     P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%                 end
%             end
%         end
%         P=reshape(P,[N_a*N_z,N_a*N_z]);
%         P=P';
        
        Ptranspose=zeros(N_a,N_a*N_z);
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptranspose=(kron(pi_z_J(:,:,jj)',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptranspose));
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    if N_d==0
        PolicyIndexesKron=reshape(PolicyIndexesKron,[1,N_a*N_z,N_j]);
    else
        PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_z,N_j]);
    end

    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        optaprime=PolicyIndexesKron(1,:,jj);
        
        Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptranspose=(kron(pi_z_J(:,:,jj)',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptranspose));
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
   
    StationaryDistKron=sparse(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;

    if N_d==0
        PolicyIndexesKron=reshape(PolicyIndexesKron,[1,N_a*N_z,N_j]);
    else
        PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_z,N_j]);
    end

    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        optaprime=PolicyIndexesKron(1,:,jj);

        PtransposeA=sparse(N_a,N_a*N_z);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=1;
        
        pi_z=sparse(pi_z_J(:,:,jj));
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(ones(N_z,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            pi_z=gather(pi_z); % The indexing used can only be done on cpu
            Ptranspose=kron(ones(N_z,1),PtransposeA);
            for ii=1:N_z
                Ptranspose(:,(1:1:N_a)+N_a*(ii-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)).*kron(pi_z(ii,:)',ones(N_a,N_a));
            end
        end

        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse? (Sparse gpu is very limited functionality, so since we return the gpuArray we want to change to full)

    if gpuDeviceCount>0 % Move the solution to the gpu if there is one
        StationaryDistKron=gpuArray(StationaryDistKron);
    end
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on gpu
    % StationaryDistKron is a gpuArray
    % StationaryDistKron_jj and Ptranspose are treated as sparse gpu arrays.    
    
    StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    StationaryDistKron_jj=sparse(StationaryDistKron(:,1));
    
    if N_d==0
        PolicyIndexesKron=reshape(PolicyIndexesKron,[1,N_a*N_z,N_j]);
    else
        PolicyIndexesKron=reshape(PolicyIndexesKron(2,:,:,:),[1,N_a*N_z,N_j]);
    end

    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        optaprime=PolicyIndexesKron(1,:,jj);

        PtransposeA=sparse(N_a,N_a*N_z);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=1;
        
        pi_z=sparse(pi_z_J(:,:,jj));
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(ones(N_z,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            pi_z=gather(pi_z); % The indexing used can only be donoe on cpu
            Ptranspose=kron(ones(N_z,1),PtransposeA);
            for ii=1:N_z
                Ptranspose(:,(1:1:N_a)+N_a*(ii-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)).*kron(pi_z(ii,:)',ones(N_a,N_a));
            end
        end
        
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
