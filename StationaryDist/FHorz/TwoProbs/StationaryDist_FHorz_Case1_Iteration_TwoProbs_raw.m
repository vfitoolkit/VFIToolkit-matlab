function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_TwoProbs_raw(jequaloneDistKron,AgeWeightParamNames,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z,Parameters,simoptions)
% 'TwoProbs' refers to two probabilities.
% Policy_aprime has an additional final dimension of length 2 which is
% the two points (and contains only the aprime indexes, no d indexes as would usually be the case). 
% PolicyProbs are the corresponding probabilities of each of these two.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel


eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
            end
        end
        
        Ptranspose=zeros(N_a,N_a*N_z);
        % Lower grid point
        optaprime=reshape(Policy_aprime(:,:,jj,1),[1,N_a*N_z]);
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=PolicyProbs(:,:,jj,1);
        % Repeat for upper grid point
        optaprime=reshape(Policy_aprime(:,:,jj,2),[1,N_a*N_z]);
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=PolicyProbs(:,:,jj,2);
        % Note: the lower and upper grid points are mutually exclusive (conditional on a,z). So don't need to worry about overwriting.

        Ptranspose=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptranspose));
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end
        
        Ptranspose=zeros(N_a,N_a*N_z,'gpuArray');

        % Lower grid point
        optaprime=reshape(Policy_aprime(:,:,jj,1),[1,N_a*N_z]);
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=PolicyProbs(:,:,jj,1);
        % Repeat for upper grid point
        optaprime=reshape(Policy_aprime(:,:,jj,2),[1,N_a*N_z]);
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=PolicyProbs(:,:,jj,2);
        % Note: the lower and upper grid points are mutually exclusive (conditional on a,z). So don't need to worry about overwriting.

        Ptranspose=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptranspose));
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
   
    StationaryDistKron=sparse(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end
        
        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
            end
        end

        PtransposeA=sparse(N_a,N_a*N_z);

        % Lower grid point
        optaprime_jj=reshape(Policy_aprime(:,:,jj,1),[1,N_a*N_z]);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=PolicyProbs(:,:,jj,1);
        % Repeat for upper grid point
        optaprime_jj=reshape(Policy_aprime(:,:,jj,2),[1,N_a*N_z]);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=PolicyProbs(:,:,jj,2);
        % Note: the lower and upper grid points are mutually exclusive (conditional on a,z). So don't need to worry about overwriting.

        pi_z=sparse(pi_z);
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
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        if fieldexists_pi_z_J==1
            pi_z=simoptions.pi_z_J(:,:,jj);
        elseif fieldexists_ExogShockFn==1
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [~,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
            end
        end
        
        PtransposeA=sparse(N_a,N_a*N_z);

        % Lower grid point
        optaprime_jj=reshape(Policy_aprime(:,:,jj,1),[1,N_a*N_z]);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=PolicyProbs(:,:,jj,1);
        % Repeat for upper grid point
        optaprime_jj=reshape(Policy_aprime(:,:,jj,2),[1,N_a*N_z]);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=PolicyProbs(:,:,jj,2);
        % Note: the lower and upper grid points are mutually exclusive (conditional on a,z). So don't need to worry about overwriting.

        pi_z=sparse(pi_z);
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
        pi_z=gpuArray(pi_z);
        
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
