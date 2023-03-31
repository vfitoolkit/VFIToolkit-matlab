function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_noz_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_j,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        Ptranspose=zeros(N_a,N_a); %P(a,aprime)=proby of going to (a') given in (a)
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime_jj=PolicyIndexesKron(:,jj)'; % Note transpose
        else
            optaprime_jj=PolicyIndexesKron(2,:,jj);
        end
        Ptranspose(optaprime_jj+N_a*(0:1:N_a-1))=1;
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end
        
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime_jj=reshape(PolicyIndexesKron(:,jj),[1,N_a]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,jj),[1,N_a]);
        end
        Ptranspose=zeros(N_a,N_a,'gpuArray');
        Ptranspose(optaprime_jj+N_a*(gpuArray(0:1:N_a-1)))=1;
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
   
    StationaryDistKron=sparse(N_a,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime_jj=reshape(PolicyIndexesKron(:,jj),[1,N_a]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,jj),[1,N_a]);
        end
        Ptranspose=sparse(N_a,N_a);
        Ptranspose(optaprime_jj+N_a*(0:1:N_a-1))=1;

        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse? (Sparse gpu is very limited functionality, so since we return the gpuArray we want to change to full)

    if gpuDeviceCount>0 % Move the solution to the gpu if there is one
        StationaryDistKron=gpuArray(StationaryDistKron);
    end
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on cpu
    % StationaryDistKron is a gpuArray
    % StationaryDistKron_jj and Ptranspose are treated as sparse gpu arrays.    
    
    StationaryDistKron=zeros(N_a,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    StationaryDistKron_jj=sparse(StationaryDistKron(:,1));
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
        end
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime_jj=reshape(PolicyIndexesKron(:,jj),[1,N_a]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,jj),[1,N_a]);
        end
        Ptranspose=sparse(N_a,N_a);
        Ptranspose(optaprime_jj+N_a*(0:1:N_a-1))=1;
        
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
