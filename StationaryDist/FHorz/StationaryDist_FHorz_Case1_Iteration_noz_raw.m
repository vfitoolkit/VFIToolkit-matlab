function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_noz_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_j,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

if N_d==0
    optaprime=reshape(PolicyIndexesKron,[1,N_a,N_j]);
else
    optaprime=reshape(PolicyIndexesKron(2,:,:),[1,N_a,N_j]);
end

optaprime=gather(optaprime);
jequaloneDistKron=gather(jequaloneDistKron);

StationaryDistKron=zeros(N_a,N_j);
StationaryDistKron(:,1)=jequaloneDistKron;

StationaryDistKron_jj=sparse(jequaloneDistKron);

for jj=1:(N_j-1)

    if simoptions.verbose==1
        fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
    end

    Gammatranspose=sparse(optaprime(1,:,jj),1:1:N_a,ones(N_a,1),N_a,N_a);

    StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;
    StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
end

if simoptions.parallel==2 % Move the solution to the gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
end

%% THIS COMMENTED OUT SECTION IS LEGACY CODE
% Tried out doing this using cpu matrices, gpu matrices, sparse cpu
% matrices, and sparce gpu matrices.
% The sparse cpu matrices seem to be the fastest, and since they also use
% least memory I have hardcoded them above.


% if simoptions.parallel<2 || simoptions.parallel==3
%     optaprime=gather(optaprime);
%     jequaloneDistKron=gather(jequaloneDistKron);
% end
% 
% if simoptions.parallel<2 % Full CPU
% 
%     StationaryDistKron=zeros(N_a,N_j);
%     StationaryDistKron(:,1)=jequaloneDistKron;
% 
%     for jj=1:(N_j-1)
% 
%         if simoptions.verbose==1
%             fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
%         end
% 
%         optaprime_jj=optaprime(1,:,jj);
% 
%         Gammatranspose=zeros(N_a,N_a); %probability of going to (a') given in (a)
%         Gammatranspose(optaprime_jj+N_a*(0:1:N_a-1))=1;
% 
%         StationaryDistKron(:,jj+1)=Gammatranspose*StationaryDistKron(:,jj);
%     end
% 
% elseif simoptions.parallel==2 % Full GPU
% 
%     StationaryDistKron=zeros(N_a,N_j,'gpuArray');
%     StationaryDistKron(:,1)=jequaloneDistKron;
% 
%     for jj=1:(N_j-1)
% 
%         if simoptions.verbose==1
%             fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
%         end
% 
%         optaprime_jj=optaprime(1,:,jj);
% 
%         Gammatranspose=zeros(N_a,N_a,'gpuArray');
%         Gammatranspose(optaprime_jj+N_a*(gpuArray(0:1:N_a-1)))=1;
% 
%         StationaryDistKron(:,jj+1)=Gammatranspose*StationaryDistKron(:,jj);
%     end
% 
% elseif simoptions.parallel==3 % Sparse CPU
% 
%     StationaryDistKron=zeros(N_a,N_j);
%     StationaryDistKron(:,1)=jequaloneDistKron;
% 
%     StationaryDistKron_jj=sparse(jequaloneDistKron);
% 
%     for jj=1:(N_j-1)
% 
%         if simoptions.verbose==1
%             fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
%         end
% 
%         optaprime_jj=optaprime(1,:,jj);
% 
%         Gammatranspose=sparse(N_a,N_a);
%         Gammatranspose(optaprime_jj+N_a*(0:1:N_a-1))=1;
% 
%         StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;
%         StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
%     end
% 
%     if gpuDeviceCount>0 % Move the solution to the gpu if there is one
%         StationaryDistKron=gpuArray(StationaryDistKron);
%     end
% 
% elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on gpu
% 
%     StationaryDistKron=zeros(N_a,N_j,'gpuArray');
%     StationaryDistKron(:,1)=jequaloneDistKron;
% 
%     StationaryDistKron_jj=sparse(StationaryDistKron(:,1));
%     for jj=1:(N_j-1)
% 
%         if simoptions.verbose==1
%             fprintf('Stationary Distribution iteration horizon: %i of %i \n',jj, N_j)
%         end
% 
%         optaprime_jj=optaprime(1,:,jj);
% 
%         Gammatranspose=sparse(N_a,N_a);
%         Gammatranspose(optaprime_jj+N_a*(0:1:N_a-1))=1;
% 
%         Gammatranspose=gpuArray(Gammatranspose);
% 
% %         StationaryDistKron(:,jj+1)=Gammatranspose*StationaryDistKron(:,jj); % Cannot index sparse gpuArray, so have to use StationaryDistKron_jj instead
%         StationaryDistKron_jj=Gammatranspose*StationaryDistKron_jj;
%         StationaryDistKron(:,jj+1)=full(StationaryDistKron_jj);
%     end
% 
% end

%%

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
