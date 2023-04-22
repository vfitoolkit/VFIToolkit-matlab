function StationaryDistKron=StationaryDist_FHorz_Case1_SemiExo_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,n_d1,n_d2,N_a,N_z,N_semiz,N_j,pi_z,pi_semiz_J,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

N_bothz=N_z*N_semiz;

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_bothz,N_j);
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
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_bothz,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_bothz]);
        Ptranspose=zeros(N_a,N_a*N_bothz);
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(ones(N_bothz,1),Ptranspose); % Note: without semiz, normally semiztransitions is just kron(pi_z',ones(N_a,N_a))
        
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
        
        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_bothz,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_bothz]);
        Ptranspose=zeros(N_a,N_a*N_bothz,'gpuArray');
        Ptranspose(optaprime+N_a*(gpuArray(0:1:N_a*N_bothz-1)))=1;
        Ptranspose=kron(kron(ones(N_semiz,N_semiz,'gpuArray'),pi_z'),ones(N_a,N_a,'gpuArray')).*kron(semiztransitions',ones(N_a*N_z,1,'gpuArray')).*kron(ones(N_bothz,1,'gpuArray'),Ptranspose);
        
        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
   
    StationaryDistKron=sparse(N_a*N_bothz,N_j);
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
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_bothz,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        optaprime_jj=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_bothz]);
        PtransposeA=sparse(N_a,N_a*N_bothz);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_bothz-1))=1;
        
        pi_z=sparse(pi_z);
        semiztransitions=sparse(semiztransitions);
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(ones(N_bothz,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            semiztransitions=gather(semiztransitions); % The indexing used can only be done on cpu
            Ptranspose=kron(ones(N_bothz,1),PtransposeA);
            for ii=1:N_bothz
                Ptranspose(:,(1:1:N_a)+N_a*(ii-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)).*kron(semiztransitions',ones(N_a*N_z,1)).*kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a));
            end
        end

        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
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
                pi_z=gpuArray(pi_z);
            else
                [~,pi_z]=simoptions.ExogShockFn(jj);
                pi_z=gpuArray(pi_z);
            end
        end

        pi_semiz_jj=pi_semiz_J(:,:,:,jj);

        % z transitions based on semiz
        optdprime=reshape(PolicyIndexesKron(1,:,:,jj),[N_a*N_bothz,1]); % Note has to be column vector to use in next line
        dsub=ind2sub_vec_homemade([n_d1,n_d2],optdprime);
        d2_c=dsub(:,end); % This is the decision variable that is determining the transition probabilities for the semi-exogenous state
        % Get the right part of pi_semiz_J % d2 depends on (a,z,semiz), and
        % pi_semiz is going to be about (semiz,semiz'), so I need to put it all
        % together as (a,z,semiz,semiz').
        semizindexcorrespondingtod2_c=kron((1:1:N_semiz)',ones(N_a*N_z,1));
        fullindex=semizindexcorrespondingtod2_c+N_semiz*((1:1:N_semiz)-1)+(N_semiz*N_semiz)*(d2_c-1);
        semiztransitions=pi_semiz_jj(fullindex); % (a,z,semiz,semiz')

        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        optaprime_jj=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_bothz]);
        PtransposeA=sparse(N_a,N_a*N_bothz);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_bothz-1))=1;
        
        pi_z=sparse(pi_z);
        semiztransitions=sparse(semiztransitions);
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(kron(ones(N_semiz,N_semiz),pi_z'),ones(N_a,N_a)).*kron(semiztransitions',ones(N_a*N_semiz,1)).*kron(ones(N_bothz,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            % semiztransitions=gather(semiztransitions); % The indexing used can only be donoe on cpu
            % Ptranspose=kron(ones(N_bothz,1),PtransposeA);
            % for ii=1:N_bothz
            %     Ptranspose(:,(1:1:N_a)+N_a*(ii-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)).*% kron(pi_z(ii,:)',ones(N_a,N_a));
            % end
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
