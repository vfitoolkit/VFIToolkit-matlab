function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_e,N_j,pi_z,pi_e,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
           
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        P=zeros(N_a,N_z,N_e,N_a,N_z); %P(a,ze,,aprime,zprime)=proby of going to (a',z') given in (a,z,e)
        for a_c=1:N_a
            for z_c=1:N_z
                for e_c=1:N_e
                    if N_d==0 %length(n_d)==1 && n_d(1)==0
                        optaprime=PolicyIndexesKron(a_c,z_c,e_c,jj);
                    else
                        optaprime=PolicyIndexesKron(2,a_c,z_c,e_c,jj);
                    end
                    for zprime_c=1:N_z
                        P(a_c,z_c,e_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                    end
                end
            end
        end
        P=reshape(P,[N_a*N_z*N_e,N_a*N_z]);
        P=P';
        
        StationaryDistKron(:,jj+1)=kron(pi_e,(P*StationaryDistKron(:,jj)));
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_z*N_e,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime=reshape(PolicyIndexesKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_z*N_e]);
        end
        Ptran=zeros(N_a,N_a*N_z*N_e,'gpuArray'); % Start with P (a,z,e) to a' (Note, create P')
        Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_z*N_e-1)))=1; % Fill in the a' transitions based on Policy
        
        % Now use pi_z to switch to P (a,z,e) to (a',z')
        Ptran=(kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a,'gpuArray')))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
                
        StationaryDistKron(:,jj+1)=kron(pi_e, Ptran*StationaryDistKron(:,jj) );
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        

        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime_jj=reshape(PolicyIndexesKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_z*N_e]);
        end
        PtransposeA=sparse(N_a,N_a*N_z*N_e);  % Start with P (a,z,e) to a' (Note, create P')
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z*N_e-1))=1; % Fill in the a' transitions based on Policy
        
        
        pi_z=sparse(pi_z);
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            pi_z=gather(pi_z); % The indexing used can only be done on cpu
            Ptranspose=kron(ones(N_z*N_e,1),PtransposeA);
            for ii=1:N_z
                % NOT SURE ABOUT THIS LINE!!??
                Ptranspose(:,(1:1:N_a)+N_a*(ii-1)+N_a*N_z*((1:1:N_e)-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)+N_a*N_z*((1:1:N_e)-1)).*kron(ones(N_e,N_e),kron(pi_z(ii,:)',ones(N_a,N_a)));
            end
        end

        StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj));
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse?
    % Move the solution to the gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_z*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime_jj=reshape(PolicyIndexesKron(:,:,:,jj),[1,N_a*N_z*N_e]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,:,:,jj),[1,N_a*N_z*N_e]);
        end
        PtransposeA=sparse(N_a,N_a*N_z*N_e);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z*N_e-1))=1;
        
        pi_z=sparse(pi_z);
        try % Following formula only works if pi_z is already sparse, otherwise kron(pi_z',ones(N_a,N_a)) is not sparse.
            Ptranspose=kron(ones(1,N_e),kron(pi_z',ones(N_a,N_a))).*kron(ones(N_z,1),PtransposeA);
        catch % Otherwise do something slower but which is sparse regardless of whether pi_z is sparse
            pi_z=gather(pi_z); % The indexing used can only be donoe on cpu
            Ptranspose=kron(ones(N_z,1),PtransposeA);
            for ii=1:N_z
                for jj=1:N_e
                    % NOT SURE ABOUT THIS LINE!!??
                    Ptranspose(:,(1:1:N_a)+N_a*(ii-1)+N_a*N_z*(jj-1))=Ptranspose(:,(1:1:N_a)+N_a*(ii-1)+N_a*N_z*(jj-1)).*kron(pi_z(ii,:)',ones(N_a,N_a));
                end
            end
    
        end
        
        Ptranspose=gpuArray(Ptranspose);
        pi_z=gpuArray(pi_z);
        
        try
            StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj));
        catch
            error('The transition matrix is big, please use simoptions.parallel=3 (instead of 4) \n')
        end
    end
    StationaryDistKron=full(StationaryDistKron);
    
    if simoptions.parallel==4 % Move solution to gpu
        StationaryDistKron=gpuArray(StationaryDistKron);
    end
    
elseif simoptions.parallel==5 % Same as 2, except loops over e
    
    StationaryDistKron=zeros(N_a*N_z,N_e,N_j,'gpuArray');
    StationaryDistKron(:,:,1)=reshape(jequaloneDistKron,[N_a*N_z,N_e]);
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        % Take advantage of fact that for each (a,z,e) we can map into
        % (a',z'), looping over e
        StatDisttemp=zeros(N_a*N_z,1,'gpuArray');
        for e_c=1:N_e
            % Following is essentially just the code for simoptions.parallel=2 where there is no e
            % Because conditional on e, we are looking for transitions
            % from (a,z) to (a',z'). We can add these up across the e, and
            % then once finish the for-loop we distribute them across e'
            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime=reshape(PolicyIndexesKron(:,:,e_c,jj),[1,N_a*N_z]);
            else
                optaprime=reshape(PolicyIndexesKron(2,:,:,e_c,jj),[1,N_a*N_z]);
            end
            Ptran=zeros(N_a,N_a*N_z,'gpuArray');
            Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
            Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
        
            StatDisttemp=StatDisttemp+Ptran*StationaryDistKron(:,e_c,jj);
        end
        % And now just distribute over the e'
        StationaryDistKron(:,:,jj+1)=StatDisttemp.*pi_e';

    end
    
elseif simoptions.parallel==6 % Same as 4, except loops over e
    
    StationaryDistKron=zeros(N_a*N_z,N_e,N_j,'gpuArray');
    StationaryDistKron(:,:,1)=reshape(jequaloneDistKron,[N_a*N_z,N_e]);
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        if fieldexists_pi_e_J==1
            pi_e=simoptions.pi_e_J(:,jj);
        elseif fieldexists_EiidShockFn==1
            if fieldexists_EiidShockFnParamNames==1
                EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                for ii=1:length(EiidShockFnParamsVec)
                    EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                end
                [~,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
            else
                [~,pi_e]=simoptions.EiidShockFn(jj);
            end
        end
        
        % Take advantage of fact that for each (a,z,e) we can map into (a',z'), looping over e
        StatDisttemp=sparse(N_a*N_z,1);
        for e_c=1:N_e
            %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
            if N_d==0 %length(n_d)==1 && n_d(1)==0
                optaprime_jj=reshape(PolicyIndexesKron(:,:,e_c,jj),[1,N_a*N_z]);
            else
                optaprime_jj=reshape(PolicyIndexesKron(2,:,:,e_c,jj),[1,N_a*N_z]);
            end
            PtransposeA=sparse(N_a,N_a*N_z);
            PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_z-1))=1;
            
            pi_z=sparse(pi_z);
            Ptranspose=kron(pi_z',ones(N_a,N_a)).*kron(ones(N_z,1),PtransposeA);
            
            StatDisttemp=StatDisttemp+Ptranspose*StationaryDistKron(:,e_c,jj);
        end
        % And now just distribute over the e'
        StationaryDistKron(:,:,jj+1)=full(gpuArray(StatDisttemp.*pi_e'));

    end
    
end

% Reweight the different ages based on 'AgeWeightParamNames'. (it is
% assumed there is only one Age Weight Parameter (name))
FullParamNames=fieldnames(Parameters);
nFields=length(FullParamNames);
found=0;
for iField=1:nFields
    if strcmp(AgeWeightParamNames{1},FullParamNames{iField})
        AgeWeights=Parameters.(FullParamNames{iField});
        found=1;
    end
end
if found==0 % Have added this check so that user can see if they are missing a parameter
    fprintf(['FAILED TO FIND PARAMETER ',AgeWeightParamNames{1}])
end
% I assume AgeWeights is a row vector
if simoptions.parallel==5
    StationaryDistKron=StationaryDistKron.*repmat(shiftdim(AgeWeights,-1),N_a*N_z,N_e,1);
else
    StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z*N_e,1)*AgeWeights);    
end

end
