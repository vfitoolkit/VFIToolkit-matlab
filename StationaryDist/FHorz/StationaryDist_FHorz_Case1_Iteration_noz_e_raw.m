function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_noz_e_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_e,N_j,pi_e,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel

eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')

if simoptions.parallel<2
    
    StationaryDistKron=zeros(N_a*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
        P=zeros(N_a,N_e,N_a); %P(a,ze,,aprime,zprime)=proby of going to (a',z') given in (a,z,e)
        for a_c=1:N_a
            for e_c=1:N_e
                if N_d==0 %length(n_d)==1 && n_d(1)==0
                    optaprime=PolicyIndexesKron(a_c,e_c,jj);
                else
                    optaprime=PolicyIndexesKron(2,a_c,e_c,jj);
                end
            end
        end
        P=reshape(P,[N_a*N_e,N_a]);
        P=P';
        
        StationaryDistKron(:,jj+1)=kron(pi_e,(P*StationaryDistKron(:,jj)));
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_e,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
            optaprime=reshape(PolicyIndexesKron(:,:,jj),[1,N_a*N_e]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_e]);
        end
        Ptran=zeros(N_a,N_a*N_e,'gpuArray'); % Start with P (a,e) to a' (Note, create P')
        Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_e-1)))=1; % Fill in the a' transitions based on Policy
        
        StationaryDistKron(:,jj+1)=kron(pi_e, Ptran*StationaryDistKron(:,jj) );
    end
    
elseif simoptions.parallel==3 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
            optaprime_jj=reshape(PolicyIndexesKron(:,:,jj),[1,N_a*N_e]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_e]);
        end
        PtransposeA=sparse(N_a,N_a*N_e);  % Start with P (a,z,e) to a' (Note, create P')
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_e-1))=1; % Fill in the a' transitions based on Policy
        
        StationaryDistKron(:,jj+1)=kron(pi_e, Ptranspose*StationaryDistKron(:,jj));
    end
    
    StationaryDistKron=full(StationaryDistKron); % Why do I do this? Why not just leave it sparse?
    % Move the solution to the gpu
    StationaryDistKron=gpuArray(StationaryDistKron);
    
elseif simoptions.parallel==4 % Sparse matrix instead of a standard matrix for P, on cpu
    
    StationaryDistKron=sparse(N_a*N_e,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
        
        if simoptions.verbose==1
            fprintf('Stationary Distribution iteration horizon: %i of %i (counting backwards to 1) \n',jj, N_j)
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
            optaprime_jj=reshape(PolicyIndexesKron(:,:,jj),[1,N_a*N_e]);
        else
            optaprime_jj=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_e]);
        end
        PtransposeA=sparse(N_a,N_a*N_e);
        PtransposeA(optaprime_jj+N_a*(0:1:N_a*N_e-1))=1;
        
        Ptranspose=gpuArray(Ptranspose);
        
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
end

% Reweight the different ages based on 'AgeWeightParamNames'. (it is assumed there is only one Age Weight Parameter (name))
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
if simoptions.parallel==5 || simoptions.parallel==6 
    StationaryDistKron=StationaryDistKron.*repmat(shiftdim(AgeWeights,-1),N_a,N_e,1);
else
    StationaryDistKron=StationaryDistKron.*(ones(N_a*N_e,1)*AgeWeights);    
end

end
