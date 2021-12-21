function StationaryDistKron=StationaryDist_FHorz_Case1_Iteration_raw(jequaloneDistKron,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions)
%Will treat the agents as being on a continuum of mass 1.

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
           
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
        for a_c=1:N_a
            for z_c=1:N_z
                if N_d==0 %length(n_d)==1 && n_d(1)==0
                    optaprime=PolicyIndexesKron(a_c,z_c,jj);
                else
                    optaprime=PolicyIndexesKron(2,a_c,z_c,jj);
                end
                for zprime_c=1:N_z
                    P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
                end
            end
        end
        P=reshape(P,[N_a*N_z,N_a*N_z]);
        P=P';
        
        StationaryDistKron(:,jj+1)=P*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
    
    StationaryDistKron=zeros(N_a*N_z,N_j,'gpuArray');
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
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
        
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime=reshape(PolicyIndexesKron(:,:,jj),[1,N_a*N_z]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_z]);
        end
        Ptran=zeros(N_a,N_a*N_z,'gpuArray');
        Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
        
        StationaryDistKron(:,jj+1)=Ptran*StationaryDistKron(:,jj);
    end
    
elseif simoptions.parallel>2 % Same as <2, but now using a sparse matrix instead of a standard matrix for P
    
    StationaryDistKron=sparse(N_a*N_z,N_j);
    StationaryDistKron(:,1)=jequaloneDistKron;
    
    for jj=1:(N_j-1)
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
        
        %First, generate the transition matrix P=g of Q (the convolution of the optimal policy function and the transition fn for exogenous shocks)
        Ptranspose=sparse(N_a*N_z,N_a*N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z) [ So Ptranspose(aprime,zprime,a,z)=proby of going to (a',z') given in (a,z)]
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            PolicyIndexesKron_jj=reshape(PolicyIndexesKron(:,:,jj),[N_a*N_z,1]);
        else
            PolicyIndexesKron_jj=reshape(PolicyIndexesKron(2,:,:,jj),[N_a*N_z,1]);
        end
                
%         whos N_a N_z StationaryDistKron PolicyIndexesKron_jj pi_z Ptranspose
        parfor az_c=1:N_a*N_z
            Ptranspose_az=sparse(N_a*N_z,1);
            optaprime=PolicyIndexesKron_jj(az_c);
            az_sub=ind2sub_homemade([N_a,N_z],az_c);
            pi_z_temp=pi_z(az_sub(2),:);
            sum_pi_z_temp=sum(pi_z_temp);
            for zprime_c=1:N_z
                optaprimezprime=sub2ind_homemade([N_a,N_z],[optaprime,zprime_c]);
                Ptranspose_az(optaprimezprime)=pi_z_temp(zprime_c)/sum_pi_z_temp;
            end
            Ptranspose(:,az_c)=Ptranspose_az;
        end

        StationaryDistKron(:,jj+1)=Ptranspose*StationaryDistKron(:,jj);
    end
    StationaryDistKron=full(StationaryDistKron);
    
    if simoptions.parallel==4
        StationaryDistKron=gpuArray(StationaryDistKron);
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
StationaryDistKron=StationaryDistKron.*(ones(N_a*N_z,1)*AgeWeights);

end
