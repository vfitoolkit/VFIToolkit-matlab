function AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeightParamNames,PolicyIndexesKron,N_d,N_a,N_z,N_j,pi_z,Parameters,simoptions)
% Will treat the agents as being on a continuum of mass 1.

% Options needed
%  simoptions.maxit
%  simoptions.tolerance
%  simoptions.parallel


% This implementation is slightly inefficient when shocks are not age dependent, but speed loss is fairly trivial
eval('fieldexists_pi_z_J=1;vfoptions.pi_z_J;','fieldexists_pi_z_J=0;')
eval('fieldexists_ExogShockFn=1;vfoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;vfoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
if fieldexists_pi_z_J==0 && fieldexists_ExogShockFn==0
    pi_z_J=pi_z.*ones(1,1,N_j);
elseif fieldexists_pi_z_J==1
    pi_z_J=vfoptions.pi_z_J;
elseif fieldexists_ExogShockFn==1
    pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
    for jj=1:N_j
        if fieldexists_ExogShockFnParamNames==1
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ii=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
            end
            [~,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
            pi_z_J(:,jj)=gpuArray(pi_z);
        else
            [~,pi_z]=vfoptions.ExogShockFn(jj);
            pi_z_J(:,:,jj)=gpuArray(pi_z);
        end
    end
end

if simoptions.parallel~=2
    
    AgentDist=reshape(AgentDist,[N_a*N_z,N_j]);
%   Implicitly: AgentDist(:,1)=AgentDist(:,1); % Assume that endowment characteristics of newborngeneration are not changing over the transition path.
    
    % Remove the existing age weights, then impose the new age weights at the end 
    % (note, this is unnecessary overhead when the age weights are unchanged, but can't be bothered doing a clearer version)
%     size((ones(N_a*N_z,1)*sum(AgentDist,1)))
    AgentDist=AgentDist./(ones(N_a*N_z,1)*sum(AgentDist,1)); % Note: sum(AgentDist,1) are the current age weights

    for jj=1:(N_j-1)
        pi_z=pi_z_J(:,:,jj);
        
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
        
        AgentDist(:,jj+1)=P*AgentDist(:,jj);
    end
    
elseif simoptions.parallel==2 % Using the GPU
        
    % Remove the existing age weights, then impose the new age weights at the end 
    % (note, this is unnecessary overhead when the age weights are unchanged, but can't be bothered doing a clearer version)
    AgentDist=AgentDist./(ones(N_a*N_z,1)*sum(AgentDist,1)); % Note: sum(AgentDist,1) are the current age weights
    
%     AgentDist=zeros(N_a*N_z,N_j,'gpuArray');
%     AgentDist(:,1)=AgentDist; % Assume that endowment characteristics of newborn
%     generation are not changing over the transition path.
    
    % First, generate the transition matrix P=g of Q (the convolution of the 
    % optimal policy function and the transition fn for exogenous shocks)
    for jj=1:(N_j-1)
        pi_z=pi_z_J(:,:,jj);
        
        if N_d==0 %length(n_d)==1 && n_d(1)==0
            optaprime=reshape(PolicyIndexesKron(:,:,jj),[1,N_a*N_z]);
        else
            optaprime=reshape(PolicyIndexesKron(2,:,:,jj),[1,N_a*N_z]);
        end
        Ptran=zeros(N_a,N_a*N_z,'gpuArray');
        Ptran(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptran));
        
        AgentDist(:,jj+1)=Ptran*AgentDist(:,jj);
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
    fprintf(['FAILED TO FIND PARAMETER ', AgeWeightParamNames{1}])
end
% I assume AgeWeights is a row vector, if it has been given as column then transpose it.
if length(AgeWeights)~=size(AgeWeights,2)
    AgeWeights=AgeWeights';
end

% Need to remove the old age weights, and impose the new ones
% Already removed the old age weights earlier, so now just impose the new ones.
% I assume AgeWeights is a row vector
AgentDist=AgentDist.*(ones(N_a*N_z,1)*AgeWeights);

end
