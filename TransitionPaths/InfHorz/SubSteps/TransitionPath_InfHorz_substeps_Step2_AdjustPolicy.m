function [PolicyPath_ForAgentDistIter,PolicyProbsPath,PolicyValuesPath]=TransitionPath_InfHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath,T,Parameters,n_d,n_a,n_z,n_e,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions)


%%
if simoptions.experienceasset==1
    % Note: Have not yet implemented to permite the aprimeFn parameters to vary over time path tt

    whichisdforexpasset=length(n_d)-simoptions.setup_experienceasset.l_dexperienceasset+1:length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
    if N_e==0 && N_z==0
        a2primeIndexesPath=zeros(N_a,T-1,'gpuArray');
        a2primeProbsPath=zeros(N_a,T-1,'gpuArray');
        for tt=1:T-1
            aprimeFnParamsVec=CreateVectorFromParams(Parameters,simoptions.setup_experienceasset.aprimeFnParamNames);
            % [1,number of params]

            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset(PolicyIndexesPath(:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, 0, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec);
            % Note: a2primeIndexes and a2primeProbs are both [N_a,1]
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,tt)=a2primeIndexes;
            a2primeProbsPath(:,tt)=a2primeProbs;
        end
    else
        if N_z>0 && N_e>0
            N_ze=N_z*N_e;
        elseif N_z>0 % N_e==0
            N_ze=N_z;
        elseif N_e>0 % N_z==0
            N_ze=N_e;
        end
        a2primeIndexesPath=zeros(N_a,N_ze,T-1,'gpuArray');
        a2primeProbsPath=zeros(N_a,N_ze,T-1,'gpuArray');
        for tt=1:T-1
            aprimeFnParamsVec=CreateVectorFromParams(Parameters,simoptions.setup_experienceasset.aprimeFnParamNames);
            % [1,number of params]

            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset(PolicyIndexesPath(:,:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, N_ze, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec);
            % Note: a2primeIndexes and a2primeProbs are both [N_a,N_z] for fastOLG=0
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,:,tt)=a2primeIndexes;
            a2primeProbsPath(:,:,tt)=a2primeProbs;
        end
    end

    if N_e==0 && N_z==0
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,1,T-1]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,2,1);
        a2primeIndexesPath(:,2,:)=a2primeIndexesPath(:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,1,T-1]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,2,1);
        a2primeProbsPath(:,2,:)=1-a2primeProbsPath(:,2,:); % upper prob
    else
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_ze,1,T-1]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,2,1);
        a2primeIndexesPath(:,:,2,:)=a2primeIndexesPath(:,:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_ze,1,T-1]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,1,2,1);
        a2primeProbsPath(:,:,2,:)=1-a2primeProbsPath(:,:,2,:); % upper prob
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*N_ze,2,T-1]);
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a*N_ze,2,T-1]);
    end


    if simoptions.setup_experienceasset.N_a1==0
        error('Not yet implemented for experienceasset without a1 (ask on forum if you need this)')
        % Note to self: Problem is that the create of PolicyaprimePath
        % assumes that there is an a1, if no a1 this needs to be skipped
        % and just use the a2primeIndexesPath on its own.
    end
end


if simoptions.experienceasset==1
    n_a1=simoptions.setup_experienceasset.n_a1;
else
    n_a1=n_a;
end


%%
if N_z==0 && N_e==0
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,0,T-1,d_gridvals,aprime_gridvals,vfoptions,1);
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,T-1]),[2,1,3]); %[N_a,l_d+l_a,T-1]
    % Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a1)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:),[N_a,T-1]); % aprime index
    elseif length(n_a1)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:)-1),[N_a,T-1]);
    elseif length(n_a1)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:)-1),[N_a,T-1]);
    elseif length(n_a1)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:)-1),[N_a,T-1]);
    end
    % Just use PolicyaprimePath for simoptions.gridinterplayer==0, otherwise
    if simoptions.gridinterplayer==1
        L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:),[N_a,1,T-1]); % PolicyPath is of size [l_d+l_aprime+1,N_a,T]
        PolicyaprimePath=reshape(PolicyaprimePath,[N_a,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimePath=repelem(PolicyaprimePath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimePath(:,2,:)=PolicyaprimePath(:,2,:)+1; % upper grid index
        PolicyProbsPath=zeros(N_a,2,T-1,'gpuArray'); % preallocate
        PolicyProbsPath(:,2,:)=L2index; % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
    elseif N_probs>1 % for a reason other than gridinterplayer
        PolicyaprimePath=reshape(PolicyaprimePath,[N_a,1,T-1]); % so can assume this size later
    end
    clear PolicyIndexesPath L2index
    if simoptions.experienceasset==1
        if simoptions.setup_experienceasset.N_a1==0
            PolicyaprimePath=repmat(PolicyaprimePath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
        else
            PolicyaprimePath=repmat(PolicyaprimePath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
        end
        if exist('PolicyProbsPath','var')
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        else
            PolicyProbsPath=a2primeProbsPath;
        end
    end

    PolicyaprimePath=gather(PolicyaprimePath);
    if simoptions.gridinterplayer==1
        PolicyProbsPath=gather(PolicyProbsPath);
    end

elseif N_z>0 && N_e==0
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,T-1,d_gridvals,aprime_gridvals,vfoptions,1);
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T-1]),[2,3,1,4]); %[N_a,N_z,l_d+l_a,T-1]
    % Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime
    % If there is z then PolicyaprimezPath
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a1)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:),[N_a*N_z,T-1]); % aprime index
    elseif length(n_a1)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1),[N_a*N_z,T-1]);
    elseif length(n_a1)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1),[N_a*N_z,T-1]);
    elseif length(n_a1)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,:)-1),[N_a*N_z,T-1]);
    end
    PolicyaprimezPath=gather(PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1));

    if simoptions.gridinterplayer==1
        L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,:),[N_a*N_z,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,T]
        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimezPath=repelem(PolicyaprimezPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimezPath(:,2,:)=PolicyaprimezPath(:,2,:)+1; % upper grid index
        PolicyProbsPath=zeros(N_a*N_z,2,T-1,'gpuArray'); % preallocate
        PolicyProbsPath(:,2,:)=L2index; % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        PolicyProbsPath=gather(PolicyProbsPath);
    elseif N_probs>1 % for a reason other than gridinterplayer
        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T-1]); % so can assume this size later
    end

    clear PolicyIndexesPath PolicyaprimePath L2index
    if simoptions.experienceasset==1
        if simoptions.setup_experienceasset.N_a1==0
            PolicyaprimePath=repmat(PolicyaprimezPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
        else
            PolicyaprimePath=repmat(PolicyaprimezPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
        end
        if exist('PolicyProbsPath','var')
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        else
            PolicyProbsPath=a2primeProbsPath;
        end
    end

    PolicyaprimezPath=gather(PolicyaprimezPath);
    if simoptions.gridinterplayer==1
        PolicyProbsPath=gather(PolicyProbsPath);
    end

elseif N_z==0 && N_e>0
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,n_e,T-1,d_gridvals,aprime_gridvals,vfoptions,1);
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_ze,T-1]),[2,3,1,4]); %[N_a,N_e,l_d+l_a,T-1]
    % Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime
    % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a1)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:),[N_a*N_e,T-1]); % aprime index
    elseif length(n_a1)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1),[N_a*N_e,T-1]);
    elseif length(n_a1)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1),[N_a*N_e,T-1]);
    elseif length(n_a1)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,:)-1),[N_a*N_e,T-1]);
    end
    % Just use PolicyaprimePath for simoptions.gridinterplayer==0, otherwise
    if simoptions.gridinterplayer==1
        L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,:),[N_a*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_e,T]
        PolicyaprimePath=reshape(PolicyaprimePath,[N_a*N_e,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimePath=repelem(PolicyaprimePath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimePath(:,2,:)=PolicyaprimePath(:,2,:)+1; % upper grid index
        PolicyProbsPath=zeros(N_a*N_e,2,T-1,'gpuArray'); % preallocate
        PolicyProbsPath(:,2,:)=L2index; % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
    elseif N_probs>1 % for a reason other than gridinterplayer
        PolicyaprimePath=reshape(PolicyaprimePath,[N_a*N_e,1,T-1]); % so can assume this size later
    end
    clear PolicyIndexesPath L2index
    if simoptions.experienceasset==1
        if simoptions.setup_experienceasset.N_a1==0
            PolicyaprimePath=repmat(PolicyaprimePath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
        else
            PolicyaprimePath=repmat(PolicyaprimePath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
        end
        if exist('PolicyProbsPath','var')
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        else
            PolicyProbsPath=a2primeProbsPath;
        end
    end

    PolicyaprimePath=gather(PolicyaprimePath);
    if simoptions.gridinterplayer==1
        PolicyProbsPath=gather(PolicyProbsPath);
    end

elseif N_z>0 && N_e>0
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],T-1,d_gridvals,aprime_gridvals,vfoptions,1);
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z*N_e,T-1]),[2,3,1,4]); %[N_a,N_z*N_e,l_d+l_a,T-1]
    % Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime
    % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a1)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,:),[N_a*N_z*N_e,T-1]); % aprime index
    elseif length(n_a1)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:,:)-1),[N_a*N_z*N_e,T-1]);
    elseif length(n_a1)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:,:)-1),[N_a*N_z*N_e,T-1]);
    elseif length(n_a1)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,:,:)-1),[N_a*N_z*N_e,T-1]);
    end
    PolicyaprimezPath=gather(PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1),N_e,1));
    if simoptions.gridinterplayer==1
        L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,:,:),[N_a*N_z*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,N_e,T]
        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimezPath=repelem(PolicyaprimezPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimezPath(:,2,:)=PolicyaprimezPath(:,2,:)+1; % upper grid index
        PolicyProbsPath=zeros(N_a*N_z*N_e,2,T-1,'gpuArray'); % preallocate
        PolicyProbsPath(:,2,:)=L2index; % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        PolicyProbsPath=gather(PolicyProbsPath);
    elseif N_probs>1 % for a reason other than gridinterplayer
        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z*N_e,1,T-1]); % so can assume this size later
    end

    clear PolicyIndexesPath PolicyaprimePath L2index
    if simoptions.experienceasset==1
        if simoptions.setup_experienceasset.N_a1==0
            PolicyaprimePath=repmat(PolicyaprimezPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
        else
            PolicyaprimePath=repmat(PolicyaprimezPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
        end
        if exist('PolicyProbsPath','var')
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        else
            PolicyProbsPath=a2primeProbsPath;
        end
    end

    PolicyaprimezPath=gather(PolicyaprimezPath);
    if simoptions.gridinterplayer==1
        PolicyProbsPath=gather(PolicyProbsPath);
    end
end


% clear a2primeIndexesPath a2primeProbsPath % Free up some memory


%% Clean up output
if N_z==0
    PolicyPath_ForAgentDistIter=PolicyaprimePath;
elseif N_z>0
    PolicyPath_ForAgentDistIter=PolicyaprimezPath;
end

if N_probs==1 % =1 means not being used
    PolicyProbsPath=[];
end









end
