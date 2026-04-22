function [PolicyPath_ForAgentDistIter,PolicyProbsPath,PolicyValuesPath]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath,T,Parameters,n_d,n_a,n_z,n_e,N_j,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions)


%%
if simoptions.experienceasset==1
    
    whichisdforexpasset=length(n_d)-simoptions.setup_experienceasset.l_dexperienceasset+1:length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
    if N_e==0 && N_z==0
        a2primeIndexesPath=zeros(N_a,N_j-1,T-1,'gpuArray');
        a2primeProbsPath=zeros(N_a,N_j-1,T-1,'gpuArray');
        for tt=1:T-1
            aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters,simoptions.setup_experienceasset.aprimeFnParamNames,N_j);
            % [N_j,number of params]

            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_J(PolicyIndexesPath(:,:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, 0, N_j, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec,transpathoptions.fastOLG);
            % Note: a2primeIndexes and a2primeProbs are both [N_a,N_j]
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,:,tt)=a2primeIndexes(:,1:end-1);
            a2primeProbsPath(:,:,tt)=a2primeProbs(:,1:end-1);
        end
    else
        if N_z>0 && N_e>0
            N_ze=N_z*N_e;
        elseif N_z>0 % N_e==0
            N_ze=N_z;
        elseif N_e>0 % N_z==0
            N_ze=N_e;
        end
        if transpathoptions.fastOLG==0
            a2primeIndexesPath=zeros(N_a,N_ze,N_j-1,T-1,'gpuArray');
            a2primeProbsPath=zeros(N_a,N_ze,N_j-1,T-1,'gpuArray');
            for tt=1:T-1
                aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters,simoptions.setup_experienceasset.aprimeFnParamNames,N_j);
                % [N_j,number of params]

                [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_J(PolicyIndexesPath(:,:,:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, N_ze, N_j, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec,transpathoptions.fastOLG);
                % Note: a2primeIndexes and a2primeProbs are both [N_a,N_z,N_j] for fastOLG=0
                % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
                a2primeIndexesPath(:,:,:,tt)=a2primeIndexes(:,:,1:end-1);
                a2primeProbsPath(:,:,:,tt)=a2primeProbs(:,:,1:end-1);
            end
        elseif transpathoptions.fastOLG==1
            a2primeIndexesPath=zeros(N_a,N_j-1,N_ze,T-1,'gpuArray');
            a2primeProbsPath=zeros(N_a,N_j-1,N_ze,T-1,'gpuArray');
            for tt=1:T-1
                aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters,simoptions.setup_experienceasset.aprimeFnParamNames,N_j);
                % [N_j,number of params]

                [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_J(PolicyIndexesPath(:,:,:,:,tt),simoptions.setup_experienceasset.aprimeFn, whichisdforexpasset, n_d, simoptions.setup_experienceasset.n_a1,simoptions.setup_experienceasset.n_a2, N_ze, N_j, simoptions.setup_experienceasset.d_grid, simoptions.setup_experienceasset.a2_grid, aprimeFnParamsVec,transpathoptions.fastOLG);
                % Note: a2primeIndexes and a2primeProbs are both [N_a,N_j,N_z] for fastOLG=1
                % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
                a2primeIndexesPath(:,:,:,tt)=a2primeIndexes(:,1:end-1,:);
                a2primeProbsPath(:,:,:,tt)=a2primeProbs(:,1:end-1,:);
            end
        end
    end
    
    if N_e==0 && N_z==0
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_j-1,1,T-1]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,2,1);
        a2primeIndexesPath(:,:,2,:)=a2primeIndexesPath(:,:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_j-1,1,T-1]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,1,2,1);
        a2primeProbsPath(:,:,2,:)=1-a2primeProbsPath(:,:,2,:); % upper prob
        if simoptions.fastOLG==1
            a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*(N_j-1),2,T-1]);
            a2primeProbsPath=reshape(a2primeProbsPath,[N_a*(N_j-1),2,T-1]);
        end
    else
        if transpathoptions.fastOLG==0
            a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_ze,N_j-1,1,T-1]);
            a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,1,2,1);
            a2primeIndexesPath(:,:,:,2,:)=a2primeIndexesPath(:,:,:,2,:)+1; % upper index
            a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_ze,N_j-1,1,T-1]);
            a2primeProbsPath=repmat(a2primeProbsPath,1,1,1,2,1);
            a2primeProbsPath(:,:,:,2,:)=1-a2primeProbsPath(:,:,:,2,:); % upper prob
            if simoptions.fastOLG==0
                a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*N_ze,N_j-1,2,T-1]);
                a2primeProbsPath=reshape(a2primeProbsPath,[N_a*N_ze,N_j-1,2,T-1]);
            elseif simoptions.fastOLG==1
                a2primeIndexesPath=reshape(permute(a2primeIndexesPath,[1,3,2,4,5]),[N_a*(N_j-1)*N_ze,2,T-1]);
                a2primeProbsPath=reshape(permute(a2primeProbsPath,[1,3,2,4,5]),[N_a*(N_j-1)*N_ze,2,T-1]);
            end
        elseif transpathoptions.fastOLG==1
            a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_j-1,N_ze,1,T-1]);
            a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,1,2,1);
            a2primeIndexesPath(:,:,:,2,:)=a2primeIndexesPath(:,:,:,2,:)+1; % upper index
            a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_j-1,N_ze,1,T-1]);
            a2primeProbsPath=repmat(a2primeProbsPath,1,1,1,2,1);
            a2primeProbsPath(:,:,:,2,:)=1-a2primeProbsPath(:,:,:,2,:); % upper prob
            if simoptions.fastOLG==0
                a2primeIndexesPath=reshape(permute(a2primeIndexesPath,[1,3,2,4,5]),[N_a*N_ze,N_j-1,2,T-1]);
                a2primeProbsPath=reshape(permute(a2primeProbsPath,[1,3,2,4,5]),[N_a*N_ze,N_j-1,2,T-1]);
            elseif simoptions.fastOLG==1
                a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*(N_j-1)*N_ze,2,T-1]);
                a2primeProbsPath=reshape(a2primeProbsPath,[N_a*(N_j-1)*N_ze,2,T-1]);
            end
        end
    end

    a2primeIndexesPath=gather(a2primeIndexesPath);
    a2primeProbsPath=gather(a2primeProbsPath);
end


if simoptions.experienceasset==1
    n_a1=simoptions.setup_experienceasset.n_a1;
else
    n_a1=n_a;
end

%% slowOLG
if transpathoptions.fastOLG==0
    if N_z==0 && N_e==0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,0,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,1,4]); %[N_a,N_j,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:),[N_a,N_j-1,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimePath_slowOLG=gather(PolicyaprimePath);
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_j-1,T-1]),[1,2,3]),[N_a*(N_j-1),T-1]);
            PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:),[1,N_a,N_j-1,T-1]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_j,T]
                L2index=reshape(permute(L2index,[2,3,1,4]),[N_a*(N_j-1),1,T-1]);
                PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
                PolicyProbsPath=zeros(N_a*(N_j-1),2,T-1,'gpuArray'); % preallocate
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
                PolicyProbsPath=gather(PolicyProbsPath);
            end
            PolicyaprimejPath=gather(PolicyaprimejPath);
        end
        clear PolicyIndexesPath PolicyaprimePath L2index
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimePath_slowOLG=reshape(PolicyaprimePath_slowOLG,[N_a,(N_j-1),1,T])+a2primeIndexesPath;
                else
                    PolicyaprimePath_slowOLG=reshape(PolicyaprimePath_slowOLG,[N_a,(N_j-1),1,T])+simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1);
                end
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
                else
                    PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
                end
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    elseif N_z>0 && N_e==0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_z,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_z,N_j-1,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimezPath_slowOLG=gather(PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1));
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_z,T-1]);
            PolicyaprimejzPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_z,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z,1,T-1]);
                PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z,1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
                PolicyProbsPath=zeros(N_a*(N_j-1)*N_z,2,T-1,'gpuArray'); % preallocate
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
                PolicyProbsPath=gather(PolicyProbsPath);
            end
            PolicyaprimejzPath=gather(PolicyaprimejzPath);
        end
        clear PolicyIndexesPath PolicyaprimePath L2index
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimezPath_slowOLG=reshape(PolicyaprimezPath_slowOLG,[N_a*N_z,(N_j-1),1,T])+a2primeIndexesPath;
                else
                    PolicyaprimezPath_slowOLG=reshape(PolicyaprimezPath_slowOLG,[N_a*N_z,(N_j-1),1,T])+simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1);
                end
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
                else
                    PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
                end
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    elseif N_z==0 && N_e>0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_e,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_e,N_j-1,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimePath_slowOLG=gather(PolicyaprimePath);
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_e,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_e,T-1]);
            PolicyaprimejPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1),N_e,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_e,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_e,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_e,1,T-1]);
                PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1)*N_e,1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
                PolicyProbsPath=zeros(N_a*(N_j-1)*N_e,2,T-1,'gpuArray'); % preallocate
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
                PolicyProbsPath=gather(PolicyProbsPath);
            end
            PolicyaprimejPath=gather(PolicyaprimejPath);
        end
        clear PolicyIndexesPath PolicyaprimePath L2index
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimePath_slowOLG=reshape(PolicyaprimePath_slowOLG,[N_a*N_e,(N_j-1),1,T])+a2primeIndexesPath;
                else
                    PolicyaprimePath_slowOLG=reshape(PolicyaprimePath_slowOLG,[N_a*N_e,(N_j-1),1,T])+simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1);
                end
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
                else
                    PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
                end
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    elseif N_z>0 && N_e>0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_z*N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:),[N_a*N_z*N_e,N_j-1,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,:,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimezPath_slowOLG=gather(PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1),N_e,1));
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z*N_e,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_z*N_e,T-1]);
            PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,:,1:N_j-1,:),[1,N_a,N_z*N_e,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,N_e,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z*N_e,1,T-1]);
                PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
                PolicyProbsPath=zeros(N_a*(N_j-1)*N_z*N_e,2,T-1,'gpuArray'); % preallocate
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
                PolicyProbsPath=gather(PolicyProbsPath);
            end
            PolicyaprimejzPath=gather(PolicyaprimejzPath);
        end
        clear PolicyIndexesPath PolicyaprimePath L2index
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimezPath_slowOLG=reshape(PolicyaprimezPath_slowOLG,[N_a*N_z*N_e,(N_j-1),1,T])+a2primeIndexesPath;
                else
                    PolicyaprimezPath_slowOLG=reshape(PolicyaprimezPath_slowOLG,[N_a*N_z*N_e,(N_j-1),1,T])+simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1);
                end
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                if simoptions.setup_experienceasset.N_a1==0
                    PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
                else
                    PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
                end
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    end


    %% fastOLG
elseif transpathoptions.fastOLG==1
    if N_z==0 && N_e==0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,0,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,1,4]); %[N_a,N_j,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:),[N_a*(N_j-1),T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
        end
        PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
        clear PolicyaprimePath % try free up some memory
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:),[N_a*(N_j-1),1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,T-1]
            PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
            PolicyProbsPath=zeros(N_a*(N_j-1),2,T-1,'gpuArray'); % preallocate
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            PolicyProbsPath=gather(PolicyProbsPath);
        end
        PolicyaprimejPath=gather(PolicyaprimejPath);
        clear PolicyIndexesPath L2index
        if simoptions.experienceasset==1
            if simoptions.setup_experienceasset.N_a1==0
                PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
            else
                PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
            end
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        end
    elseif N_z>0 && N_e==0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_z,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_z,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_z,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_z,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_z,T-1]);
        end
        PolicyaprimejzPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1);
        clear PolicyaprimePath % try free up some memory
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_z,T-1]
            PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z,1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
            PolicyProbsPath=zeros(N_a*(N_j-1)*N_z,2,T-1,'gpuArray'); % preallocate
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            PolicyProbsPath=gather(PolicyProbsPath);
        end
        PolicyaprimejzPath=gather(PolicyaprimejzPath);
        clear PolicyIndexesPath L2index
        if simoptions.experienceasset==1
            if simoptions.setup_experienceasset.N_a1==0
                PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
            else
                PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
            end
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        end
    elseif N_z==0 && N_e>0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_e,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_e,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_e,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_e,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_e,T-1]);
        end
        PolicyaprimejPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1),N_e,1);
        clear PolicyaprimePath % try free up some memory
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_e,T-1]
            PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1)*N_e,1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
            PolicyProbsPath=zeros(N_a*(N_j-1)*N_e,2,T-1,'gpuArray'); % preallocate
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            PolicyProbsPath=gather(PolicyProbsPath);
        end
        PolicyaprimejPath=gather(PolicyaprimejPath);
        clear PolicyIndexesPath L2index
        if simoptions.experienceasset==1
            if simoptions.setup_experienceasset.N_a1==0
                PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
            else
                PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);
            end
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        end
    elseif N_z>0 && N_e>0
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_z*N_e,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_z*N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a1)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,T-1]); % aprime index
        elseif length(n_a1)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
        elseif length(n_a1)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
        elseif length(n_a1)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a1(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1)+n_a1(1)*n_a1(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:,:)-1)+n_a1(1)*n_a1(2)*n_a1(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
        end
        PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
        clear PolicyaprimePath % try free up some memory
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_z,N_e,T-1]
            PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
            PolicyProbsPath=zeros(N_a*(N_j-1)*N_z*N_e,2,T-1,'gpuArray'); % preallocate
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            PolicyProbsPath=gather(PolicyProbsPath);
        end
        PolicyaprimejzPath=gather(PolicyaprimejzPath);
        clear PolicyIndexesPath L2index
        if simoptions.experienceasset==1
            if simoptions.setup_experienceasset.N_a1==0
                PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(a2primeIndexesPath,1,2,1);
            else
                PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(simoptions.setup_experienceasset.N_a1*(a2primeIndexesPath-1),1,2,1);                
            end
            PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
        end
    end
end

% clear a2primeIndexesPath a2primeProbsPath % Free up some memory


%% Clean up output
if simoptions.fastOLG==0
    if N_z==0
        PolicyPath_ForAgentDistIter=PolicyaprimePath_slowOLG;
    elseif N_z>0
        PolicyPath_ForAgentDistIter=PolicyaprimezPath_slowOLG;
    end
elseif simoptions.fastOLG==1
    if N_z==0
        PolicyPath_ForAgentDistIter=PolicyaprimejPath;
    elseif N_z>0
        PolicyPath_ForAgentDistIter=PolicyaprimejzPath;
    end
end

if N_probs==1 % =1 means not being used
    PolicyProbsPath=[];
end











end
