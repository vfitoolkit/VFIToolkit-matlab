function [PolicyPath_ForAgentDistIter,PolicyProbsPath, PolicyValuesPath]=TransitionPath_FHorz_shooting_main_Step2_AdjustPolicy(PolicyIndexesPath,PolicyProbsPath,T,n_d,n_a,n_z,n_e,N_j,l_d,l_aprime,N_a,N_z,N_e,N_probs,d_gridvals,aprime_gridvals,transpathoptions,vfoptions,simoptions)

if transpathoptions.fastOLG==0
    if N_z==0 && N_e==0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:),[N_a,N_j-1,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimePath_slowOLG=PolicyaprimePath;
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_j-1,T-1]),[1,2,3]),[N_a*(N_j-1),T-1]);
            PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:),[1,N_a,N_j-1,T-1]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_j,T]
                L2index=reshape(permute(L2index,[2,3,1,4]),[N_a*(N_j-1),1,T-1]);
                PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,0,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,1,4]); %[N_a,N_j,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    elseif N_z>0 && N_e==0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_z,N_j-1,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimezPath_slowOLG=PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_z,T-1]);
            PolicyaprimejzPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_z,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z,1,T-1]);
                PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z,1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_z,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    elseif N_z==0 && N_e>0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_e,N_j-1,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimePath_slowOLG=PolicyaprimePath;
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_e,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_e,T-1]);
            PolicyaprimejPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1),N_e,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_e,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_e,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_e,1,T-1]);
                PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1)*N_e,1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_e,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    elseif N_z>0 && N_e>0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:),[N_a*N_z*N_e,N_j-1,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
        end
        if simoptions.fastOLG==0
            PolicyaprimezPath_slowOLG=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1),N_e,1);
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z*N_e,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_z*N_e,T-1]);
            PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,:,1:N_j-1,:),[1,N_a,N_z*N_e,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,N_e,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z*N_e,1,T-1]);
                PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
                PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,0);
        PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_z*N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    end


    %% fastOLG
elseif transpathoptions.fastOLG==1
    if N_z==0 && N_e==0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:),[N_a*(N_j-1),T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
        end
        PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:),[N_a*(N_j-1),1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,T-1]
            PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,0,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,1,4]); %[N_a,N_j,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    elseif N_z>0 && N_e==0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_z,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_z,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_z,T-1]);
        end
        PolicyaprimejzPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1);
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_z,T-1]
            PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z,1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_z,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_z,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    elseif N_z==0 && N_e>0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_e,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_e,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:)-1),[N_a*(N_j-1)*N_e,T-1]);
        end
        PolicyaprimejPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1),N_e,1);
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_e,T-1]
            PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1)*N_e,1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,n_e,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_e,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    elseif N_z>0 && N_e>0
        % Modify PolicyIndexesPath into forms needed for forward iteration
        % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
        % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
        if isscalar(n_a)
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,T-1]); % aprime index
        elseif length(n_a)==2
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
        elseif length(n_a)==3
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
        elseif length(n_a)==4
            PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
        end
        PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_z,N_e,T-1]
            PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        end
        % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
        PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_z*N_e,T]
        PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_z*N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars
    end
end


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

if N_probs==0
    PolicyProbsPath=[];
end











end