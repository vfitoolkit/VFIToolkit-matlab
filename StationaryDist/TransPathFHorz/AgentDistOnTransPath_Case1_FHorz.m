function AgentDistPath=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial, jequalOneDist, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions)
% Note: PricePath is not used, it is just there for legacy compatibility
% jequalOneDist can be a jequalOneDistPath

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if isfield(simoptions,'n_e')
    n_e=simoptions.n_e;
else
    n_e=0;
end
N_e=prod(n_e);

l_a=length(n_a);
if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end


%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.verbose=0;
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.fastOLG=1; % parallel over j, faster but uses more memory
    simoptions.gridinterplayer=0;
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions,'fastOLG')
        if isfield(transpathoptions,'fastOLG')
            simoptions.fastOLG=transpathoptions.fastOLG;
        else
            simoptions.fastOLG=1; % parallel over j, faster but uses more memory
        end
    end
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    elseif simoptions.gridinterplayer==1
        if ~isfield(simoptions,'ngridinterp')
            error('You have simoptions.gridinterplayer, so must also set simoptions.ngridinterp')
        end
    end
end


%% Note: Internally PricePath is matrix of size T-by-'number of prices', similarly for ParamPath
% PricePath is matrix of size T-by-'number of prices'.
% Actually, some of those prices may be 1-by-N_j, so is more subtle than this.
PricePathNames=fieldnames(PricePath);
PricePathStruct=PricePath; 
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for ii=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{ii});
    tempsize=size(temp);
    PricePathSizeVec(ii)=tempsize(tempsize~=T); % Get the dimension which is not T
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePath=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for ii=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{ii}),1)==T
        PricePath(:,PricePathSizeVec(1,ii):PricePathSizeVec(2,ii))=PricePathStruct.(PricePathNames{ii});
    else % Need to transpose
        PricePath(:,PricePathSizeVec(1,ii):PricePathSizeVec(2,ii))=PricePathStruct.(PricePathNames{ii})';
    end
end
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices may be 1-by-N_j, so is more subtle than this.
ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for ii=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{ii});
    tempsize=size(temp);
    ParamPathSizeVec(ii)=tempsize(tempsize~=T); % Get the dimension which is not T
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for ii=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{ii}),1)==T
        ParamPath(:,ParamPathSizeVec(1,ii):ParamPathSizeVec(2,ii))=ParamPathStruct.(ParamPathNames{ii});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,ii):ParamPathSizeVec(2,ii))=ParamPathStruct.(ParamPathNames{ii})';
    end
end


%%
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
AgentDist_initial=gpuArray(AgentDist_initial);


%% Set up exogenous shock processes
[~, pi_z_J, pi_z_J_sim, ~, pi_e_J, pi_e_J_sim, transpathoptions, simoptions]=ExogShockSetup_TPath_FHorz(n_z,[],pi_z,N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,2);
% Convert z and e to age-dependent joint-grids and transtion matrix
% output: z_gridvals_J, pi_z_J, e_gridvals_J, pi_e_J, transpathoptions,vfoptions,simoptions

% Sets up
% transpathoptions.zpathtrivial=1; % z_gridvals_J and pi_z_J are not varying over the path
%                              =0; % they vary over path, so z_gridvals_J_T and pi_z_J_T
% transpathoptions.epathtrivial=1; % e_gridvals_J and pi_e_J are not varying over the path
%                              =0; % they vary over path, so e_gridvals_J_T and pi_e_J_T
% and
% transpathoptions.gridsinGE=1; % grids depend on a GE parameter and so need to be recomputed every iteration
%                           =0; % grids are exogenous


%%
if transpathoptions.verbose==1
    transpathoptions
end


%% Get the age weights, check if they depend on path, and make sure they are the right shape
% It is assumed there is only one Age Weight Parameter (name))
try
    AgeWeights=gpuArray(Parameters.(AgeWeightsParamNames{1}));
catch
    error(['Failed to find parameter ', AgeWeightsParamNames{1}])
end
% If the AgeWeights do not vary over the transition, then we will just set them up now.
transpathoptions.ageweightstrivial=1;
if all(size(AgeWeights)==[N_j,1])
    % Does not depend on transition path period
elseif all(size(AgeWeights)==[1,N_j])
    % Does not depend on transition path period
    % Make AgeWeights a column vector
    AgeWeights=AgeWeights';
end
AgeWeights_T=repelem(AgeWeights,1,T); % N_j-by-T
% Check ParamPath to see if the AgeWeights vary over the transition
temp=strcmp(ParamPathNames,AgeWeightsParamNames{1});
if any(temp)
    transpathoptions.ageweightstrivial=0; % AgeWeights vary over the transition
    [~,kk]=max(temp); % Get index for the AgeWeightsParamNames{1} in ParamPathNames
    % Create AgeWeights_T
    AgeWeights_T=ParamPath(:,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk))'; % N_j-by-T
end

%% Setup for AgentDist_initial (includes a check that if AgeWeights do not change over the transition then they should match the initial agent distribution)
if N_e==0  % no z, no e
    if N_z==0
        AgentDist_initial=reshape(AgentDist_initial,[N_a,N_j]);
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if simoptions.fastOLG==1
            AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j,1]);
        end
    else % z, no e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]);
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if simoptions.fastOLG==1
            AgentDist_initial=reshape(AgentDist_initial,[N_a,N_z,N_j]);
            AgentDist_initial=permute(AgentDist_initial,[1,3,2]);
            AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j*N_z,1]);
        end
    end
else
    if N_z==0 % no z, e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_e,N_j]);
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if simoptions.fastOLG==1
            AgentDist_initial=reshape(AgentDist_initial,[N_a,N_e,N_j]);
            AgentDist_initial=permute(AgentDist_initial,[1,3,2]);
            AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j,N_e]);
        end
    else % z & e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z*N_e,N_j]);
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if simoptions.fastOLG==1
            AgentDist_initial=reshape(AgentDist_initial,[N_a,N_z,N_e,N_j]);
            AgentDist_initial=permute(AgentDist_initial,[1,4,3,2]);
            AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j*N_z,N_e]);
        end
    end
end
if transpathoptions.ageweightstrivial==1
    if max(abs(AgeWeights_initial-AgeWeights'))>10^(-9) % was 10^(-13), but this was problematic with numerical rounding errors
        % Note: AgeWeights_inital is [1,N_j], while AgeWeights is [N_j,1], hence we need the transpose on AgeWeights
        fprintf('AgeWeights are: \n')
        AgeWeights
        fprintf('AgeWeights implicit in the initial agent distribution are: \n')
        AgeWeights_initial
        error('AgeWeights differs from the weights implicit in the initial agent distribution')
    end
end

%% Some inputs needed for simoptions.fastOLG=1 that get precomputed
if simoptions.fastOLG==1
    if N_z==0 && N_e==0
        % No need to do anything
    elseif N_z>0 && N_e==0
        % Precompute some things needed for fastOLG agent dist iteration
        exceptlastj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        exceptfirstj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        justfirstj=repmat(1:1:N_a,1,N_z)+N_a*N_j*repelem(0:1:N_z-1,1,N_a);
    elseif N_z==0 && N_e>0
        % Precompute some things needed for fastOLG agent dist iteration
        exceptlastj=kron(ones(1,(N_j-1)*N_e),1:1:N_a)+kron(kron(ones(1,N_e),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        exceptfirstj=kron(ones(1,(N_j-1)*N_e),1:1:N_a)+kron(kron(ones(1,N_e),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        justfirstj=repmat(1:1:N_a,1,N_e)+N_a*N_j*repelem(0:1:N_e-1,1,N_a);
    elseif N_z>0 && N_e>0
        % Precompute some things needed for fastOLG agent dist iteration
        exceptlastj=kron(ones(1,(N_j-1)*N_z*N_e),1:1:N_a)+kron(kron(ones(1,N_z*N_e),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z*N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        exceptfirstj=kron(ones(1,(N_j-1)*N_z*N_e),1:1:N_a)+kron(kron(ones(1,N_z*N_e),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z*N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        justfirstj=repmat(1:1:N_a,1,N_z*N_e)+N_a*N_j*repelem(0:1:N_z*N_e-1,1,N_a);
    end
end


%% Reorganize PolicyPath to get just what we need, and in the shape needed
if N_e==0
    if N_z==0
        PolicyPath=reshape(PolicyPath, [size(PolicyPath,1),N_a,N_j,T]);
    else
        PolicyPath=reshape(PolicyPath, [size(PolicyPath,1),N_a,N_z,N_j,T]);
    end
else
    if N_z==0
        PolicyPath=reshape(PolicyPath, [size(PolicyPath,1),N_a,N_e,N_j,T]);
    else
        PolicyPath=reshape(PolicyPath, [size(PolicyPath,1),N_a,N_z,N_e,N_j,T]);
    end
end
if l_a==1
    if N_e==0
        if N_z==0
            if simoptions.fastOLG==0
                Policy_aprimePath=reshape(PolicyPath(l_d+1,:,:,:),[1,N_a,N_j,T]);
            elseif simoptions.fastOLG==1
                % PolicyPath=permute(PolicyPath,[1,2,3,4]); % no shocks, so no permute here
                if simoptions.gridinterplayer==0
                    Policy_aprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:),[1,N_a*(N_j-1),T]);
                else
                    Policy_aprimePath=[reshape(PolicyPath(l_d+1,:,1:N_j-1,:),[N_a*(N_j-1),1,T]), 1+reshape(PolicyPath(l_d+1,:,1:N_j-1,:),[N_a*(N_j-1),1,T])]; % lower and upper gridpoints [N_a*(N_j-1),2,T]
                    upperprob=(reshape(PolicyPath(l_d+2,:,1:N_j-1,:),[N_a*(N_j-1),1,T])-1)/(simoptions.ngridinterp+1);
                    PolicyProbsPath=zeros(N_a*(N_j-1),2,T,'gpuArray');
                    PolicyProbsPath(:,1,:)=1-upperprob;
                    PolicyProbsPath(:,2,:)=upperprob;
                end
            end
        else
            if simoptions.fastOLG==0
                Policy_aprimePath=reshape(PolicyPath(l_d+1,:,:,:,:),[1,N_a*N_z,N_j,T]);
            elseif simoptions.fastOLG==1
                PolicyPath=permute(PolicyPath,[1,2,4,3,5]); % swap j and z
                if simoptions.gridinterplayer==0
                    Policy_aprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:),[1,N_a*(N_j-1)*N_z,T]);
                else
                    Policy_aprimePath=[reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,1,T]), 1+reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,1,T])]; % lower and upper gridpoints [N_a*(N_j-1),2,T]
                    upperprob=(reshape(PolicyPath(l_d+2,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_z,1,T])-1)/(simoptions.ngridinterp+1);
                    PolicyProbsPath=zeros(N_a*(N_j-1)*N_z,2,T,'gpuArray');
                    PolicyProbsPath(:,1,:)=1-upperprob;
                    PolicyProbsPath(:,2,:)=upperprob;
                end
            end
        end
    else
        if N_z==0
            if simoptions.fastOLG==0
                Policy_aprimePath=reshape(PolicyPath(l_d+1,:,:,:,:),[1,N_a*N_e,N_j,T]);
            elseif simoptions.fastOLG==1
                PolicyPath=permute(PolicyPath,[1,2,4,3,5]); % swap j and e
                if simoptions.gridinterplayer==0
                    Policy_aprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:),[1,N_a*(N_j-1)*N_e,T]);
                else
                    Policy_aprimePath=[reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,1,T]), 1+reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,1,T])]; % lower and upper gridpoints [N_a*(N_j-1),2,T]
                    upperprob=(reshape(PolicyPath(l_d+2,:,1:N_j-1,:,:),[N_a*(N_j-1)*N_e,1,T])-1)/(simoptions.ngridinterp+1);
                    PolicyProbsPath=zeros(N_a*(N_j-1)*N_e,2,T,'gpuArray');
                    PolicyProbsPath(:,1,:)=1-upperprob;
                    PolicyProbsPath(:,2,:)=upperprob;
                end
            end
        else
            if simoptions.fastOLG==0
                Policy_aprimePath=reshape(PolicyPath(l_d+1,:,:,:,:),[1,N_a*N_z*N_e,N_j,T]);
            elseif simoptions.fastOLG==1
                PolicyPath=permute(PolicyPath,[1,2,5,3,4,6]); % swap j and z,e
                if simoptions.gridinterplayer==0
                    Policy_aprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:,:),[1,N_a*(N_j-1)*N_z*N_e,T]);
                else
                    Policy_aprimePath=[reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T]), 1+reshape(PolicyPath(l_d+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T])]; % lower and upper gridpoints [N_a*(N_j-1),2,T]
                    upperprob=(reshape(PolicyPath(l_d+2,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T])-1)/(simoptions.ngridinterp+1);
                    PolicyProbsPath=zeros(N_a*(N_j-1)*N_z*N_e,2,T,'gpuArray');
                    PolicyProbsPath(:,1,:)=1-upperprob;
                    PolicyProbsPath(:,2,:)=upperprob;
                end
            end
        end
    end
else
    error('Only one endogenous state currently supported for TPath')
end


%% Check if jequalOneDistPath is a path or not (and reshape appropriately)
jequalOneDist=gpuArray(jequalOneDist);
temp=size(jequalOneDist);
% Note: simoptions.fastOLG is handled via 'justfirstj', rather than via shape of jequalOneDist
if temp(end)==T % jequalOneDist depends on T
    transpathoptions.trivialjequalonedist=0;
    if N_z==0
        if N_e==0
            jequalOneDist=reshape(jequalOneDist,[N_a,T]);
        else
            jequalOneDist=reshape(jequalOneDist,[N_a*N_e,T]);
        end
    else
        if N_e==0
            jequalOneDist=reshape(jequalOneDist,[N_a*N_z,T]);
        else
            jequalOneDist=reshape(jequalOneDist,[N_a*N_z*N_e,T]);
        end
    end
else
    transpathoptions.trivialjequalonedist=1;
    if N_z==0
        if N_e==0
            jequalOneDist=reshape(jequalOneDist,[N_a,1]);
        else
            jequalOneDist=reshape(jequalOneDist,[N_a*N_e,1]);
        end
    else
        if N_e==0
            jequalOneDist=reshape(jequalOneDist,[N_a*N_z,1]);
        else
            jequalOneDist=reshape(jequalOneDist,[N_a*N_z*N_e,1]);
        end
    end
end

if transpathoptions.trivialjequalonedist==0
    jequalOneDist_T=jequalOneDist;
    jequalOneDist=jequalOneDist_T(:,1);
end


%% Remove the age weights, do all the iterations, then put the age weights back in at the end. (faster as saves putting weights in and then removing them T times)


%% Do the AgentDistPath calculations
if simoptions.gridinterplayer==0
    if N_e==0
        if N_z==0 % no z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, no e
                AgentDistPath=zeros(N_a,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_raw(AgentDist,Policy_aprime,N_a,N_j,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, no z, no e
                AgentDistPath=zeros(N_a*N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_raw(AgentDist,Policy_aprime,N_a,N_j,jequalOneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(AgeWeights_T,N_a,1); % put in the age weights
            end

        else % z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, no e
                AgentDistPath=zeros(N_a*N_z,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_raw(AgentDist,Policy_aprime,N_a,N_z,N_j,pi_z_J,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, z, no e
                % AgentDist is [N_a*N_j*N_z,1]
                AgentDistPath=zeros(N_a*N_j*N_z,T,'gpuArray');
                AgentDist=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_raw(AgentDist,Policy_aprime,N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % put in the age weights
            end
        end
    else
        if N_z==0 % no z, e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, e
                AgentDistPath=zeros(N_a*N_e,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_e_raw(AgentDist,Policy_aprime,N_a,N_e,N_j,pi_e_J,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, no z, e
                % AgentDist is [N_a*N_j,N_e,1]
                AgentDistPath=zeros(N_a*N_j,N_e,T,'gpuArray');
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_e_raw(AgentDist,Policy_aprime,N_a,N_e,N_j,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1); % put in the age weights
            end

        else % z and e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, e
                AgentDistPath=zeros(N_a*N_z*N_e,N_j,T,'gpuArray'); % Whether or not using simoptions.fastOLG
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                    end
                    if transpathoptions.epathtrivial==0
                        simoptions.pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_e_raw(AgentDist,Policy_aprime,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, z, e
                AgentDistPath=zeros(N_a*N_j*N_z,N_e,T,'gpuArray'); % Whether or not using simoptions.fastOLG
                AgentDist=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.epathtrivial==0
                        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,Policy_aprime,N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1),N_z,1); % put in the age weights
            end
        end
    end


elseif simoptions.gridinterplayer==1
    if N_e==0
        if N_z==0 % no z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, no e, gridinterplayer=1
                error('Not yet implemented')
            else
                %% fastOLG=1, no z, no e, gridinterplayer=1
                AgentDistPath=zeros(N_a*N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_TwoProbs_noz_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_j,jequalOneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(AgeWeights_T,N_a,1); % put in the age weights
            end

        else % z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, no e, gridinterplayer=1
                error('Not yet implemented')
            else
                %% fastOLG=1, z, no e, gridinterplayer=1
                % AgentDist is [N_a*N_j*N_z,1]
                AgentDistPath=zeros(N_a*N_j*N_z,T,'gpuArray');
                AgentDist=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_TwoProbs_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % put in the age weights
            end

        end
    else
        if N_z==0 % no z, e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, e, gridinterplayer=1
                error('Not yet implemented')
            else
                %% fastOLG=1, no z, e, gridinterplayer=1
                AgentDistPath=zeros(N_a*N_j,N_e,T,'gpuArray'); % Whether or not using simoptions.fastOLG
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_TwoProbs_noz_e_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_e,N_j,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1); % put in the age weights
            end

        else % z and e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, e, gridinterplayer=1
                error('Not yet implemented')
            else
                %% fastOLG=1, z, e, gridinterplayer=1
                AgentDistPath=zeros(N_a*N_j*N_z,N_e,T,'gpuArray'); % Whether or not using simoptions.fastOLG
                AgentDist=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.epathtrivial==0
                        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=Policy_aprimePath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_TwoProbs_e_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1),N_z,1); % put in the age weights
            end
        end
    end
end


%%
if N_e==0
    if N_z==0
        AgentDistPath=reshape(AgentDistPath,[n_a,N_j,T]);
    else
        if simoptions.fastOLG==1
            AgentDistPath=permute(reshape(AgentDistPath,[N_a,N_j,N_z,T]),[1,3,2,4]);
        end
        AgentDistPath=reshape(AgentDistPath,[n_a,n_z,N_j,T]);
    end
else
    if N_z==0
        if simoptions.fastOLG==1
            AgentDistPath=permute(reshape(AgentDistPath,[N_a,N_j,N_e,T]),[1,3,2,4]);
        end
        AgentDistPath=reshape(AgentDistPath,[n_a,n_e,N_j,T]);
    else
        if simoptions.fastOLG==1
            AgentDistPath=permute(reshape(AgentDistPath,[N_a,N_j,N_z,N_e,T]),[1,3,4,2,5]);
        end
        AgentDistPath=reshape(AgentDistPath,[n_a,n_z,n_e,N_j,T]);
    end
end


end