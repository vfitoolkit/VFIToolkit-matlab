function AgentDistPath=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial, jequalOneDist, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions)
% Note: PricePath is not used, it is just there for legacy compatibility
% jequalOneDist can be a jequalOneDistPath

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if isfield(simoptions,'n_e')
    n_e=simoptions.n_e;
    N_e=prod(n_e);
else
    N_e=0;
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
[~, pi_z_J, ~, pi_e_J, transpathoptions, simoptions]=ExogShockSetup_TPath_FHorz(n_z,[],pi_z,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,2);
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
    % Make AgeWeights a row vector, as this is what subcommands hardcode
    AgeWeights=AgeWeights';
elseif all(size(AgeWeights)==[1,N_j])
    % Does not depend on transition path period
end
% Check ParamPath to see if the AgeWeights vary over the transition
temp=strcmp(ParamPathNames,AgeWeightsParamNames{1});
if any(temp)
    transpathoptions.ageweightstrivial=0; % AgeWeights vary over the transition
    [~,kk]=max(temp); % Get index for the AgeWeightsParamNames{1} in ParamPathNames
    % Create AgeWeights_T
    AgeWeights=ParamPath(:,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk))'; % This will always be N_j-by-T (as transpose)
end

% If using simoptions.fastOLG==1, need to make AgeWeights a different shape
% This is dones later, as want to keep current AgeWeights so when it is
% zpathtrival==0 we can make sure the age weights match what is implicit in the AgentDist_initial

%% Reorganize PolicyPath to get just what we need, and in the shape needed
if N_z==0
    if N_e==0
        PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j, T, simoptions);
    else
        PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_e, N_j, T, simoptions);
    end
else
    if N_e==0
        PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j, T, simoptions);
    else
        PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_e(PolicyPath, n_d, n_a, n_z, n_e, N_j, T, simoptions);
    end
end
% We only need aprime for the AgentDistPath, so throw away and d variables
if N_d==0
    optaprimePath=PolicyPath;
else
    if N_z==0
        if N_e==0
            optaprimePath=shiftdim(PolicyPath(2,:,:,:),1); % a,j,t
        else
            optaprimePath=shiftdim(PolicyPath(2,:,:,:,:),1);            
        end
    else
        if N_e==0
            optaprimePath=shiftdim(PolicyPath(2,:,:,:,:),1);            
        else
            optaprimePath=shiftdim(PolicyPath(2,:,:,:,:,:),1);            
        end
    end
end

%% Setup for various objects
if N_z==0
    if N_e==0  % no z, no e
        AgentDist_initial=reshape(AgentDist_initial,[N_a,N_j]); % if simoptions.fastOLG==0
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if transpathoptions.ageweightstrivial==0
            % AgeWeights_T is N_j-by-T (or if simoptions.fastOLG=1, then N_a*N_j-by-T, which will be done later)
            AgeWeights_T=AgeWeights;
        elseif transpathoptions.ageweightstrivial==1
            if max(abs(AgeWeights_initial-AgeWeights))>10^(-9) % was 10^(-13), but this was problematic with numerical rounding errors
                error('AgeWeights differs from the weights implicit in the initial agent distribution')
            end
            AgeWeights=AgeWeights_initial;
            AgeWeightsOld=AgeWeights;
        end
        if simoptions.fastOLG==1
            AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j,1]);
            % Note: do the double reshape() as cannot get AgeWeights_initial from the final shape
            AgeWeights_initial=kron(AgeWeights_initial',ones(N_a,1,'gpuArray'));
            if transpathoptions.ageweightstrivial==0
                AgeWeights_T=repelem(AgeWeights_T,N_a,1); % Vectorized as N_a*N_j-by-T
            else % AgeWeights do not change over time, so just set them all to same as AgeWeights_initial
                AgeWeights=AgeWeights_initial;
                AgeWeightsOld=AgeWeights;
            end
        end
    else % no z, e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_e,N_j]); % if simoptions.fastOLG==0
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if transpathoptions.ageweightstrivial==0
            % AgeWeights_T is N_j-by-T (or if simoptions.fastOLG=1, then N_a*N_e*N_j-by-T, which will be done later)
            AgeWeights_T=AgeWeights;
        elseif transpathoptions.ageweightstrivial==1
            if max(abs(AgeWeights_initial-AgeWeights))>10^(-13)
                error('AgeWeights differs from the weights implicit in the initial agent distribution (get different weights if calculate from AgentDist_initial vs if look in Parameters at AgeWeightsParamNames)')
            end
            AgeWeights=AgeWeights_initial;
            AgeWeightsOld=AgeWeights;
        end
        if simoptions.fastOLG==1
            % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
            AgentDist_initial=reshape(permute(reshape(AgentDist_initial,[N_a,N_e,N_j]),[1,3,2]),[N_a*N_j,N_e]);
            % Note: do the double reshape() as cannot get AgeWeights_initial from the final shape
            AgeWeights_initial=kron(AgeWeights_initial',ones(N_a,1,'gpuArray')); % simoptions.fastOLG=1 so this is (a,j,z)-by-1 [weights are identical over e so just ignore this dimension]
            % Similarly, we want pi_z_J to be (j,z,z')
            pi_z_J_sim=gather(reshape(pi_z_J(1:N_j-1,:,:),[(N_j-1)*N_z,N_z])); % For agent dist we want it to be (j,z,z')
            if transpathoptions.ageweightstrivial==0
                AgeWeights_T=repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % Vectorized as N_a*N_j*N_z-by-T
            else % AgeWeights do not change over time, so just set them all to same as AgeWeights_initial
                AgeWeights=AgeWeights_initial;
                AgeWeightsOld=AgeWeights;
            end

            % Precompute some things needed for fastOLG agent dist iteration
            exceptlastj=kron(ones(1,(N_j-1)*N_e),1:1:N_a)+kron(kron(ones(1,N_e),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
            exceptfirstj=kron(ones(1,(N_j-1)*N_e),1:1:N_a)+kron(kron(ones(1,N_e),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
            justfirstj=repmat(1:1:N_a,1,N_e)+N_a*N_j*repelem(0:1:N_e-1,1,N_a);
            % and we now need additional pi_e_J_sim
            pi_e_J_sim=kron(kron(ones(N_z,1,'gpuArray'),gpuArray(pi_e_J(:,2:end))'),ones(N_a,1,'gpuArray')); % (a,j,z)-by-e (but only for jj=2:end)
        end
    end
else
    if N_e==0 % z, no e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]); % if simoptions.fastOLG==0
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if transpathoptions.ageweightstrivial==0
            % AgeWeights_T is N_j-by-T (or if simoptions.fastOLG=1, then N_a*N_j*N_z-by-T, which will be done later)
            AgeWeights_T=AgeWeights;
        elseif transpathoptions.ageweightstrivial==1
            if max(abs(AgeWeights_initial-AgeWeights))>10^(-9) % was 10^(-13), but this was problematic with numerical rounding errors
                error('AgeWeights differs from the weights implicit in the initial agent distribution (get different weights if calculate from AgentDist_initial vs if look in Parameters at AgeWeightsParamNames)')
            end
            AgeWeights=AgeWeights_initial;
            AgeWeightsOld=AgeWeights;
        end
        if simoptions.fastOLG==1
            % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
            AgentDist_initial=reshape(permute(reshape(AgentDist_initial,[N_a,N_z,N_j]),[1,3,2]),[N_a*N_j*N_z,1]);
            % Note: do the double reshape() as cannot get AgeWeights_initial from the final shape
            AgeWeights_initial=kron(ones(N_z,1,'gpuArray'),kron(AgeWeights_initial',ones(N_a,1,'gpuArray'))); % simoptions.fastOLG=1 so this is (a,j,z)-by-1
            % Similarly, we want pi_z_J to be (j,z,z')
            pi_z_J_sim=gather(reshape(pi_z_J(1:N_j-1,:,:),[(N_j-1)*N_z,N_z])); % For agent dist we want it to be (j,z,z')
            if transpathoptions.ageweightstrivial==0
                AgeWeights_T=repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % Vectorized as N_a*N_j*N_z-by-T
            else % AgeWeights do not change over time, so just set them all to same as AgeWeights_initial
                AgeWeights=AgeWeights_initial;
                AgeWeightsOld=AgeWeights;
            end

            % Precompute some things needed for fastOLG agent dist iteration
            exceptlastj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
            exceptfirstj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
            justfirstj=repmat(1:1:N_a,1,N_z)+N_a*N_j*repelem(0:1:N_z-1,1,N_a);
            II1=repmat(1:1:(N_j-1)*N_z,1,N_z);
            II2=repmat(1:1:(N_j-1),1,N_z*N_z)+repelem((N_j-1)*(0:1:N_z-1),1,N_z*(N_j-1));
            pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
        end
    else % z & e
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z*N_e,N_j]); % if simoptions.fastOLG==0
        AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
        if transpathoptions.ageweightstrivial==0
            % AgeWeights_T is N_j-by-T (or if simoptions.fastOLG=1, then N_a*N_z*N_e*N_j-by-T, which will be done later)
            AgeWeights_T=AgeWeights;
        elseif transpathoptions.ageweightstrivial==1
            if max(abs(AgeWeights_initial-AgeWeights))>10^(-13)
                error('AgeWeights differs from the weights implicit in the initial agent distribution (get different weights if calculate from AgentDist_initial vs if look in Parameters at AgeWeightsParamNames)')
            end
            AgeWeights=AgeWeights_initial;
            AgeWeightsOld=AgeWeights;
        end
        if simoptions.fastOLG==1
            % simoptions.fastOLG==1, so AgentDist is treated as : (a,j,z)-by-1
            AgentDist_initial=reshape(permute(reshape(AgentDist_initial,[N_a,N_z,N_e,N_j]),[1,4,2,3]),[N_a*N_j*N_z,N_e]);
            % Note: do the double reshape() as cannot get AgeWeights_initial from the final shape
            AgeWeights_initial=kron(ones(N_z,1,'gpuArray'),kron(AgeWeights_initial',ones(N_a,1,'gpuArray'))); % simoptions.fastOLG=1 so this is (a,j,z)-by-1 [weights are identical over e so just ignore this dimension]
            % Similarly, we want pi_z_J to be (j,z,z')
            pi_z_J_sim=gather(reshape(pi_z_J(1:N_j-1,:,:),[(N_j-1)*N_z,N_z])); % For agent dist we want it to be (j,z,z')
            if transpathoptions.ageweightstrivial==0
                AgeWeights_T=repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % Vectorized as N_a*N_j*N_z-by-T
            else % AgeWeights do not change over time, so just set them all to same as AgeWeights_initial
                AgeWeights=AgeWeights_initial;
                AgeWeightsOld=AgeWeights;
            end

            % Precompute some things needed for fastOLG agent dist iteration
            exceptlastj=kron(ones(1,(N_j-1)*N_z*N_e),1:1:N_a)+kron(kron(ones(1,N_z*N_e),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z*N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
            exceptfirstj=kron(ones(1,(N_j-1)*N_z*N_e),1:1:N_a)+kron(kron(ones(1,N_z*N_e),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z*N_e-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
            justfirstj=repmat(1:1:N_a,1,N_z*N_e)+N_a*N_j*repelem(0:1:N_z*N_e-1,1,N_a);
            % note that following are not affected by e
            II1=repmat(1:1:(N_j-1)*N_z,1,N_z);
            II2=repmat(1:1:(N_j-1),1,N_z*N_z)+repelem((N_j-1)*(0:1:N_z-1),1,N_z*(N_j-1));
            pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
            % and we now need additional pi_e_J_sim
            pi_e_J_sim=kron(kron(ones(N_z,1,'gpuArray'),gpuArray(pi_e_J(:,2:end))'),ones(N_a,1,'gpuArray')); % (a,j,z)-by-e (but only for jj=2:end)
        end
    end
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


%% Do the AgentDistPath calculations
if N_e==0
    if N_z==0
        if simoptions.fastOLG==0
            %% fastOLG=0, no z, no e
            AgentDistPath=zeros(N_a,N_j,T,'gpuArray');
            AgentDistPath(:,:,1)=AgentDist_initial;

            AgentDist=AgentDist_initial;
            if transpathoptions.ageweightstrivial==0
                AgeWeights=AgeWeights_initial;
            end
            for tt=1:T-1
                if transpathoptions.ageweightstrivial==0
                    AgeWeightsOld=AgeWeights;
                    AgeWeights=AgeWeights_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                if transpathoptions.trivialjequalonedist==0
                    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                %Get the current optimal policy
                optaprime=optaprimePath(:,:,tt);
                AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_j,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            end
        else
            %% fastOLG=1, no z, no e
            AgentDistPath=zeros(N_a*N_j,T,'gpuArray');
            AgentDistPath(:,1)=AgentDist_initial;

            AgentDist=AgentDist_initial;
            if transpathoptions.ageweightstrivial==0
                AgeWeights=AgeWeights_initial;
            end
            for tt=1:T-1
                if transpathoptions.ageweightstrivial==0
                    AgeWeightsOld=AgeWeights;
                    AgeWeights=AgeWeights_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                if transpathoptions.trivialjequalonedist==0
                    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                %Get the current optimal policy
                optaprime=optaprimePath(:,:,tt);
                optaprime=gather(reshape(optaprime(:,1:end-1),[1,N_a*(N_j-1)])); % swap order to j,z
                AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_j,jequalOneDist);
                AgentDistPath(:,tt+1)=AgentDist;
            end
        end

    else
        if simoptions.fastOLG==0
            %% fastOLG=0, z, no e
            AgentDistPath=zeros(N_a*N_z,N_j,T,'gpuArray');
            AgentDistPath(:,:,1)=AgentDist_initial;
            AgentDist=AgentDist_initial;
            if transpathoptions.ageweightstrivial==0
                AgeWeights=AgeWeights_initial;
            end
            for tt=1:T-1
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on T, so is just in simoptions already
                if transpathoptions.ageweightstrivial==0
                    AgeWeightsOld=AgeWeights;
                    AgeWeights=AgeWeights_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                if transpathoptions.trivialjequalonedist==0
                    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                %Get the current optimal policy
                optaprime=optaprimePath(:,:,:,tt);
                AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_z,N_j,pi_z_J,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            end
        else
            %% fastOLG=1, z, no e
            % AgentDist is [N_a*N_j*N_z,1]

            AgentDistPath=zeros(N_a*N_j*N_z,T,'gpuArray');
            AgentDistPath(:,1)=AgentDist_initial;
            AgentDist=AgentDist_initial;
            if transpathoptions.ageweightstrivial==0
                AgeWeights=AgeWeights_initial;
            end
            for tt=1:T-1
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                    pi_z_J_sim=gather(pi_z_J(1:end-1,:,:));
                    pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on T, so is just in simoptions already
                if transpathoptions.ageweightstrivial==0
                    AgeWeightsOld=AgeWeights;
                    AgeWeights=AgeWeights_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                if transpathoptions.trivialjequalonedist==0
                    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                %Get the current optimal policy
                optaprime=optaprimePath(:,:,:,tt);
                optaprime=gather(reshape(permute(optaprime(:,:,1:end-1),[1,3,2]),[1,N_a*(N_j-1)*N_z])); % swap order to j,z
                AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                AgentDistPath(:,tt+1)=AgentDist;
            end
        end

    end
else
    if N_z==0
        if simoptions.fastOLG==0
            %% fastOLG=0, no z, e
        else
            %% fastOLG=1, no z, e
        end

    else
        if simoptions.fastOLG==0
            %% fastOLG=0, z, e
            AgentDistPath=zeros(N_a*N_z*N_e,N_j,T,'gpuArray'); % Whether or not using simoptions.fastOLG
            AgentDistPath(:,:,1)=AgentDist_initial;

            AgentDist=AgentDist_initial;
            if transpathoptions.ageweightstrivial==0
                AgeWeights=AgeWeights_initial;
            end

            for tt=1:T-1
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on T, so is just in simoptions already
                if transpathoptions.epathtrivial==0
                    simoptions.pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
                end
                % transpathoptions.epathtrivial==1 % Does not depend on T

                if transpathoptions.ageweightstrivial==0
                    AgeWeightsOld=AgeWeights;
                    AgeWeights=AgeWeights_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                if transpathoptions.trivialjequalonedist==0
                    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                % Get the current optimal policy
                optaprime=optaprimePath(:,:,:,:,tt);
                AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_e_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
            end
        else
            %% fastOLG=1, z, e
            AgentDistPath=zeros(N_a*N_j*N_z,N_e,T,'gpuArray'); % Whether or not using simoptions.fastOLG
            AgentDistPath(:,:,1)=AgentDist_initial;

            AgentDist=AgentDist_initial;
            if transpathoptions.ageweightstrivial==0
                AgeWeights=AgeWeights_initial;
            end

            for tt=1:T-1
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                    if simoptions.fastOLG==1
                        pi_z_J_sim=gather(pi_z_J(1:end-1,:,:));
                        pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
                    end
                end
                % transpathoptions.zpathtrivial==1 % Does not depend on T, so is just in simoptions already
                if transpathoptions.epathtrivial==0
                    simoptions.pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
                    if simoptions.fastOLG==1
                        pi_e_J_sim=kron(kron(ones(N_z,1,'gpuArray'),gpuArray(simoptions.pi_e_J)'),ones(N_a,1,'gpuArray')); % (a,j,z)-by-e
                    end
                end
                % transpathoptions.epathtrivial==1 % Does not depend on T

                if transpathoptions.ageweightstrivial==0
                    AgeWeightsOld=AgeWeights;
                    AgeWeights=AgeWeights_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                if transpathoptions.trivialjequalonedist==0
                    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                end
                % Get the current optimal policy
                optaprime=optaprimePath(:,:,:,:,tt);
                optaprime=gather(reshape(permute(optaprime(:,:,:,1:end-1),[1,4,2,3]),[1,N_a*(N_j-1)*N_z*N_e])); % swap order to j,z
                AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_e_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
                AgentDistPath(:,:,tt+1)=AgentDist;
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