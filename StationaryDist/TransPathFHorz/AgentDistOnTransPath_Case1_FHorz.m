function AgentDistPath=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial, jequaloneDist, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions)
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

if N_z==0
    if N_e==0
        N_ze=0;
    else
        N_ze=N_e;
    end
else
    if N_e==0
        N_ze=N_z;
    else
        N_ze=N_z*N_e;
    end
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
    simoptions.fastOLG=1; % parallel over j, faster but uses more memory
    simoptions.gridinterplayer=0;
    % Model setup
    simoptions.experienceasset=0;
else
    % Check simoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
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
    % Model setup
    if ~isfield(simoptions,'experienceasset')
        simoptions.experienceasset=0;
    end
end

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_FHorz_StructToMatrix(PricePath,ParamPath,N_j,T);

%%
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
AgentDist_initial=gpuArray(AgentDist_initial);


%% Set up exogenous shock processes
[~, pi_z_J, pi_z_J_sim, ~, pi_e_J, pi_e_J_sim, ~, transpathoptions, simoptions]=ExogShockSetup_TPath_FHorz(n_z,[],pi_z,N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,2);
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
            AgentDist_initial=permute(AgentDist_initial,[1,4,2,3]);
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


%%
N_probs=1; % Not using N_probs
if simoptions.gridinterplayer==1
    N_probs=N_probs*2;
end
if simoptions.experienceasset==1
    N_probs=N_probs*2;
end

l_a=length(n_a);
if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_aprime=l_a;
if simoptions.experienceasset==1
    l_aprime=l_aprime-1;
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
        PolicyPath=reshape(PolicyPath, [size(PolicyPath,1),N_a,N_z*N_e,N_j,T]);
    end
end

n_aprime=n_a;
if simoptions.experienceasset==1
    if ~isfield(simoptions,'l_dexperienceasset')
        simoptions.l_dexperienceasset=1;
    end
    n_aprime=n_aprime(1:end-1);
    n_a1=n_a(1:end-1);
    n_a2=n_a(end);
    N_a1=prod(n_a1);
    n_d2=n_d(end-simoptions.l_dexperienceasset+1:end);

    a2_grid=simoptions.a_grid(sum(n_a1)+1:end);

    if ~isfield(simoptions,'aprimeFn')
        error('To use an experience asset you must define simoptions.aprimeFn')
    end
    
    % aprimeFnParamNames in same fashion
    l_d2=length(n_d2);
    l_a2=length(n_a2);
    temp=getAnonymousFnInputNames(simoptions.aprimeFn);
    if length(temp)>(l_d2+l_a2)
        aprimeFnParamNames={temp{l_d2+l_a2+1:end}}; % the first inputs will always be (d2,a2)
    else
        aprimeFnParamNames={};
    end
    
    a2primeIndexesPath=zeros(N_a,N_ze,N_j-1,T,'gpuArray');
    a2primeProbsPath=zeros(N_a,N_ze,N_j-1,T,'gpuArray');
    
    whichisdforexpasset=length(n_d)-simoptions.l_dexperienceasset+1:length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
    if N_e==0 && N_z==0
        for tt=1:T
            aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters,aprimeFnParamNames,N_j);
            % [N_j,number of params]

            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_J(PolicyPath(:,:,tt),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_ze, N_j, simoptions.d_grid, a2_grid, aprimeFnParamsVec);
            % Note: a2primeIndexes and a2primeProbs are both [N_a,N_j]
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,:,tt)=a2primeIndexes(:,1:end-1);
            a2primeProbsPath(:,:,tt)=a2primeProbs(:,1:end-1);
        end
    else
        for tt=1:T
            aprimeFnParamsVec=CreateAgeMatrixFromParams(Parameters,aprimeFnParamNames,N_j);
            % [N_j,number of params]
            
            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_J(PolicyPath(:,:,:,tt),simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_ze, N_j, simoptions.d_grid, a2_grid, aprimeFnParamsVec);
            % Note: a2primeIndexes and a2primeProbs are both [N_a,N_z,N_j]
            % Note: a2primeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the a2primeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            a2primeIndexesPath(:,:,:,tt)=a2primeIndexes(:,:,1:end-1);
            a2primeProbsPath(:,:,:,tt)=a2primeProbs(:,:,1:end-1);
        end
    end
    
    if N_e==0 && N_z==0
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_j-1,1,T]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,2,1);
        a2primeIndexesPath(:,:,2,:)=a2primeIndexesPath(:,:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_j-1,1,T]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,1,2,1);
        a2primeProbsPath(:,:,2,:)=1-a2primeProbsPath(:,:,2,:); % upper prob
        if simoptions.fastOLG==1
            a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*(N_j-1),2,T]);
            a2primeProbsPath=reshape(a2primeProbsPath,[N_a*(N_j-1),2,T]);
        end
    else
        a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a,N_ze,N_j-1,1,T]);
        a2primeIndexesPath=repmat(a2primeIndexesPath,1,1,1,2,1);
        a2primeIndexesPath(:,:,:,2,:)=a2primeIndexesPath(:,:,:,2,:)+1; % upper index
        a2primeProbsPath=reshape(a2primeProbsPath,[N_a,N_ze,N_j-1,1,T]);
        a2primeProbsPath=repmat(a2primeProbsPath,1,1,1,2,1);
        a2primeProbsPath(:,:,:,2,:)=1-a2primeProbsPath(:,:,:,2,:); % upper prob
        if simoptions.fastOLG==0
            a2primeIndexesPath=reshape(a2primeIndexesPath,[N_a*N_ze,N_j-1,2,T]);
            a2primeProbsPath=reshape(a2primeProbsPath,[N_a*N_ze,N_j-1,2,T]);
        elseif simoptions.fastOLG==1
            a2primeIndexesPath=reshape(permute(a2primeIndexesPath,[1,3,2,4,5]),[N_a*(N_j-1)*N_ze,2,T]);
            a2primeProbsPath=reshape(permute(a2primeProbsPath,[1,3,2,4,5]),[N_a*(N_j-1)*N_ze,2,T]);
        end
    end
end

if N_e==0
    if N_z==0 % no z, no e
        % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_aprime)
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:),[N_a,N_j-1,T]); % aprime index
        elseif length(n_aprime)==2
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,1:N_j-1,:)-1),[N_a,N_j-1,T]);
        elseif length(n_aprime)==3
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,1:N_j-1,:)-1),[N_a,N_j-1,T]);
        elseif length(n_aprime)==4
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*n_aprime(3)*(PolicyPath(l_d+4,:,1:N_j-1,:)-1),[N_a,N_j-1,T]);
        end
        if simoptions.fastOLG==0
            % PolicyaprimePath=PolicyaprimePath;
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_j-1,T]),[1,2,3]),[N_a*(N_j-1),T]);
            PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyPath(l_d+l_aprime+1,:,1:N_j-1,:),[1,N_a,N_j-1,T]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_j,T]
                L2index=reshape(permute(L2index,[2,3,1,4]),[N_a*(N_j-1),1,T]);
                PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T]); % reinterpret this as lower grid index
                PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                PolicyaprimePath=reshape(PolicyaprimePath,[N_a,(N_j-1),1,T])+N_a1*(a2primeIndexesPath-1);
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(N_a1*(a2primeIndexesPath-1),1,2,1);
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    else % z, no e
        % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_aprime)
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_z,N_j-1,T]); % aprime index
        elseif length(n_aprime)==2
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T]);
        elseif length(n_aprime)==3
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T]);
        elseif length(n_aprime)==4
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*n_aprime(3)*(PolicyPath(l_d+4,:,:,1:N_j-1,:)-1),[N_a*N_z,N_j-1,T]);
        end
        if simoptions.fastOLG==0
            PolicyaprimezPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z,N_j-1,T]),[1,3,2,4]),[N_a*(N_j-1)*N_z,T]);
            PolicyaprimejzPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_z,N_j-1,T]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_z,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z,1,T]);
                PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z,1,T]); % reinterpret this as lower grid index
                PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,(N_j-1),1,T])+N_a1*(a2primeIndexesPath-1);
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(N_a1*(a2primeIndexesPath-1),1,2,1);
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    end
else
    if N_z==0 % no z, e
        % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_aprime)
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_e,N_j-1,T]); % aprime index
        elseif length(n_aprime)==2
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T]);
        elseif length(n_aprime)==3
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T]);
        elseif length(n_aprime)==4
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*n_aprime(3)*(PolicyPath(l_d+4,:,:,1:N_j-1,:)-1),[N_a*N_e,N_j-1,T]);
        end
        if simoptions.fastOLG==0
            % PolicyaprimePath=PolicyaprimePath;
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_e,N_j-1,T]),[1,3,2,4]),[N_a*(N_j-1)*N_e,T]);
            PolicyaprimejPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1),N_e,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_e,N_j-1,T]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_e,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_e,1,T]);
                PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1)*N_e,1,T]); % reinterpret this as lower grid index
                PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                PolicyaprimePath=reshape(PolicyaprimePath,[N_a*N_e,(N_j-1),1,T])+N_a1*(a2primeIndexesPath-1);
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                PolicyaprimejPath=repmat(PolicyaprimejPath,1,2,1)+repelem(N_a1*(a2primeIndexesPath-1),1,2,1);
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    else % z, e
        % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
        % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
        % When using grid interpolation layer also PolicyProbsPath
        if isscalar(n_aprime)
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:),[N_a*N_z*N_e,N_j-1,T]); % aprime index
        elseif length(n_aprime)==2
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T]);
        elseif length(n_aprime)==3
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T]);
        elseif length(n_aprime)==4
            PolicyaprimePath=reshape(PolicyPath(l_d+1,:,:,1:N_j-1,:)+n_aprime(1)*(PolicyPath(l_d+2,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*(PolicyPath(l_d+3,:,:,1:N_j-1,:)-1)+n_aprime(1)*n_aprime(2)*n_aprime(3)*(PolicyPath(l_d+4,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T]);
        end
        if simoptions.fastOLG==0
            PolicyaprimezPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1),N_e,1);
        elseif simoptions.fastOLG==1
            PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z*N_e,N_j-1,T]),[1,3,2,4]),[N_a*(N_j-1)*N_z*N_e,T]);
            PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
            if simoptions.gridinterplayer==1
                L2index=reshape(PolicyPath(l_d+l_aprime+1,:,:,1:N_j-1,:),[1,N_a,N_z*N_e,N_j-1,T]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_z,N_e,N_j,T]
                L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z*N_e,1,T]);
                PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T]); % reinterpret this as lower grid index
                PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
                PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
                PolicyProbsPath(:,2,:)=L2index; % L2 index
                PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
                PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
            end
        end
        if simoptions.experienceasset==1
            if simoptions.fastOLG==0
                PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z*N_e,(N_j-1),1,T])+N_a1*(a2primeIndexesPath-1);
                PolicyProbsPath=a2primeProbsPath;
            elseif simoptions.fastOLG==1
                PolicyaprimejzPath=repmat(PolicyaprimejzPath,1,2,1)+repelem(N_a1*(a2primeIndexesPath-1),1,2,1);
                PolicyProbsPath=repmat(PolicyProbsPath,1,2,1).*repelem(a2primeProbsPath,1,2,1);
            end
        end
    end
end


%% Check if jequalOneDistPath is a path or not (and reshape appropriately)
jequaloneDist=gpuArray(jequaloneDist);
temp=size(jequaloneDist);
% Note: simoptions.fastOLG is handled via 'justfirstj', rather than via shape of jequalOneDist
if temp(end)==T % jequalOneDist depends on T
    transpathoptions.trivialjequalonedist=0;
    if N_z==0 && N_e==0
        jequaloneDist=reshape(jequaloneDist,[N_a,T]);
    else
        jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,T]);
    end
else
    transpathoptions.trivialjequalonedist=1;
    if N_z==0 && N_e==0
        jequaloneDist=reshape(jequaloneDist,[N_a,1]);
    else
        jequaloneDist=reshape(jequaloneDist,[N_a*N_ze,1]);
    end
end

if transpathoptions.trivialjequalonedist==0
    jequalOneDist_T=jequaloneDist;
    jequaloneDist=jequalOneDist_T(:,1);
end


%% Remove the age weights, do all the iterations, then put the age weights back in at the end. (faster as saves putting weights in and then removing them T times)

%% Because of the need to drop j=N_j (exceptlastj) before each agent dist iteration and then put j=1 (justfirstj) after each iteration I keep the AgentDist on gpu, and then use sparse(gather()) and gpuArray(full()) before and after

%% Do the AgentDistPath calculations
if N_probs==1 % Not using N_probs
    if N_e==0
        if N_z==0 % no z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, no e
                II1=1:1:N_a;
                II2=ones(N_a,1);
                AgentDistPath=zeros(N_a,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=PolicyaprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_raw(AgentDist,Policy_aprime,N_a,N_j,II1,II2,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, no z, no e
                II1=1:1:N_a*(N_j-1);
                II2=ones(N_a*(N_j-1),1);
                AgentDistPath=zeros(N_a*N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimej=PolicyaprimejPath(:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_raw(AgentDist,Policy_aprimej,N_a,N_j,II1,II2,jequaloneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(AgeWeights_T,N_a,1); % put in the age weights
            end

        else % z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, no e
                II1=1:1:N_a*N_z;
                II2=ones(N_a*N_z,1);
                AgentDistPath=zeros(N_a*N_z,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimez=PolicyaprimezPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_raw(AgentDist,Policy_aprimez,N_a,N_z,N_j,pi_z_J,II1,II2,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, z, no e
                % AgentDist is [N_a*N_j*N_z,1]
                II1=1:1:N_a*(N_j-1)*N_z;
                II2=ones(N_a*(N_j-1)*N_z,1);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
                AgentDistPath=zeros(N_a*N_j*N_z,T,'gpuArray');
                AgentDist=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimejz=PolicyaprimejzPath(:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_raw(AgentDist,Policy_aprimejz,N_a,N_z,N_j,pi_z_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequaloneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % put in the age weights
            end
        end
    else
        if N_z==0 % no z, e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, e
                II1=1:1:N_a*N_e;
                II2=ones(N_a*N_e,1);
                AgentDistPath=zeros(N_a*N_e,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=PolicyaprimePath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_e_raw(AgentDist,Policy_aprime,N_a,N_e,N_j,pi_e_J,II1,II2,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, no z, e
                % AgentDist is [N_a*N_j,N_e,1]
                II1=1:1:N_a*(N_j-1)*N_e;
                II2=ones(N_a*(N_j-1)*N_e,1);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
                AgentDistPath=zeros(N_a*N_j,N_e,T,'gpuArray');
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimej=PolicyaprimejPath(:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_e_raw(AgentDist,Policy_aprimej,N_a,N_e,N_j,pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1); % put in the age weights
            end

        else % z and e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, e
                II1=1:1:N_a*N_z*N_e;
                II2=ones(N_a*N_z*N_e,1);
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
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimez=PolicyaprimezPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_e_raw(AgentDist,Policy_aprimez,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,II1,II2,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, z, e
                II1=1:1:N_a*(N_j-1)*N_z*N_e;
                II2=ones(N_a*(N_j-1)*N_z*N_e,1);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a,1);
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
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimejz=PolicyaprimejzPath(:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,Policy_aprimejz,N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1),N_z,1); % put in the age weights
            end
        end
    end


else
    %% N_probs>1
    if N_e==0
        if N_z==0 % no z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, no e, N_probs
                II1=repelem((1:1:N_a)',1,N_probs);
                AgentDistPath=zeros(N_a,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=PolicyaprimePath(:,:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_noz_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_j,II1,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, no z, no e, N_probs
                II=repelem((1:1:N_a*(N_j-1))',1,N_probs);
                AgentDistPath=zeros(N_a*N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimej=PolicyaprimejPath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_raw(AgentDist,Policy_aprimej,PolicyProbs,N_a,N_j,II,jequaloneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(AgeWeights_T,N_a,1); % put in the age weights
            end

        else % z, no e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, no e, N_probs
                II1=repelem((1:1:N_a*N_z)',1,N_probs);
                AgentDistPath=zeros(N_a*N_z,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimez=PolicyaprimezPath(:,:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_raw(AgentDist,Policy_aprimez,PolicyProbs,N_a,N_z,N_j,pi_z_J,II1,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights
            else
                %% fastOLG=1, z, no e, N_probs
                % AgentDist is [N_a*N_j*N_z,1]
                II=repelem((1:1:N_a*(N_j-1)*N_z)',1,N_probs);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z,1)+repelem(N_a*N_j*(0:1:N_z-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z,1)+N_a*N_j*repelem((0:1:N_z-1)',N_a,1);
                AgentDistPath=zeros(N_a*N_j*N_z,T,'gpuArray');
                AgentDist=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
                AgentDistPath(:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.zpathtrivial==0
                        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimejz=PolicyaprimejzPath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_raw(AgentDist,Policy_aprimejz,PolicyProbs,N_a,N_z,N_j,pi_z_J_sim,II,exceptlastj,exceptfirstj,justfirstj,jequaloneDist);
                    AgentDistPath(:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % put in the age weights
            end

        end
    else
        if N_z==0 % no z, e
            if simoptions.fastOLG==0
                %% fastOLG=0, no z, e, N_probs
                II1=repelem((1:1:N_a*N_e)',1,N_probs);
                AgentDistPath=zeros(N_a*N_e,N_j,T,'gpuArray');
                AgentDist=AgentDist_initial./AgeWeights_initial; % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprime=PolicyaprimePath(:,:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_noz_e_raw(AgentDist,Policy_aprime,PolicyProbs,N_a,N_e,N_j,pi_e_J,II1,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights                  %% fastOLG=0, no z, e
            else
                %% fastOLG=1, no z, e, N_probs
                II=repelem((1:1:N_a*(N_j-1)*N_e)',1,N_probs);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_e,1)+repelem(N_a*N_j*(0:1:N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_e,1)+N_a*N_j*repelem((0:1:N_e-1)',N_a,1);
                AgentDistPath=zeros(N_a*N_j,N_e,T,'gpuArray'); % Whether or not using simoptions.fastOLG
                AgentDist=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
                AgentDistPath(:,:,1)=AgentDist;
                for tt=1:T-1
                    if transpathoptions.epathtrivial==0
                        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt);
                    end
                    if transpathoptions.trivialjequalonedist==0
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimej=PolicyaprimejPath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_e_raw(AgentDist,Policy_aprimej,PolicyProbs,N_a,N_e,N_j,pi_e_J_sim,II,exceptlastj,exceptfirstj,justfirstj,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1); % put in the age weights
            end

        else % z and e
            if simoptions.fastOLG==0
                %% fastOLG=0, z, e, N_probs
                II1=repelem((1:1:N_a*N_z*N_e)',1,N_probs);
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
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimez=PolicyaprimezPath(:,:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_Iteration_nProbs_e_raw(AgentDist,Policy_aprimez,PolicyProbs,N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,II1,jequaloneDist);
                    AgentDistPath(:,:,tt+1)=AgentDist;
                end
                AgentDistPath=AgentDistPath.*shiftdim(AgeWeights_T,-1); % put in the age weights 
            else
                %% fastOLG=1, z, e, N_probs
                II=repelem((1:1:N_a*(N_j-1)*N_z*N_e)',1,N_probs);
                exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
                justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
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
                        jequaloneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
                    end
                    % Get the current optimal policy
                    Policy_aprimejz=PolicyaprimejzPath(:,:,tt);
                    PolicyProbs=PolicyProbsPath(:,:,tt);
                    AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_e_raw(AgentDist,Policy_aprimejz,PolicyProbs,N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,II,exceptlastj,exceptfirstj,justfirstj,jequaloneDist);
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