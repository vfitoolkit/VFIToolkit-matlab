function AgentDistPath=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial, PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions)
% Note: PricePath is not used, it is just there for legacy compatibility

N_a=prod(n_a);
N_z=prod(n_z);

N_e=0;
if exist('simoptions','var')
    if isfield(simoptions,'n_e')
        n_e=simoptions.n_e;
        N_e=prod(n_e);
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
    simoptions.nsims=10^4;
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.fastOLG=1;
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'nsims')
        simoptions.nsims=10^4;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'ncores')
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions,'fastOLG')
        simoptions.fastOLG=1;
    end
end

%%
% Note: No PricePath as all we need in terms of parameters is the agent
% distribution
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



%% Check the sizes of some of the inputs
if isempty(n_d)
    N_d=0;
else
    N_d=prod(n_d);
end
N_z=prod(n_z);
N_a=prod(n_a);


%%
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
AgentDist_initial=gpuArray(AgentDist_initial);


%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

if N_z>0
    % transpathoptions.zpathprecomputed=1; % Hardcoded: I do not presently allow for z to be determined by an ExogShockFn which includes parameters from PricePath

    if ismatrix(pi_z) % (z,zprime)
        % Just a basic pi_z, but convert to pi_z_J for codes
        % z_grid_J=z_grid.*ones(1,N_j);
        pi_z_J=pi_z.*ones(1,1,N_j);
        transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
        if isfield(simoptions,'pi_z_J') % This is just legacy, intend to depreciate it
            z_grid_J=simoptions.z_grid_J;
            pi_z_J=simoptions.pi_z_J;
        end
    elseif ndims(pi_z)==3 % (z,zprime,j)
        % Inputs are already z_grid_J and pi_z_J
        % z_grid_J=z_grid;
        pi_z_J=pi_z;
        transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
    elseif ndims(pi_z)==4 % (z,zprime,j,t)
        transpathoptions.zpathtrivial=0; % z_grid_J and pi_z_J var over the path
        transpathoptions.pi_z_J_T=pi_z;
        % transpathoptions.z_grid_J_T=z_grid;
        % z_grid_J=z_grid(:,:,1); % placeholder
        pi_z_J=pi_z(:,:,:,1); % placeholder
    end
    % These inputs get overwritten if using simoptions.ExogShockFn
    if isfield(simoptions,'ExogShockFn')
        % Note: If ExogShockFn depends on the path, it must be done via a parameter
        % that depends on the path (i.e., via ParamPath or PricePath)
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
        overlap=0;
        for ii=1:length(simoptions.ExogShockFnParamNames)
            if strcmp(simoptions.ExogShockFnParamNames{ii},PricePathNames)
                overlap=1;
            end
        end
        if overlap==1
            error('It is not allowed for z to be determined by an ExogShockFn which includes parameters from PricePath')
        else % overlap==0
            % If ExogShockFn does not depend on any of the prices (in PricePath), then
            % we can simply create it now rather than within each 'subfn' or 'p_grid'

            % Check if it depends on the ParamPath
            transpathoptions.zpathtrivial=1;
            for ii=1:length(simoptions.ExogShockFnParamNames)
                if strcmp(simoptions.ExogShockFnParamNames{ii},ParamPathNames)
                    transpathoptions.zpathtrivial=0;
                end
            end
            if transpathoptions.zpathtrivial==1
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                % z_grid_J=zeros(N_z,N_j,'gpuArray');
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    % z_grid_J(:,jj)=gpuArray(z_grid);
                end
                % Now store them in simoptions and simoptions
                simoptions.pi_z_J=pi_z_J;
                % simoptions.z_grid_J=z_grid_J;
                simoptions.pi_z_J=pi_z_J;
                % simoptions.z_grid_J=z_grid_J;
            elseif transpathoptions.zpathtrivial==0
                % z_grid_J and/or pi_z_J varies along the transition path (but only depending on ParamPath, not PricePath
                transpathoptions.pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
                % transpathoptions.z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                % z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
                for ttr=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    for jj=1:N_j
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                        pi_z_J(:,:,jj)=gpuArray(pi_z);
                        % z_grid_J(:,jj)=gpuArray(z_grid);
                    end
                    transpathoptions.pi_z_J_T(:,:,:,ttr)=pi_z_J;
                    % transpathoptions.z_grid_J_T(:,:,ttr)=z_grid_J;
                end
            end
        end
    end
    
end

%% If using e variables do the same for e as we just did for z
if N_e>0
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

    transpathoptions.epathprecomputed=0;
    if isfield(simoptions,'pi_e_J')
        e_grid_J=simoptions.e_grid_J;
        pi_e_J=simoptions.pi_e_J;
        transpathoptions.epathprecomputed=1;
        transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
    elseif isfield(simoptions,'EiidShockFn')
        % Note: If EiidShockFn depends on the path, it must be done via a parameter
        % that depends on the path (i.e., via ParamPath or PricePath)
        simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
        overlap=0;
        for ii=1:length(simoptions.EiidShockFnParamNames)
            if strcmp(simoptions.EiidShockFnParamNames{ii},PricePathNames)
                overlap=1;
            end
        end
        if overlap==0
            transpathoptions.epathprecomputed=1;
            % If ExogShockFn does not depend on any of the prices (in PricePath), then
            % we can simply create it now rather than within each 'subfn' or 'p_grid'

            % Check if it depends on the ParamPath
            transpathoptions.epathtrivial=1;
            for ii=1:length(simoptions.EiidShockFnParamNames)
                if strcmp(simoptions.EiidShockFnParamNames{ii},ParamPathNames)
                    transpathoptions.epathtrivial=0;
                end
            end
            if transpathoptions.epathtrivial==1
                pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
                e_grid_J=zeros(N_e,N_j,'gpuArray');
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,pi_e]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
                    pi_e_J(:,jj)=gpuArray(pi_e);
                    e_grid_J(:,jj)=gpuArray(e_grid);
                end
                % Now store them in simoptions and simoptions
                simoptions.pi_e_J=pi_e_J;
                simoptions.e_grid_J=e_grid_J;
                simoptions.pi_e_J=pi_e_J;
                simoptions.e_grid_J=e_grid_J;
            elseif transpathoptions.epathtrivial==0
                % e_grid_J and/or pi_e_J varies along the transition path (but only depending on ParamPath, not PricePath)
                transpathoptions.pi_e_J_T=zeros(N_e,N_e,N_j,T,'gpuArray');
                transpathoptions.e_grid_J_T=zeros(sum(n_e),N_j,T,'gpuArray');
                pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
                e_grid_J=zeros(sum(n_e),N_j,'gpuArray');
                for ttr=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    for jj=1:N_j
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                        end
                        [e_grid,pi_e]=simoptions.ExogShockFn(EiidShockFnParamsCell{:});
                        pi_e_J(:,jj)=gpuArray(pi_e);
                        e_grid_J(:,jj)=gpuArray(e_grid);
                    end
                    transpathoptions.pi_e_J_T(:,:,ttr)=pi_e_J;
                    transpathoptions.e_grid_J_T(:,:,ttr)=e_grid_J;
                end
            end
        end
    end
end

%%
if transpathoptions.verbose==1
    transpathoptions
end

if transpathoptions.verbose==1
    ParamPathNames
    PricePathNames
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


%% Setup for various objects
if N_z==0
    PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j, T);
else
    PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j, T);
end

if N_z==0
    AgentDist_initial=reshape(AgentDist_initial,[N_a,N_j]); % if simoptions.fastOLG==0
    AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
    if simoptions.fastOLG==1
        AgentDist_initial=reshape(AgentDist_initial,[N_a*N_j,1]);
        % Note: do the double reshape() as cannot get AgeWeights_initial from the final shape
        AgeWeights_initial=kron(AgeWeights_initial',ones(N_a,1,'gpuArray'));
    end
    if transpathoptions.ageweightstrivial==0
        % AgeWeights_T is N_j-by-T (or if simoptions.fastOLG=1, then N_a*N_j-by-T )
        if simoptions.fastOLG==1
            AgeWeights_T=kron(AgeWeights,ones(N_a,1,'gpuArray')); % simoptions.fastOLG=1 so this is (a,j)-by-1
        else
            AgeWeights_T=AgeWeights;
        end
    elseif transpathoptions.ageweightstrivial==1
        if max(abs(AgeWeights_initial-AgeWeights))>10^(-13)
            error('AgeWeights differs from the weights implicit in the initial agent distribution')
        end
        AgeWeights=AgeWeights_initial;
        AgeWeightsOld=AgeWeights;
    end
else
    AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]); % if simoptions.fastOLG==0
    AgeWeights_initial=sum(AgentDist_initial,1); % [1,N_j]
    if transpathoptions.ageweightstrivial==0
        % AgeWeights_T is N_j-by-T (or if simoptions.fastOLG=1, then N_a*N_j*N_z-by-T )
        AgeWeights_T=AgeWeights; % Has already been adjusted based on simoptions.fastOLG
    elseif transpathoptions.ageweightstrivial==1
        if max(abs(AgeWeights_initial-AgeWeights))>10^(-13)
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
        % Similarly, we want pi_z_J to be (j,z,z'), but we need to keep the standard pi_z_J for the value function
        pi_z_J_sim=gather(reshape(permute(pi_z_J(:,:,1:N_j-1),[3,1,2]),[(N_j-1)*N_z,N_z])); % For agent dist we want it to be (j,z,z')
        if transpathoptions.ageweightstrivial==0
            AgeWeights_T=kron(ones(N_z,1,'gpuArray'),kron(AgeWeights_T',ones(N_a,1,'gpuArray'))); % Vectorized as N_a*N_j*N_z-by-T
        else % AgeWeights do not change over time, so just set them all to same as AgeWeights_initial
            AgeWeights=AgeWeights_initial;
            AgeWeightsOld=AgeWeights;
        end

        % Precompute some things needed for fastOLG agent dist iteration
        exceptlastj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(0:1:N_j-2)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        exceptfirstj=kron(ones(1,(N_j-1)*N_z),1:1:N_a)+kron(kron(ones(1,N_z),N_a*(1:1:N_j-1)),ones(1,N_a))+kron(N_a*N_j*(0:1:N_z-1),ones(1,N_a*(N_j-1))); % Note: there is one use of N_j which is because we want to index AgentDist
        II1=repmat(1:1:(N_j-1)*N_z,1,N_z);
        II2=repmat(1:1:(N_j-1),1,N_z*N_z)+repelem((N_j-1)*(0:1:N_z-1),1,N_z*(N_j-1));
        pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
    end
end

%%
if N_z==0
    %%
    if simoptions.fastOLG==0
        AgentDistPath=zeros(N_a,N_j,T,'gpuArray');
        AgentDistPath(:,:,1)=AgentDist_initial;
    else
        AgentDistPath=zeros(N_a*N_j,T,'gpuArray');
        AgentDistPath(:,1)=AgentDist_initial;
    end

    AgentDist=AgentDist_initial;
    if transpathoptions.ageweightstrivial==0
        AgeWeights=AgeWeights_initial;
    end
    for tt=1:T-1
        %Get the current optimal policy
        if N_d>0
            Policy=PolicyPath(:,:,:,tt);
        else
            Policy=PolicyPath(:,:,tt);
        end
        if transpathoptions.ageweightstrivial==0
            AgeWeightsOld=AgeWeights;
            AgeWeights=AgeWeights_T(:,tt);
        end
        if simoptions.fastOLG==0
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_j);
            AgentDistPath(:,:,tt+1)=AgentDist;
        else % simoptions.fastOLG==1
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_noz_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_j);
            AgentDistPath(:,tt+1)=AgentDist;
        end
    end
else
    %%
    if simoptions.fastOLG==0
        AgentDistPath=zeros(N_a*N_z,N_j,T,'gpuArray');
        AgentDistPath(:,:,1)=AgentDist_initial;
    else
        AgentDistPath=zeros(N_a*N_j*N_z,T,'gpuArray');
        AgentDistPath(:,1)=AgentDist_initial;
    end

    AgentDist=AgentDist_initial;
    if transpathoptions.ageweightstrivial==0
        AgeWeights=AgeWeights_initial;
    end
    for tt=1:T-1

        %Get the current optimal policy
        if N_d>0
            Policy=PolicyPath(:,:,:,:,tt);
        else
            Policy=PolicyPath(:,:,:,tt);
        end
        if transpathoptions.zpathtrivial==0
            pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
            if simoptions.fastOLG==1
                pi_z_J_sim=gather(pi_z_J(1:end-1,:,:));
                pi_z_J_sim=sparse(II1,II2,pi_z_J_sim,(N_j-1)*N_z,(N_j-1)*N_z);
            end
        end

        if transpathoptions.ageweightstrivial==0
            AgeWeightsOld=AgeWeights;
            AgeWeights=AgeWeights_T(:,tt);
        end

        if simoptions.fastOLG==0
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_Iteration_raw(AgentDist,AgeWeights,AgeWeightsOld,Policy,N_d,N_a,N_z,N_j,pi_z_J);
            AgentDistPath(:,:,tt+1)=AgentDist;
        else % simoptions.fastOLG==1
            if N_d==0
                optaprime=gather(reshape(permute(Policy(:,:,1:end-1),[1,3,2]),[1,N_a*(N_j-1)*N_z])); % swap order to j,z
            else
                optaprime=gather(reshape(permute(Policy(2,:,:,1:end-1),[1,2,4,3]),[1,N_a*(N_j-1)*N_z])); % swap order to j,z
            end
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep_IterFast_raw(AgentDist,AgeWeights,AgeWeightsOld,optaprime,N_a,N_z,N_j,pi_z_J_sim,exceptlastj,exceptfirstj);
            AgentDistPath(:,tt+1)=AgentDist;
        end
    end
end


%%
if N_z==0
    AgentDistPath=reshape(AgentDistPath,[n_a,N_j,T]);
else
    if simoptions.fastOLG==1
        AgentDistPath=permute(reshape(AgentDistPath,[N_a,N_j,N_z,T]),[1,3,2,4]);
    end
    AgentDistPath=reshape(AgentDistPath,[n_a,n_z,N_j,T]);
end

end