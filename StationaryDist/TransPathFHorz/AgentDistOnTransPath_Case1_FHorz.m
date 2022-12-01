function AgentDistPath=AgentDistOnTransPath_Case1_FHorz(AgentDist_initial,PricePath, ParamPath, PolicyPath, AgeWeightsParamNames,n_d,n_a,n_z,N_j,pi_z, T,Parameters, transpathoptions, simoptions)
n_e=0; % NOT YET IMPLEMENTED FOR TRANSITION PATHS

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
end

%% Check which simoptions have been used, set all others to defaults 
if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.verbose=0;
    simoptions.ncores=feature('numcores'); % Number of CPU cores
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check simoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        simoptions.ncores=feature('numcores'); % Number of CPU cores
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end

%% Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices are 1-by-N_j, so is more subtle than this.
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

PricePathNames
ParamPathNames

if N_z==0
    AgentDistPath=AgentDistOnTransPath_Case1_FHorz_noz(AgentDist_initial,PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, PolicyPath, AgeWeightsParamNames,n_d,n_a,N_j, T,Parameters, transpathoptions, simoptions);
    return
end

%%
if transpathoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
end

%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

transpathoptions.zpathprecomputed=0;
if isfield(simoptions,'pi_z_J')
    transpathoptions.zpathprecomputed=1;
    transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
elseif isfield(simoptions,'ExogShockFn')
    % Note: If ExogShockFn depends on the path, it must be done via a parameter
    % that depends on the path (i.e., via ParamPath or PricePath)
    overlap=0;
    for ii=1:length(simoptions.ExogShockFnParamNames)
        if strcmp(simoptions.ExogShockFnParamNames{ii},PricePathNames)
            overlap=1;
        end
    end
    if overlap==0
        transpathoptions.zpathprecomputed=1;
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
            z_grid_J=zeros(N_z,N_j,'gpuArray');
            for jj=1:N_j
                if isfield(simoptions,'ExogShockFnParamNames')
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                else
                    [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                end
                pi_z_J(:,:,jj)=gpuArray(pi_z);
                z_grid_J(:,jj)=gpuArray(z_grid);
            end
            % Now store them in simoptions and simoptions
            simoptions.pi_z_J=pi_z_J;
            simoptions.z_grid_J=z_grid_J;
            simoptions.pi_z_J=pi_z_J;
            simoptions.z_grid_J=z_grid_J;
        elseif transpathoptions.zpathtrivial==0
            % z_grid_J and/or pi_z_J varies along the transition path (but only depending on ParamPath, not PricePath
            transpathoptions.pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
            transpathoptions.z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
            pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
            z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
            for tt=1:T
                for ii=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                end
                % Note, we know the PricePath is irrelevant for the current purpose
                for jj=1:N_j
                    if isfield(simoptions,'ExogShockFnParamNames')
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    else
                        [z_grid,pi_z]=simoptions.ExogShockFn(jj);
                    end
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    z_grid_J(:,jj)=gpuArray(z_grid);
                end
                transpathoptions.pi_z_J_T(:,:,:,tt)=pi_z_J;
                transpathoptions.z_grid_J_T(:,:,tt)=z_grid_J;
            end
        end
    end
end

%%
PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j,T);
AgentDistPath=zeros(N_a*N_z,N_j,T);

% Now we have the full PolicyIndexesPath, we go forward in time from 1
% to T using the policies to update the agents distribution generating anew price path

% Call AgentDist the current periods distn
AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]);
AgentDist=AgentDist_initial;
AgentDistPath(:,:,1)=AgentDist_initial;
for tt=1:T-1
    
    %Get the current optimal policy
    if N_d>0
        Policy=PolicyPath(:,:,:,:,tt);
    else
        Policy=PolicyPath(:,:,:,tt);
    end
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    
    if transpathoptions.zpathprecomputed==1
        if transpathoptions.zpathtrivial==1
            simoptions.pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
            simoptions.z_grid_J=transpathoptions.z_grid_J_T(:,:,tt);
        end
        % transpathoptions.zpathtrivial==0 % Does not depend on T, so is just in simoptions already
    end
    % transpathoptions.zpathprecomputed==0 % Depends on the price path  parameters, so just have to use simoptions.ExogShockFn within StationaryDist and FnEvaluation command
    
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
    
    AgentDistPath(:,:,tt+1)=AgentDist;

end

AgentDistPath=reshape(AgentDistPath,[n_a,n_z,N_j,T]);





end