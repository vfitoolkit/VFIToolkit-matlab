function varargout=TransitionPath_Case1_FHorz(PricePathOld, ParamPath, T, V_final, AgentDist_initial, jequalOneDist, n_d, n_a, n_z, N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 
%
% PricePathOld is a structure with fields names being the Prices and each field containing a T-by-1 path.
% ParamPath is a structure with fields names being the parameter names of those parameters which change over the path and each field containing a T-by-1 path.
%
% Only works for v2, and only with GPU

% jequalOneDist can be a path

% Remark to self: No real need for T as input, as this is anyway the length
% of PricePathOld. Keeping it as helps double-check inputs are correct size.

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    % If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately); 
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=1000; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
    transpathoptions.verbose=0;
    transpathoptions.graphpricepath=0;
    transpathoptions.graphaggvarspath=0;
    transpathoptions.graphGEconditions=0;
    transpathoptions.historyofpricepath=0;
    transpathoptions.fastOLG=0; % fastOLG is done as (a,j,z), rather than standard (a,z,j)
    % transpathoptions.updateageweights % Don't declare if not being used
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(transpathoptions,'tolerance')
        transpathoptions.tolerance=10^(-4);
    end
    if ~isfield(transpathoptions,'parallel')
        transpathoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(transpathoptions,'GEnewprice')
        transpathoptions.GEnewprice=1; % 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately);
                                       % 1 is shooting algorithm, 
                                       % 2 is to do optimization routine with 'distance between old and new path'
                                       % 3 is just same as 0, but easier to set 
    end
    if ~isfield(transpathoptions,'oldpathweight')
        transpathoptions.oldpathweight=0.9;
        % Note that when using transpathoptions.GEnewprice==3
        % Implicitly it is setting transpathoptions.oldpathweight=0
        % because the user anyway has to specify them as part of setup
    end
    if ~isfield(transpathoptions,'weightscheme')
        transpathoptions.weightscheme=1;
    end
    if ~isfield(transpathoptions,'Ttheta')
        transpathoptions.Ttheta=1;
    end
    if ~isfield(transpathoptions,'maxiter')
        transpathoptions.maxiter=1000;
    end
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
    if ~isfield(transpathoptions,'graphpricepath')
        transpathoptions.graphpricepath=0;
    end
    if ~isfield(transpathoptions,'graphaggvarspath')
        transpathoptions.graphaggvarspath=0;
    end
    if ~isfield(transpathoptions,'graphGEconditions')
        transpathoptions.graphGEconditions=0;
    end
    if ~isfield(transpathoptions,'historyofpricepath')
        transpathoptions.historyofpricepath=0;
    end
    if ~isfield(transpathoptions,'fastOLG')
        transpathoptions.fastOLG=0; % fastOLG is done as (a,j,z), rather than standard (a,z,j)
    end
    % transpathoptions.updateageweights %Don't declare if not being used
end



%% Note: Internally PricePath is matrix of size T-by-'number of prices', similarly for ParamPath
% PricePath is matrix of size T-by-'number of prices'.
% Actually, some of those prices may be 1-by-N_j, so is more subtle than this.
PricePathNames=fieldnames(PricePathOld);
PricePathStruct=PricePathOld; 
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for pp=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{pp});
    tempsize=size(temp);
    PricePathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
    if ~any(PricePathSizeVec(pp)==[1,N_j])
        error(['PricePath for ', PricePathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T)'])
    end
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePathOld=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for pp=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{pp}),1)==T
        PricePathOld(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=PricePathStruct.(PricePathNames{pp});
    else % Need to transpose
        PricePathOld(:,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp))=PricePathStruct.(PricePathNames{pp})';
    end
end
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices may be 1-by-N_j, so is more subtle than this.
ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for pp=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{pp});
    tempsize=size(temp);
    ParamPathSizeVec(pp)=tempsize(tempsize~=T); % Get the dimension which is not T
    if ~any(ParamPathSizeVec(pp)==[1,N_j])
        error(['ParamPath for ', ParamPathNames{pp}, ' appears to be the wrong size (should be 1-by-T or N_j-by-T)'])
    end
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for pp=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{pp}),1)==T
        ParamPath(:,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=ParamPathStruct.(ParamPathNames{pp});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp))=ParamPathStruct.(ParamPathNames{pp})';
    end
end

PricePath=struct();

if transpathoptions.verbose>1
    PricePathNames
    ParamPathNames
end


%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.divideandconquer=0;
    vfoptions.gridinterplayer=0;
    vfoptions.parallel=transpathoptions.parallel;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences='None';
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
    vfoptions.endotype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(vfoptions,'divideandconquer')
        vfoptions.divideandconquer=0;
    elseif vfoptions.divideandconquer==1
        if ~isfield(vfoptions,'level1n')
            vfoptions.level1n=11;
        end
    end
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('You have vfoptions.gridinterplayer, so must also set vfoptions.ngridinterp')
        end
    end

    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    end
    if ~isfield(vfoptions,'lowmemory')
        vfoptions.lowmemory=0;
    end
    if ~isfield(vfoptions,'verbose')
        vfoptions.verbose=0;
    end
    if ~isfield(vfoptions,'exoticpreferences')
        vfoptions.exoticpreferences='None';
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
    end
    if ~isfield(vfoptions,'endotype')
        vfoptions.endotype=0;
    end
end



%% Check which simoptions have been used, set all others to defaults 
if transpathoptions.fastOLG==1
    simoptions.fastOLG=1;
end
if exist('simoptions','var')==0
    simoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.fastOLG=1;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=transpathoptions.parallel;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'fastOLG')
        simoptions.fastOLG=1;
    end
end

%% Check some inputs
if isstruct(GeneralEqmEqns)
    if length(PricePathNames)~=length(fieldnames(GeneralEqmEqns))
        error('Initial PricePath contains less variables than GeneralEqmEqns (structure) \n')
    end
else
    if length(PricePathNames)~=length(GeneralEqmEqns)
        error('Initial PricePath contains less variables than GeneralEqmEqns')
    end
end

%%
N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
if isfield(vfoptions, 'n_e')
    n_e=vfoptions.n_e;
else
    n_e=0;
end
N_e=prod(n_e);

%% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
PricePathOld=gpuArray(PricePathOld);
V_final=gpuArray(V_final);
% Tan improvement means we want agent dist on cpu
AgentDist_initial=gather(AgentDist_initial);


%% Handle ReturnFn and FnsToEvaluate structures
l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a);
l_aprime=l_a;
l_z=length(n_z);
if N_z==0
    l_z=0;
end
l_a_temp=l_a;
l_z_temp=l_z;
if max(vfoptions.endotype)==1
    l_a_temp=l_a-sum(vfoptions.endotype);
    l_z_temp=l_z+sum(vfoptions.endotype);
end
if N_e==0
    l_e=0;
else
    l_e=length(vfoptions.n_e);
end

% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a_temp+l_a_temp+l_z_temp)
    ReturnFnParamNames={temp{l_d+l_aprime+l_a_temp+l_z_temp+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end


%% Set up exogenous shock processes
[z_gridvals_J, pi_z_J, pi_z_J_sim, e_gridvals_J, pi_e_J, pi_e_J_sim, transpathoptions, simoptions]=ExogShockSetup_TPath_FHorz(n_z,z_grid,pi_z,N_a,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptions,4);
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


l_ze=l_z+l_e;
if l_z>0 && l_e>0
    N_ze=N_z*N_e;
    if transpathoptions.fastOLG==0
        ze_gridvals_J=[repmat(z_gridvals_J,N_e,1),repelem(e_gridvals_J,N_z,1)];
    elseif transpathoptions.fastOLG==1
        ze_gridvals_J=zeros(N_j,N_ze,l_ze,'gpuArray');
        ze_gridvals_J(:,:,1:l_z)=repmat(z_gridvals_J,1,N_e,1);
        ze_gridvals_J(:,:,l_z+1:end)=repmat(squeeze(e_gridvals_J),1,N_z,1);
    end
end

if transpathoptions.fastOLG==0
    if N_z>0 && N_e>0
        ze_gridvals_J=permute(ze_gridvals_J,[3,1,2]); % [N_j,N_z,l_z] need this for fastOLG agent dist (it is not used for value fn, so can overwrite)    
    elseif N_z>0
        z_gridvals_J_fastOLG=permute(z_gridvals_J,[3,1,2]); % [N_j,N_z,l_z] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
    elseif N_e>0
        e_gridvals_J_fastOLG=permute(e_gridvals_J,[3,1,2]); % [N_j,N_e,l_e] need this for fastOLG agent dist, but need the standard still for the value fn without fastOLG
    end
else
    if N_z>0 && N_e==0
        z_gridvals_J_fastOLG=[];
    elseif N_z==0 && N_e>0
        e_gridvals_J_fastOLG=[];
    end
end


%% Setup for V_final
% Note: I keep Policy as having a first dimension (even if it is just 1)
if N_e==0
    if N_z==0
        V_final=reshape(V_final,[N_a,N_j]);
    else
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_z,N_j]);
        else % vfoptions.fastOLG==1
            V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_j]),[1,3,2]),[N_a*N_j,N_z]);
        end
    end
else
    if N_z==0
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_e,N_j]);
        else % vfoptions.fastOLG==1
            V_final=reshape(permute(reshape(V_final,[N_a,N_e,N_j]),[1,3,2]),[N_a*N_j,N_e]);
        end
    else
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_z,N_e,N_j]);
        else % vfoptions.fastOLG==1
            V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_e,N_j]),[1,4,2,3]),[N_a*N_j,N_z,N_e]);
        end
    end
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


% Turn AgeWeights_T into appropriate size so that we can always just do AgentDist.*AgeWeights
% Currently it is N_j-by-T
if simoptions.fastOLG==0
    if N_e==0
        if N_z==0
            AgeWeights_T=repelem(shiftdim(AgeWeights_T,-1),N_a,1,1); % [N_a,N_j,T]
        else
            AgeWeights_T=repelem(shiftdim(AgeWeights_T,-1),N_a*N_z,1,1); % [N_a*N_z,N_j,T]
        end
    else % N_e>0
        if N_z==0
            AgeWeights_T=repelem(shiftdim(AgeWeights_T,-1),N_a*N_e,1,1); % [N_a*N_e,N_j,T]
        else
            AgeWeights_T=repelem(shiftdim(AgeWeights_T,-1),N_a*N_z*N_e,1,1); % [N_a*N_z*N_e,N_j,T]
        end
    end
elseif simoptions.fastOLG==1
    if N_e==0
        if N_z==0
            AgeWeights_T=repelem(AgeWeights_T,N_a,1); % [N_a*N_j,T]
        else
            AgeWeights_T=repmat(repelem(AgeWeights_T,N_a,1),N_z,1); % [N_a*N_j*N_z,T]
        end
    else % N_e>0
        if N_z==0
            AgeWeights_T=repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,N_e); % [N_a*N_j,N_e,T]
        else
            AgeWeights_T=repmat(repelem(reshape(AgeWeights_T,[N_j,1,T]),N_a,1),N_z,N_e); % [N_a*N_j*N_z,N_e,T]
        end
    end
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

%% Remove the age weights and do all the iterations. Only put the age weights back in when performing FnsToEvaluate (faster as saves putting weights in and then removing them T times)
% Weights are all in AgeWeights_T
if N_e==0
    if N_z==0
        if simoptions.fastOLG==0
            AgentDist_initial=AgentDist_initial./AgeWeights_initial; % remove age weights
        elseif simoptions.fastOLG==1
            AgentDist_initial=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
        end
    else % N_e>0
        if simoptions.fastOLG==0
            AgentDist_initial=AgentDist_initial./AgeWeights_initial; % remove age weights
        elseif simoptions.fastOLG==1
            AgentDist_initial=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
        end
    end
else % N_e>0
    if N_z==0
        if simoptions.fastOLG==0
            AgentDist_initial=AgentDist_initial./AgeWeights_initial; % remove age weights
        elseif simoptions.fastOLG==1
            AgentDist_initial=AgentDist_initial./repelem(AgeWeights_initial',N_a,1); % remove age weights
        end
    else % N_e>0
        if simoptions.fastOLG==0
            AgentDist_initial=AgentDist_initial./AgeWeights_initial; % remove age weights
        elseif simoptions.fastOLG==1
            AgentDist_initial=AgentDist_initial./repmat(repelem(AgeWeights_initial',N_a,1),N_z,1); % remove age weights
        end
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


%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_d+l_a+l_a+l_z+l_e)
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
end
FnsToEvaluate=FnsToEvaluate2;
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;
simoptions.AggVarNames=AggVarNames;


%% Set up some things for the FnsToEvaluate with fastOLG
a_gridvals=CreateGridvals(n_a,a_grid,1); % a_grivdals is [N_a,l_a]
if N_d==0
    aprime_gridvals=a_gridvals;
else
    d_gridvals=CreateGridvals(n_d,d_grid,1);
    aprime_gridvals=a_gridvals;
end


%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);

GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now: 
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(ff).Names contains the names


%% Set up GEnewprice==3 (if relevant)
if transpathoptions.GEnewprice==3
    transpathoptions.weightscheme=0;
    
    if isstruct(GeneralEqmEqns) 
        % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
        % Is same as order of fields in GeneralEqmEqns
        % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
        temp=transpathoptions.GEnewprice3.howtoupdate;
        GEeqnNames=fieldnames(GeneralEqmEqns);
        for ii=1:length(GEeqnNames)
            for jj=1:size(temp,1)
                if strcmp(temp{jj,1},GEeqnNames{ii}) % Names match
                    transpathoptions.GEnewprice3.howtoupdate{ii,1}=temp{jj,1};
                    transpathoptions.GEnewprice3.howtoupdate{ii,2}=temp{jj,2};
                    transpathoptions.GEnewprice3.howtoupdate{ii,3}=temp{jj,3};
                    transpathoptions.GEnewprice3.howtoupdate{ii,4}=temp{jj,4};
                end
            end
        end
        nGeneralEqmEqns=length(GEeqnNames);
    else
        nGeneralEqmEqns=length(GeneralEqmEqns);
    end
    transpathoptions.GEnewprice3.add=[transpathoptions.GEnewprice3.howtoupdate{:,3}];
    transpathoptions.GEnewprice3.factor=[transpathoptions.GEnewprice3.howtoupdate{:,4}];
    transpathoptions.GEnewprice3.keepold=ones(size(transpathoptions.GEnewprice3.factor));
    transpathoptions.GEnewprice3.keepold=ones(size(transpathoptions.GEnewprice3.factor));
    tempweight=transpathoptions.oldpathweight;
    transpathoptions.oldpathweight=zeros(size(transpathoptions.GEnewprice3.factor));
    for ii=1:length(transpathoptions.GEnewprice3.factor)
        if transpathoptions.GEnewprice3.factor(ii)==Inf
            transpathoptions.GEnewprice3.factor(ii)=1;
            transpathoptions.GEnewprice3.keepold(ii)=0;
            transpathoptions.oldpathweight(ii)=tempweight;
        end
    end
    if size(transpathoptions.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns % Note: inputs were already testted 
        % do nothing, this is how things should be
    else
        error('transpathoptions.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (number of rows is different to the number of GeneralEqmEqns fields) \n')
    end
    transpathoptions.GEnewprice3.permute=zeros(size(transpathoptions.GEnewprice3.howtoupdate,1),1);
    for ii=1:size(transpathoptions.GEnewprice3.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(PricePathNames)
            if strcmp(transpathoptions.GEnewprice3.howtoupdate{ii,2},PricePathNames{jj})
                transpathoptions.GEnewprice3.permute(ii)=jj;
            end
        end
    end
    if isfield(transpathoptions,'updateaccuracycutoff')==0
        transpathoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end


%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames,ParamPathNames);
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1paramNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[]; % I cannot remember what this was even for (how is it different rom tplus1priceNames??)
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for ii=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{ii})
        end
    end
end
use_tminus1params=0;
if ~isempty(tminus1paramNames)
    use_tminus1params=1;
    for ii=1:length(tminus1paramNames)
        if ~isfield(transpathoptions.initialvalues,tminus1paramNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1paramNames{ii})
        end
    end
end
use_tminus1AggVars=0;
if ~isempty(tminus1AggVarsNames)
    use_tminus1AggVars=1;
    for ii=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{ii})
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{ii})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

if transpathoptions.verbose>1
    use_tplus1price
    use_tminus1price
    use_tminus1params
    use_tminus1AggVars
    % tplus1pricePathkk
end





%%
if transpathoptions.verbose==1
    transpathoptions
end

% if transpathoptions.verbose==1
%     DiscountFactorParamNames
%     ReturnFnParamNames
%     ParamPathNames
%     PricePathNames
% end

l_p=length(PricePathNames);


%% Shooting algorithm
if transpathoptions.GEnewprice~=2
    if vfoptions.gridinterplayer==0
        if transpathoptions.fastOLG==0
            if vfoptions.divideandconquer==1
                error('Cannot yet use vfoptions.divideandconquer=1 with transpathoptions.fastOLG=0 (only implemented for fastOLG=1; tell me if you need this)')
            end
            if N_d==0
                if N_z==0
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_nod_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist ,n_a,N_j, N_a, l_a, aprime_gridvals,a_gridvals, a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_nod_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_e,N_j, N_a,N_e, l_a,l_e, aprime_gridvals,a_gridvals,a_grid,e_gridvals_J,e_gridvals_J_fastOLG, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    end
                else
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_nod(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,N_j, N_a,N_z, l_a,l_z, aprime_gridvals,a_gridvals,a_grid,z_gridvals_J,z_gridvals_J_fastOLG, pi_z_J,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_nod_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,n_e,N_j, N_a,N_z,N_e,N_ze, l_a,l_z,l_e,l_ze, aprime_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J,pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    end
                end
            else % N_d>0
                if N_z==0
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,N_j, N_d,N_a, l_d,l_a, d_gridvals,aprime_gridvals,a_gridvals, a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_e,N_j, N_d,N_a,N_e, l_d,l_a,l_e, d_gridvals,aprime_gridvals,a_gridvals,a_grid,e_gridvals_J,e_gridvals_J_fastOLG, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    end
                else
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,N_j, N_d,N_a,N_z, l_d,l_a,l_z, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals_J,z_gridvals_J_fastOLG, pi_z_J,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_z,N_e,N_ze, l_d,l_a,l_z,l_e,l_ze, d_gridvals,aprime_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J,pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                    end
                end
            end
        else % use fastOLG setting
            if N_d==0
                if N_z==0
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_nod_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,N_j, N_a, l_a, aprime_gridvals,a_gridvals,a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_nod_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_e,N_j, N_a,N_e, l_a,l_e, aprime_gridvals,a_gridvals,a_grid,e_gridvals_J, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                else
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_nod(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,N_j, N_a,N_z, l_a,l_z, aprime_gridvals,a_gridvals,a_grid,z_gridvals_J, pi_z_J,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else % use fastOLG setting
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_nod_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,n_e,N_j, N_a,N_z,N_e,N_ze, l_a,l_z,l_e,l_ze, aprime_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J, pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                end
            else % N_d>0
                if N_z==0
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,N_j, N_d,N_a, l_d,l_a, d_gridvals,aprime_gridvals,a_gridvals,d_grid,a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_e,N_j, N_d,N_a,N_e, l_d,l_a,l_e, d_gridvals,aprime_gridvals,a_gridvals,d_grid,a_grid,e_gridvals_J, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                else
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,N_j, N_d,N_a,N_z, l_d,l_a,l_z, d_gridvals,aprime_gridvals,a_gridvals,d_grid,a_grid,z_gridvals_J, pi_z_J, pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else % use fastOLG setting
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_z,N_e,N_ze, l_d,l_a,l_z,l_e,l_ze, d_gridvals,aprime_gridvals,a_gridvals,d_grid,a_grid,z_gridvals_J,e_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                end
            end
        end
    elseif vfoptions.gridinterplayer==1
        % Grid interpolation
        % vfoptions.ngridinterp=9;
        n2short=vfoptions.ngridinterp; % number of (evenly spaced) points to put between each grid point (not counting the two points themselves)
        n2long=vfoptions.ngridinterp*2+3; % total number of aprime points we end up looking at in second layer
        aprimefine_grid=interp1(1:1:N_a,a_grid,linspace(1,N_a,N_a+(N_a-1)*n2short))';
        n2aprime=length(aprimefine_grid);
        % Note: internally the VFI commands recreate these, the copies here are used for the AggVars
        aprimefine_gridvals=CreateGridvals(n2aprime,aprimefine_grid,1);
        
        if transpathoptions.fastOLG==0
            error('Cannot yet use vfoptions.gridinterplayer=1 with transpathoptions.fastOLG=0 (only implemented for fastOLG=1; tell me if you need this)')
            % NOTE TO SELF: THE VFI COMMANDS ARE WHAT NEEDS CREATING
            % 
            %
            % if N_d==0
            %     if N_z==0
            %         if N_e==0
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_nod_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist ,n_a,N_j, N_a, l_a, aprimefine_gridvals,a_gridvals, a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         else
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_nod_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_e,N_j, N_a,N_e, l_a,l_e, aprimefine_gridvals,a_gridvals,a_grid,e_gridvals_J,e_gridvals_J_fastOLG, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         end
            %     else
            %         if N_e==0
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_nod(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,N_j, N_a,N_z, l_a,l_z, aprimefine_gridvals,a_gridvals,a_grid,z_gridvals_J,z_gridvals_J_fastOLG, pi_z_J,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         else
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_nod_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,n_e,N_j, N_a,N_z,N_e,N_ze, l_a,l_z,l_e,l_ze, aprimefine_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J,pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         end
            %     end
            % else % N_d>0
            %     if N_z==0
            %         if N_e==0
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,N_j, N_d,N_a, l_d,l_a, d_gridvals,aprimefine_gridvals,a_gridvals, a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         else
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_e,N_j, N_d,N_a,N_e, l_d,l_a,l_e, d_gridvals,aprimefine_gridvals,a_gridvals,a_grid,e_gridvals_J,e_gridvals_J_fastOLG, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         end
            %     else
            %         if N_e==0
            %             PricePathOld=TransitionPath_FHorz_shooting_GI(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,N_j, N_d,N_a,N_z, l_d,l_a,l_z, d_gridvals,aprimefine_gridvals,a_gridvals,a_grid,z_gridvals_J,z_gridvals_J_fastOLG, pi_z_J,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         else
            %             PricePathOld=TransitionPath_FHorz_shooting_GI_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_z,N_e,N_ze, l_d,l_a,l_z,l_e,l_ze, d_gridvals,aprimefine_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J,pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
            %         end
            %     end
            % end
        else % use fastOLG setting
            if N_d==0
                if N_z==0
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_nod_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_a,N_j, N_a, l_a, aprimefine_gridvals,a_gridvals,a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_nod_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_a,n_e,N_j, N_a,N_e, l_a,l_e, aprimefine_gridvals,a_gridvals,a_grid,e_gridvals_J, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                else
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_nod(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_a,n_z,N_j, N_a,N_z, l_a,l_z, aprimefine_gridvals,a_gridvals,a_grid,z_gridvals_J, pi_z_J,pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else % use fastOLG setting
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_nod_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_a,n_z,n_e,N_j, N_a,N_z,N_e,N_ze, l_a,l_z,l_e,l_ze, aprimefine_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J, pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                end
            else % N_d>0
                if N_z==0
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_d,n_a,N_j, N_d,N_a, l_d,l_a, d_gridvals,aprimefine_gridvals,a_gridvals,d_grid,a_grid, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_d,n_a,n_e,N_j, N_d,N_a,N_e, l_d,l_a,l_e, d_gridvals,aprimefine_gridvals,a_gridvals,d_grid,a_grid,e_gridvals_J, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                else
                    if N_e==0
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_d,n_a,n_z,N_j, N_d,N_a,N_z, l_d,l_a,l_z, d_gridvals,aprimefine_gridvals,a_gridvals,d_grid,a_grid,z_gridvals_J, pi_z_J, pi_z_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    else % use fastOLG setting
                        [PricePath,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_z,N_e,N_ze, l_d,l_a,l_z,l_e,l_ze, d_gridvals,aprimefine_gridvals,a_gridvals,d_grid,a_grid,z_gridvals_J,e_gridvals_J,e_gridvals_J, pi_z_J, pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                    end
                end
            end
        end
    end
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
    end

    if nargout==1
        varargout={PricePath};
    elseif nargout==2
        varargout={PricePath,GEcondnPath};
    end

    return
end


%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
if transpathoptions.GEnewprice==2

    PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector for optimization algorithms. Additionally, provides a double check on sizes.

    % I HAVEN'T GOTTEN THIS TO WORK WELL ENOUGH THAT I AM COMFORTABLE LEAVING IT ENABLED
    if transpathoptions.GEnewprice==2 % Function minimization
        error('transpathoptions.GEnewprice==2 not currently enabled')
        %         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
    end
    % I WANT TO DO THIS WITH lsqnonlin()

    % if transpathoptions.GEnewprice2algo==0
    % [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
    % else
    %     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
    % end



    PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use optimization algorithms)
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii)'; % Output as 1-by-T
    end

    if nargout==1
        varargout={PricePath};
    elseif nargout==2
        varargout={PricePath,GEcondnPath};
    end

end

end
