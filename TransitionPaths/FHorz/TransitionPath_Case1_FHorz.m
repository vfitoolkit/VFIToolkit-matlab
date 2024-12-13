function PricePath=TransitionPath_Case1_FHorz(PricePathOld, ParamPath, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, n_z, N_j, d_grid,a_grid,z_grid, pi_z, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeightsParamNames, transpathoptions, simoptions, vfoptions)
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
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.GEnewprice=1; % 1 is shooting algorithm, 0 is that the GE should evaluate to zero and the 'new' is the old plus the "non-zero" (for each time period seperately); 
                                   % 2 is to do optimization routine with 'distance between old and new path', 3 is just same as 0, but easier to set up
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiter=500; % Based on personal experience anything that hasn't converged well before this is just hung-up on trying to get the 4th decimal place (typically because the number of grid points was not large enough to allow this level of accuracy).
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



%% Note: Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
% Actually, some of those prices are 1-by-N_j, so is more subtle than this.
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
    else
        if ~isfield(vfoptions,'level1n')
            vfoptions.level1n=11;
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
    if strcmp(vfoptions.exoticpreferences,'QuasiHyperbolic')
        if ~isfield(vfoptions,'quasi_hyperbolic')
            vfoptions.quasi_hyperbolic='Naive'; % This is the default, alternative is 'Sophisticated'.
        elseif ~strcmp(vfoptions.quasi_hyperbolic,'Naive') && ~strcmp(vfoptions.quasi_hyperbolic,'Sophisticated')
            error('When using Quasi-Hyperbolic discounting vfoptions.quasi_hyperbolic must be either Naive or Sophisticated ')
        end
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
    if isfield(vfoptions,'ExogShockFn')
        vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
    end
    if isfield(vfoptions,'EiidShockFn')
        vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
    end
end



%% Check which simoptions have been used, set all others to defaults 
if transpathoptions.fastOLG==1
    simoptions.fastOLG=1;
end
if exist('simoptions','var')==0
    simoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
    simoptions.fastOLG=1;
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-9);
    end
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=transpathoptions.parallel;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'iterate')
        simoptions.iterate=1;
    end
    if ~isfield(simoptions,'fastOLG')
        simoptions.fastOLG=1;
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

if N_d>0
    if size(d_grid)~=[N_d, 1]
        error('d_grid is not the correct shape (should be of size N_d-by-1) \n')
    end
end
if ~all(size(a_grid)==[N_a, 1])
    error('a_grid is not the correct shape (should be of size N_a-by-1) \n')
elseif N_z>0
    if ndims(z_grid)==2 % 2-dimensional
        if ~all(size(z_grid)==[N_z, 1])
            error('z_grid is not the correct shape (should be of size N_z-by-1) \n')
        elseif ~all(size(pi_z)==[N_z, N_z])
            error('pi_z is not the correct shape (should be of size N_z-by-N_z) \n')
        end
    else
        if ~all(size(z_grid)==[N_z, N_j])
            error('z_grid is not the correct shape (should be of size N_z-by-N_j) \n')
        elseif ~all(size(pi_z)==[N_z, N_z, N_j])
            error('pi_z is not the correct shape (should be of size N_z-by-N_z-by-N_j) \n')
        end
    end
end
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
% Make sure all the relevant inputs are GPU arrays (not standard arrays)
pi_z=gpuArray(pi_z);
if N_d>0
    d_grid=gpuArray(d_grid);
end
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
PricePathOld=gpuArray(PricePathOld);
V_final=gpuArray(V_final);
% Tan improvement means we want agent dist on cpu
AgentDist_init=gather(AgentDist_init);


%% Handle ReturnFn and FnsToEvaluate structures
l_d=length(n_d);
if n_d(1)==0
    l_d=0;
end
l_a=length(n_a);
l_z=length(n_z);
if n_z(1)==0
    l_z=0;
end
l_a_temp=l_a;
l_z_temp=l_z;
if max(vfoptions.endotype)==1
    l_a_temp=l_a-sum(vfoptions.endotype);
    l_z_temp=l_z+sum(vfoptions.endotype);
end
if ~isfield(vfoptions,'n_e')
    N_e=0;
    l_e=0;
else
    N_e=prod(vfoptions.n_e);
    if N_e==0
        l_e=0;
    else
        l_e=length(vfoptions.n_e);
    end
end
% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a_temp+l_a_temp+l_z_temp)
    ReturnFnParamNames={temp{l_d+l_a_temp+l_a_temp+l_z_temp+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end

if ~isstruct(FnsToEvaluate)
    error('Transition paths only work with version 2+ (FnsToEvaluate has to be a structure)')
end

N_z=prod(n_z);

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

%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

if N_z>0
    % transpathoptions.zpathprecomputed=1; % Hardcoded: I do not presently allow for z to be determined by an ExogShockFn which includes parameters from PricePath

    if ismatrix(pi_z) % (z,zprime)
        % Just a basic pi_z, but convert to pi_z_J for codes
        z_grid_J=z_grid.*ones(1,N_j,'gpuArray');
        pi_z_J=pi_z.*ones(1,1,N_j,'gpuArray');
        transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
        if isfield(vfoptions,'pi_z_J') % This is just legacy, intend to depreciate it
            z_grid_J=vfoptions.z_grid_J;
            pi_z_J=vfoptions.pi_z_J;
        end
    elseif ndims(pi_z)==3 % (z,zprime,j)
        % Inputs are already z_grid_J and pi_z_J
        z_grid_J=gpuArray(z_grid);
        pi_z_J=gpuArray(pi_z);
        transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
    elseif ndims(pi_z)==4 % (z,zprime,j,t)
        transpathoptions.zpathtrivial=0; % z_grid_J and pi_z_J var over the path
        transpathoptions.pi_z_J_T=pi_z;
        transpathoptions.z_grid_J_T=z_grid;
        z_grid_J=z_grid(:,:,1); % placeholder
        pi_z_J=pi_z(:,:,:,1); % placeholder
    end
    % These inputs get overwritten if using vfoptions.ExogShockFn
    if isfield(vfoptions,'ExogShockFn')
        % Note: If ExogShockFn depends on the path, it must be done via a parameter
        % that depends on the path (i.e., via ParamPath or PricePath)
        vfoptions.ExogShockFnParamNames=getAnonymousFnInputNames(vfoptions.ExogShockFn);
        overlap=0;
        for ii=1:length(vfoptions.ExogShockFnParamNames)
            if strcmp(vfoptions.ExogShockFnParamNames{ii},PricePathNames)
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
            for ii=1:length(vfoptions.ExogShockFnParamNames)
                if strcmp(vfoptions.ExogShockFnParamNames{ii},ParamPathNames)
                    transpathoptions.zpathtrivial=0;
                end
            end
            if transpathoptions.zpathtrivial==1
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                z_grid_J=zeros(N_z,N_j,'gpuArray');
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ii=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                    end
                    [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    pi_z_J(:,:,jj)=gpuArray(pi_z);
                    z_grid_J(:,jj)=gpuArray(z_grid);
                end
                % Now store them in vfoptions and simoptions
                vfoptions.pi_z_J=pi_z_J;
                vfoptions.z_grid_J=z_grid_J;
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
                        ExogShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.ExogShockFnParamNames,jj);
                        ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                        end
                        [z_grid,pi_z]=vfoptions.ExogShockFn(ExogShockFnParamsCell{:});
                        pi_z_J(:,:,jj)=gpuArray(pi_z);
                        z_grid_J(:,jj)=gpuArray(z_grid);
                    end
                    transpathoptions.pi_z_J_T(:,:,:,tt)=pi_z_J;
                    transpathoptions.z_grid_J_T(:,:,tt)=z_grid_J;
                end
            end
        end
    end

    % Transition path only ever uses z_gridvals_J, not z_grid_J
    z_gridvals_J=zeros(N_z,l_z,N_j,'gpuArray');
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid_J(:,jj),1);
    end
    
    if transpathoptions.fastOLG==1 % Reshape grid and transtion matrix for use with fastOLG
        z_gridvals_J=permute(z_gridvals_J,[3,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-N_z-by-l_z
        pi_z_J=permute(pi_z_J,[3,2,1]); % We want it to be (j,z',z) for value function 
        transpathoptions.pi_z_J_alt=permute(pi_z_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
        if transpathoptions.zpathtrivial==0
            temp=transpathoptions.z_grid_J_T;
            transpathoptions=rmfield(transpathoptions,'z_grid_J_T');
            transpathoptions.z_gridvals_J_T=zeros(N_z,l_z,N_j,T,'gpuArray');
            for tt=1:T
                for jj=1:N_j
                    z_gridvals_J(:,:,jj,tt)=CreateGridvals(n_z,temp(:,jj,tt),1);
                end
            end
            transpathoptions.z_gridvals_J_T=permute(transpathoptions.z_gridvals_J_T,[3,1,2,4]); % from [N_z,l_z,N_j,T] to [N_j,N_z,l_z,T]
            transpathoptions.pi_z_J_T=permute(transpathoptions.pi_z_J_T,[3,1,2,4]);  % We want it to be (j,z,z',t)
            transpathoptions.pi_z_J_T_alt=permute(transpathoptions.pi_z_J_T,[1,3,2,4]);  % We want it to be (j,z',z,t) [note, this permute is off the previous one]
        end
    end

    % z_gridvals_J is [N_z,l_z,N_j] if transpathoptions.fastOLG=0
    %              is [N_j,N_z,l_z] if transpathoptions.fastOLG=1
    % pi_z_J is [N_z,N_z,N_j]       if transpathoptions.fastOLG=0 (j,z,z')
    % pi_z_J is [N_j,N_z,N_z]       if transpathoptions.fastOLG=1 (j,z',z)
    % pi_z_J and z_gridvals_J are both gpuArrays
    z_gridvals_J=gpuArray(z_gridvals_J);
    pi_z_J=gpuArray(pi_z_J);
end

%% If using e variables do the same for e as we just did for z
if N_e>0
    n_e=vfoptions.n_e;
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

    transpathoptions.epathprecomputed=0;
    if isfield(vfoptions,'pi_e')
        e_grid_J=vfoptions.e_grid.*ones(1,N_j,'gpuArray');
        pi_e_J=vfoptions.pi_e.*ones(1,N_j,'gpuArray');
        transpathoptions.epathprecomputed=1;
        transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
    elseif isfield(vfoptions,'pi_e_J')
        e_grid_J=gpuArray(vfoptions.e_grid_J);
        pi_e_J=gpuArray(vfoptions.pi_e_J);
        transpathoptions.epathprecomputed=1;
        transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
    elseif isfield(vfoptions,'EiidShockFn')
        % Note: If EiidShockFn depends on the path, it must be done via a parameter
        % that depends on the path (i.e., via ParamPath or PricePath)
        vfoptions.EiidShockFnParamNames=getAnonymousFnInputNames(vfoptions.EiidShockFn);
        overlap=0;
        for ii=1:length(vfoptions.EiidShockFnParamNames)
            if strcmp(vfoptions.EiidShockFnParamNames{ii},PricePathNames)
                overlap=1;
            end
        end
        if overlap==0
            transpathoptions.epathprecomputed=1;
            % If ExogShockFn does not depend on any of the prices (in PricePath), then
            % we can simply create it now rather than within each 'subfn' or 'p_grid'

            % Check if it depends on the ParamPath
            transpathoptions.epathtrivial=1;
            for ii=1:length(vfoptions.EiidShockFnParamNames)
                if strcmp(vfoptions.EiidShockFnParamNames{ii},ParamPathNames)
                    transpathoptions.epathtrivial=0;
                end
            end
            if transpathoptions.epathtrivial==1
                pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
                e_grid_J=zeros(N_e,N_j,'gpuArray');
                for jj=1:N_j
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,pi_e]=vfoptions.EiidShockFn(EiidShockFnParamsCell{:});
                    pi_e_J(:,jj)=gpuArray(pi_e);
                    e_grid_J(:,jj)=gpuArray(e_grid);
                end
            elseif transpathoptions.epathtrivial==0
                % e_grid_J and/or pi_e_J varies along the transition path (but only depending on ParamPath, not PricePath)
                transpathoptions.pi_e_J_T=zeros(N_e,N_e,N_j,T,'gpuArray');
                transpathoptions.e_grid_J_T=zeros(sum(n_e),N_j,T,'gpuArray');
                pi_e_J=zeros(N_e,N_e,N_j,'gpuArray');
                e_grid_J=zeros(sum(n_e),N_j,'gpuArray');
                for tt=1:T
                    for ii=1:length(ParamPathNames)
                        Parameters.(ParamPathNames{ii})=ParamPathStruct.(ParamPathNames{ii});
                    end
                    % Note, we know the PricePath is irrelevant for the current purpose
                    for jj=1:N_j
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                        end
                        [e_grid,pi_e]=vfoptions.ExogShockFn(EiidShockFnParamsCell{:});
                        pi_e_J(:,jj)=gpuArray(pi_e);
                        e_grid_J(:,jj)=gpuArray(e_grid);
                    end
                    transpathoptions.pi_e_J_T(:,:,tt)=pi_e_J;
                    transpathoptions.e_grid_J_T(:,:,tt)=e_grid_J;
                end
            end
        end
    end

    % Transition path only ever uses e_gridvals_J, not e_grid_J
    e_gridvals_J=zeros(N_e,l_e,N_j,'gpuArray');
    for jj=1:N_j
        e_gridvals_J(:,:,jj)=CreateGridvals(n_e,e_grid_J(:,jj),1);
    end
    
    if transpathoptions.fastOLG==1 % Reshape grid and transtion matrix for use with fastOLG
        e_gridvals_J=permute(e_gridvals_J,[3,4,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLGe: (j,1,N_e,l_e)
        pi_e_J=reshape(kron(pi_e_J,ones(N_a,1,'gpuArray'))',[N_a*N_j,1,N_e]); % Give it the size required for fastOLG value function
        % transpathoptions.pi_e_J_alt=permute(pi_e_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
        if transpathoptions.epathtrivial==0
            temp=transpathoptions.e_grid_J_T;
            transpathoptions=rmfield(transpathoptions,'e_grid_J_T');
            transpathoptions.e_gridvals_J_T=zeros(N_e,l_e,N_j,T,'gpuArray');
            for tt=1:T
                for jj=1:N_j
                    e_gridvals_J(:,:,jj,tt)=CreateGridvals(n_e,temp(:,jj,tt),1);
                end
            end
            transpathoptions.e_gridvals_J_T=permute(transpathoptions.e_gridvals_J_T,[3,5,1,2,4]); % from (e,j,t) to (j,e,t) [second dimension is singular, this is how I want it for fastOLG value fn where first dim is j, then second is z (which is not relevant to e)]
            transpathoptions.pi_e_J_T=repelem(permute(transpathoptions.pi_e_J_T,[3,1,2,4]),N_a,1,1,1);  % We want it to be (a-j,1,e,t)
            transpathoptions.pi_e_J_sim_T=zeros(N_a*(N_j-1)*N_z,N_e,T,'gpuArray');
            for tt=1:T
                temp=reshape(transpathoptions.pi_e_J_T(:,:,:,tt),[N_a*N_j,N_e]); % transpathoptions.fastOLG means pi_e_J is [N_a*N_j,1,N_e]
                transpathoptions.pi_e_J_sim_T(:,:,tt)=kron(ones(N_z,1,'gpuArray'),gpuArray(temp(N_a+1:end,:))); 
            end
        end
    end


    vfoptions.e_grid_J=e_gridvals_J;
    vfoptions.pi_e_J=pi_e_J;
    simoptions.e_grid_J=e_gridvals_J;
    simoptions.pi_e_J=pi_e_J;


    % e_gridvals_J is [N_e,l_e,N_j]   if transpathoptions.fastOLG=0
    %              is [N_j,1,N_e,l_e] if transpathoptions.fastOLG=1
    % pi_e_J is [N_e,N_j]             if transpathoptions.fastOLG=0 (e,j)
    % pi_e_J is [N_a*N_j,1,N_e]       if transpathoptions.fastOLG=1 (a-j,1,e)
    % pi_e_J and e_gridvals_J are both gpuArrays
    e_gridvals_J=gpuArray(e_gridvals_J);
    pi_e_J=gpuArray(pi_e_J);
end


%% Check if jequalOneDistPath is a path or not (and reshape appropriately)
jequalOneDist=gpuArray(jequalOneDist);
temp=size(jequalOneDist);
if temp(end)==T % jequalOneDist depends on T
    transpathoptions.trivialjequalonedist=0;
    if N_z==0
        if N_e==0
            jequalOneDist=reshape(jequalOneDist,[N_a,T]);
        else
            if simoptions.fastOLG==0
                jequalOneDist=reshape(jequalOneDist,[N_a,N_e,T]);
            elseif simoptions.fastOLG==1
                jequalOneDist=reshape(jequalOneDist,[N_a*N_e,T]);            
            end
        end
    else
        if N_e==0
            if simoptions.fastOLG==0
                jequalOneDist=reshape(jequalOneDist,[N_a,N_z,T]);
            elseif simoptions.fastOLG==1
                jequalOneDist=reshape(jequalOneDist,[N_a*N_z,T]);            
            end
        else
            if simoptions.fastOLG==0
                jequalOneDist=reshape(jequalOneDist,[N_a,N_z,N_e,T]);
            elseif simoptions.fastOLG==1
                jequalOneDist=reshape(jequalOneDist,[N_a*N_z*N_e,T]);            
            end
        end
    end
else
    transpathoptions.trivialjequalonedist=1;
    if N_z==0
        if N_e==0
            jequalOneDist=reshape(jequalOneDist,[N_a,1]);
        else
            if simoptions.fastOLG==0
                jequalOneDist=reshape(jequalOneDist,[N_a,N_e]);
            elseif simoptions.fastOLG==1
                jequalOneDist=reshape(jequalOneDist,[N_a*N_e,1]);
            end
        end
    else
        if N_e==0
            if simoptions.fastOLG==0
                jequalOneDist=reshape(jequalOneDist,[N_a,N_z]);
            elseif simoptions.fastOLG==1
                jequalOneDist=reshape(jequalOneDist,[N_a*N_z,1]);
            end
        else
            if simoptions.fastOLG==0
                jequalOneDist=reshape(jequalOneDist,[N_a,N_z,N_e]);
            elseif simoptions.fastOLG==1
                jequalOneDist=reshape(jequalOneDist,[N_a*N_z*N_e,1]);
            end        
        end
    end
end


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

if transpathoptions.GEnewprice~=2
    if transpathoptions.parallel==2
        if transpathoptions.fastOLG==0
            if N_z==0
                if N_e==0
                    PricePathOld=TransitionPath_Case1_FHorz_shooting_noz(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, N_j, d_grid,a_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeights, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                else
                    error('e without z not yet implemented for TPath with FHorz')
                end
            else
                if N_e==0
                    PricePathOld=TransitionPath_Case1_FHorz_shooting(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, n_z, N_j, d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeights, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                else
                    PricePathOld=TransitionPath_Case1_FHorz_shooting_e(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, n_z, n_e, N_j, d_grid,a_grid,z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeights, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions,transpathoptions);
                end
            end
        else % use fastOLG setting
            if N_z==0
                if N_e==0
                    PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG_noz(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, N_j, d_grid,a_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeights, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                else
                    error('e without z not yet implemented for TPath with FHorz')
                end
            else
                if N_e==0
                    PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, n_z, N_j, d_grid,a_grid,z_gridvals_J, pi_z_J, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeights, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                else % use fastOLG setting
                    PricePathOld=TransitionPath_Case1_FHorz_shooting_fastOLG_e(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_init, jequalOneDist, n_d, n_a, n_z, n_e, N_j, d_grid,a_grid,z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, AgeWeights, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions,transpathoptions);
                end
            end
        end
    else
        error('VFI Toolkit does not offer transition path without gpu. Would be too slow to be useful.')
    end
    % Switch the solution into structure for output.
    for ii=1:length(PricePathNames)
        PricePath.(PricePathNames{ii})=PricePathOld(:,ii);
    end
    return
end


%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
l_p=size(PricePathOld,2);
if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.

% I HAVEN'T GOTTEN THIS TO WORK WELL ENOUGH THAT I AM COMFORTABLE LEAVING IT ENABLED
if transpathoptions.GEnewprice==2 % Function minimization
    error('transpathoptions.GEnewprice==2 not currently enabled')
%         GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames, vfoptions, simoptions,transpathoptions);
end

% if transpathoptions.GEnewprice2algo==0
% [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% else
%     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% end

if transpathoptions.parallel==2
    PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)
else
    PricePath=gather(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)    
end

for ii=1:length(PricePathNames)
    PricePath.(PricePathNames{ii})=PricePathOld(:,ii)'; % Output as 1-by-T
end


end