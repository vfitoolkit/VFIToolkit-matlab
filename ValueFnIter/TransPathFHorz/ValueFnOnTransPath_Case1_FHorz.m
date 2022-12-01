function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% transpathoptions, vfoptions are optional inputs

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    if exist('vfoptions','var')==1 % If vfoptions.exoticpreferences, then set transpathoptions to the same
        if isfield(vfoptions,'exoticpreferences')
            transpathoptions.exoticpreferences=vfoptions.exoticpreferences;
        else
            transpathoptions.exoticpreferences='None';
        end
    else
        transpathoptions.exoticpreferences='None';
    end
    transpathoptions.verbose=0;
    transpathoptions.fastOLG=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences='None';
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
    if isfield(transpathoptions,'fastOLG')==0
        transpathoptions.fastOLG=0;
    end
end

%% Check which vfoptions have been used, set all others to defaults 
if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences=transpathoptions.exoticpreferences;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    vfoptions.exoticpreferences=transpathoptions.exoticpreferences; % Note that if vfoptions.exoticpreferences exists then it has already been used to set transpathoptions.exoticpreferences anyway.
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
end


%% Note: Internally PricePath is matrix of size T-by-'number of prices'.
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

% % The outputted VPath and PolicyPath are T-1 periods long (periods 0 (before the reforms are announced) & T are the initial and final values; they are not created by this command and instead can be used to provide double-checks of the output (the T-1 and the final should be identical if convergence has occoured).
% if n_d(1)==0
%     PolicyPath=zeros([length(n_d),n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1
% else
%     PolicyPath=zeros([length(n_d)+length(n_a),n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1
% end
% VPath=zeros([n_a,n_z,N_j,T-1],'gpuArray'); %Periods 1 to T-1

% This code will work for all transition paths except those that involve a
% change in the transition matrix pi_z.

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePath

N_z=prod(n_z);
if N_z==0
    [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz_noz(PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, Policy_final, Parameters, n_d, n_a, N_j, d_grid, a_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions);
    return
end
    

%% Create ReturnFnParamNames
l_d=0;
if ~isempty(n_d)
    if n_d(1)~=0
        l_d=length(n_d);
    end
end
l_a=length(n_a);
l_z=length(n_z);

temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a+l_a+l_z)
    ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end

%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

transpathoptions.zpathprecomputed=0;
if isfield(vfoptions,'pi_z_J')
    transpathoptions.zpathprecomputed=1;
    transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
elseif isfield(vfoptions,'ExogShockFn')
    % Note: If ExogShockFn depends on the path, it must be done via a parameter
    % that depends on the path (i.e., via ParamPath or PricePath)
    overlap=0;
    for ii=1:length(vfoptions.ExogShockFnParamNames)
        if strcmp(vfoptions.ExogShockFnParamNames{ii},PricePathNames)
            overlap=1;
        end
    end
    if overlap==0
        transpathoptions.zpathprecomputed=1;
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
                if isfield(vfoptions,'ExogShockFnParamNames')
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
                    if isfield(vfoptions,'ExogShockFnParamNames')
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
if ~strcmp(transpathoptions.exoticpreferences,'None')
    error('Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
end

if transpathoptions.parallel~=2
    error('Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
%     PricePath=gpuArray(PricePath);
end
unkronoptions.parallel=2;

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePath,2);

if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    if N_d>0
        d_grid=gather(d_grid);
    end
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

if transpathoptions.verbose==1
    transpathoptions
end

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z,N_j]);
if N_d>0
    Policy=zeros(2,N_a,N_z,N_j,'gpuArray');
else
    Policy=zeros(N_a,N_z,N_j,'gpuArray');
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%%
% I DONT THINK THAT _tminus1 and/or _tplus1 variables ARE USED WITH Value fn. 
% AT LEAST NOT IN ANY EXAMPLES I HAVE COME ACROSS. AS SUCH THEY ARE NOT IMPLEMENTED HERE.

%%
VKronPath=zeros(N_a,N_z,N_j,T);
VKronPath(:,:,:,T)=V_final;

if N_d>0
    PolicyIndexesPath=zeros(2,N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
else
    PolicyIndexesPath=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
    PolicyIndexesPath(:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for ttr=1:T-1 %so t=T-i

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end
    
    [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The VKron input is next period value fn, the VKron output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')
    
    if N_d>0
        PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
    else
        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    end
    VKronPath(:,:,:,T-ttr)=V;
    Vnext=V;
end

%% Unkron to get into the shape for output
VPath=reshape(VKronPath,[n_a,n_z,N_j,T]);
PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyIndexesPath, n_d, n_a, n_z, N_j,T,vfoptions);



end