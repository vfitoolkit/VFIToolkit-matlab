function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, d_grid, a_grid,z_grid, pi_z, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
% transpathoptions, vfoptions are optional inputs

%%
% I DONT THINK THAT _tminus1 and/or _tplus1 variables ARE USED WITH Value fn. 
% AT LEAST NOT IN ANY EXAMPLES I HAVE COME ACROSS. AS SUCH THEY ARE NOT IMPLEMENTED HERE.

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
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
    vfoptions.divideandconquer=0;
    vfoptions.parallel=1+(gpuDeviceCount>0);
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences='None';
    vfoptions.endotype=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
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
        vfoptions.parallel=1+(gpuDeviceCount>0);
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if ~isfield(vfoptions,'returnmatrix')
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
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
    if ~isfield(vfoptions,'endotype')
        vfoptions.endotype=0;
    end
    if ~isfield(vfoptions,'polindorval')
        vfoptions.polindorval=1;
    end
    if ~isfield(vfoptions,'policy_forceintegertype')
        vfoptions.policy_forceintegertype=0;
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
if N_d>0
    d_grid=gpuArray(d_grid);
end
a_grid=gpuArray(a_grid);
z_grid=gpuArray(z_grid);
V_final=gpuArray(V_final);


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
elseif vfoptions.n_e(1)==0
    N_e=0;
    l_e=0;
else
    N_e=prod(vfoptions.n_e);
    l_e=length(vfoptions.n_e);
end
% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a_temp+l_a_temp+l_z_temp)
    ReturnFnParamNames={temp{l_d+l_a_temp+l_a_temp+l_z_temp+l_e+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end


%% Check if z_grid and/or pi_z depend on prices. If not then create pi_z_J and z_grid_J for the entire transition before we start
% If 'exogenous shock fn' is used, then precompute it to save evaluating it numerous times
% Check if using 'exogenous shock fn' (exogenous state has a grid and transition matrix that depends on age)

if N_z>0
    % transpathoptions.zpathprecomputed=1; % Hardcoded: I do not presently allow for z to be determined by an ExogShockFn which includes parameters from PricePath

    if ismatrix(pi_z) % (z,zprime)
        % Just a basic pi_z, but convert to pi_z_J for codes
        z_grid_J=z_grid.*ones(1,N_j);
        pi_z_J=pi_z.*ones(1,1,N_j);
        transpathoptions.zpathtrivial=1; % z_grid_J and pi_z_J are not varying over the path
        if isfield(vfoptions,'pi_z_J') % This is just legacy, intend to depreciate it
            z_grid_J=vfoptions.z_grid_J;
            pi_z_J=vfoptions.pi_z_J;
        end
    elseif ndims(pi_z)==3 % (z,zprime,j)
        % Inputs are already z_grid_J and pi_z_J
        z_grid_J=z_grid;
        pi_z_J=pi_z;
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
                % simoptions.pi_z_J=pi_z_J;
                % simoptions.z_grid_J=z_grid_J;
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
        z_gridvals_J=permute(z_gridvals_J,[3,1,2]); % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-N_z
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
            transpathoptions.z_gridvals_J_T=permute(transpathoptions.z_gridvals_J_T,[3,1,2,4]); % from (j,z,t) to (z,j,t)
            transpathoptions.pi_z_J_T=permute(transpathoptions.pi_z_J_T,[3,1,2,4]);  % We want it to be (j,z,z',t)
            transpathoptions.pi_z_J_T_alt=permute(transpathoptions.pi_z_J_T,[1,3,2,4]);  % We want it to be (j,z',z,t) [note, this permute is off the previous one]
        end
    end
end

%% If using e variables do the same for e as we just did for z
if N_e>0
    n_e=vfoptions.n_e;
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

    transpathoptions.epathprecomputed=0;
    if isfield(vfoptions,'pi_e')
        e_grid_J=vfoptions.e_grid.*ones(1,N_j);
        pi_e_J=vfoptions.pi_e.*ones(1,N_j);
        transpathoptions.epathprecomputed=1;
        transpathoptions.epathtrivial=1; % e_grid_J and pi_e_J are not varying over the path
    elseif isfield(vfoptions,'pi_e_J')
        e_grid_J=vfoptions.e_grid_J;
        pi_e_J=vfoptions.pi_e_J;
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
    % simoptions.e_grid_J=e_gridvals_J;
    % simoptions.pi_e_J=pi_e_J;
end


%%
if transpathoptions.verbose==1
    transpathoptions
end

if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%% Setup for various objects
if N_e==0
    if N_z==0
        Policy_final=KronPolicyIndexes_FHorz_Case1_noz(Policy_final, n_d, n_a, N_j);
        if N_d>0
            Policy_final2=Policy_final;
            Policy_final=shiftdim(Policy_final2(1,:,:)+N_d*(Policy_final2(2,:,:)-1),1);
        end
        V_final=reshape(V_final,[N_a,N_j]);
    else
        Policy_final=KronPolicyIndexes_FHorz_Case1(Policy_final,n_d,n_a,n_z,N_j);
        if N_d>0
            Policy_final2=Policy_final;
            Policy_final=shiftdim(Policy_final2(1,:,:,:)+N_d*(Policy_final2(2,:,:,:)-1),1);
        end
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_z,N_j]);
        else % vfoptions.fastOLG==1
            V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_j]),[1,3,2]),[N_a*N_j,N_z]);
            Policy_final=reshape(permute(Policy_final,[1,3,2]),[N_a,N_j,N_z]);
        end
    end
else
    if N_z==0
        Policy_final=KronPolicyIndexes_FHorz_Case1(Policy_final,n_d,n_a,n_z,N_j,n_e);
        if N_d>0
            Policy_final2=Policy_final;
            Policy_final=shiftdim(Policy_final2(1,:,:,:)+N_d*(Policy_final2(2,:,:,:)-1),1);
        end
        V_final=reshape(V_final,[N_a,N_e,N_j]);
    else
        Policy_final=KronPolicyIndexes_FHorz_Case1(Policy_final,n_d,n_a,n_z,N_j,n_e);
        if N_d>0
            Policy_final2=Policy_final;
            Policy_final=shiftdim(Policy_final2(1,:,:,:,:)+N_d*(Policy_final2(2,:,:,:,:)-1),1);
        end
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_z,N_e,N_j]);
        else % vfoptions.fastOLG==1
            V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_e,N_j]),[1,4,2,3]),[N_a*N_j,N_z,N_e]);
            Policy_final=reshape(permute(Policy_final,[1,4,2,3]),[N_a,N_j,N_z,N_e]);
        end
    end
end


%%
if N_e==0
    if N_z==0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, no z, no e
            VPath=zeros(N_a,N_j,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,T-ttr)=Policy;
                VPath(:,:,T-ttr)=V;
            end
        else
            %% fastOLG=1, no z, no e

            VPath=zeros(N_a,N_j,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i
                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,T-ttr)=Policy;
                VPath(:,:,T-ttr)=V;
            end
        end

    else % N_z>0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, z, no e
            VPath=zeros(N_a,N_z,N_j,T,'gpuArray');
            VPath(:,:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,T)=Policy_final;
        
            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,T-ttr)=Policy;
                VPath(:,:,:,T-ttr)=V;
            end

        else
            %% fastOLG=1, z, no e
            % Note: fastOLG with z: use V as (a,j)-by-z and Policy as a-by-j-by-z
            VPath=zeros(N_a*N_j,N_z,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_j,N_z,T,'gpuArray');
            PolicyPath(:,:,:,T)=Policy_final;

            %First, go from T-1 to 1 calculating the Value function and Optimal
            %policy function at each step. Since we won't need to keep the value
            %functions for anything later we just store the next period one in
            %Vnext, and the current period one to be calculated in V
            V=V_final;
            for ttr=1:T-1 %so tt=T-ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

                PolicyPath(:,:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
                VPath(:,:,T-ttr)=V;
            end
        end
    end

else % N_e
    if N_z==0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, no z, e
            VPath=zeros(N_a,N_e,N_j,T,'gpuArray');
            VPath(:,:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_e,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,T)=Policy_final;

            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_noze(V,n_d,n_a,n_z,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,T-ttr)=Policy;
                VPath(:,:,:,T-ttr)=V;
            end

        else
            %% fastOLG=1, no z, e
            % Note: fastOLG with e: use V as (a,j)-by-e and Policy as a-by-j-by-e
            VPath=zeros(N_a*N_j,N_e,T,'gpuArray');
            VPath(:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_j,N_e,T,'gpuArray');
            PolicyPath(:,:,:,T)=Policy_final;

            %First, go from T-1 to 1 calculating the Value function and Optimal
            %policy function at each step. Since we won't need to keep the value
            %functions for anything later we just store the next period one in
            %Vnext, and the current period one to be calculated in V
            V=V_final;
            for ttr=1:T-1 %so tt=T-ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % fastOLG value function uses (j,e)
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_noz_e(V,n_d,n_a,n_z,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

                PolicyPath(:,:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
                VPath(:,:,T-ttr)=V;
            end
        end
    else % N_z>0
        if transpathoptions.fastOLG==0
            %% fastOLG=0, z, e
            VPath=zeros(N_a,N_z,N_e,N_j,T,'gpuArray');
            VPath(:,:,:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_z,N_e,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,:,T)=Policy_final;
            
            % Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            V=V_final;
            for ttr=1:T-1 %so t=T-i

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy is kept in the form where it is just a single-value in (d,a')

                PolicyPath(:,:,:,:,T-ttr)=Policy;
                VPath(:,:,:,:,T-ttr)=V;
            end

        else % transpathoptions.fastOLG==1
            %% fastOLG=1, z, e
            VPath=zeros(N_a*N_j,N_z,N_e,T,'gpuArray');
            VPath(:,:,:,T)=V_final;
            PolicyPath=zeros(N_a,N_j,N_z,N_e,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;

            %First, go from T-1 to 1 calculating the Value function and Optimal
            %policy function at each step. Since we won't need to keep the value
            %functions for anything later we just store the next period one in
            %Vnext, and the current period one to be calculated in V
            V=V_final;
            for ttr=1:T-1 %so tt=T-ttr

                for kk=1:length(PricePathNames)
                    Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                end
                for kk=1:length(ParamPathNames)
                    Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                end

                if transpathoptions.epathtrivial==0
                    pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
                    e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
                end
                if transpathoptions.zpathtrivial==0
                    pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
                    z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
                end

                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG_e(V,n_d,n_a,n_z,n_e, N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
                % The VKron input is next period value fn, the VKron output is this period.
                % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

                PolicyPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
                VPath(:,:,:,T-ttr)=V;
            end
        end
    end
end


%% Unkron to get into the shape for output
% First, when there is N_d, seperate d from aprime
if N_d>0
    PolicyPath2=PolicyPath;
    if N_e==0
        if N_z==0
            PolicyPath=zeros(2,N_a,N_j,T,'gpuArray');
            PolicyPath(1,:,:,:)=shiftdim(rem(PolicyPath2-1,N_d)+1,-1);
            PolicyPath(2,:,:,:)=shiftdim(ceil(PolicyPath2/N_d),-1);
        else
            if transpathoptions.fastOLG==1
                PolicyPath2=permute(PolicyPath2,[1,3,2,4]); % was (a,j,z,t), now (a,z,j,t)
            end
            PolicyPath=zeros(2,N_a,N_z,N_j,T,'gpuArray');
            PolicyPath(1,:,:,:,:)=shiftdim(rem(PolicyPath2-1,N_d)+1,-1);
            PolicyPath(2,:,:,:,:)=shiftdim(ceil(PolicyPath2/N_d),-1);
        end
    else
        if N_z==0
            if transpathoptions.fastOLG==1
                PolicyPath2=permute(PolicyPath2,[1,3,2,4]); % was (a,j,e,t), now (a,e,j,t)
            end
            PolicyPath=zeros(2,N_a,N_e,N_j,T,'gpuArray');
            PolicyPath(1,:,:,:,:)=shiftdim(rem(PolicyPath2-1,N_d)+1,-1);
            PolicyPath(2,:,:,:,:)=shiftdim(ceil(PolicyPath2/N_d),-1);
        else
            if transpathoptions.fastOLG==1
                PolicyPath2=permute(PolicyPath2,[1,3,4,2,5]); % was (a,j,z,e,t), now (a,z,e,j,t)
            end
            PolicyPath=zeros(2,N_a,N_z,N_e,N_j,T,'gpuArray');
            PolicyPath(1,:,:,:,:,:)=shiftdim(rem(PolicyPath2-1,N_d)+1,-1);
            PolicyPath(2,:,:,:,:,:)=shiftdim(ceil(PolicyPath2/N_d),-1);
        end
    end
elseif transpathoptions.fastOLG==1
    % no d, but as fastOLG, still need to permute
    if N_e==0
        if N_z==0
            % no need to do anything
        else
            PolicyPath=permute(PolicyPath,[1,3,2,4]); % was (a,j,e,t), now (a,e,j,t)
        end
    else
        if N_z==0
            PolicyPath=permute(PolicyPath,[1,3,2,4]); % was (a,j,z,t), now (a,z,j,t)
        else
            PolicyPath=permute(PolicyPath,[1,3,4,2,5]); % was (a,j,z,e,t), now (a,z,e,j,t)
        end
    end
end


% Then the unkron itself (includes permute() when fastOLG=1)
if N_e==0
    if N_z==0
        VPath=reshape(VPath,[n_a,N_j,T]);
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_noz(PolicyPath, n_d, n_a,N_j,T);
    else
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_z,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,T]),[1,3,2,4]),[n_a,n_z,N_j,T]);
        end
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath, n_d, n_a, n_z, N_j, T);
    end
else
    if N_z==0
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_e,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_e,T]),[1,3,2,4]),[n_a,n_e,N_j,T]);
        end
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath, n_d, n_a, n_e, N_j, T);
    else
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_z,n_e,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,N_e,T]),[1,3,4,2,5]),[n_a,n_z,n_e,N_j,T]);
        end
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_e(PolicyPath, n_d, n_a, n_z, n_e, N_j,T);
    end
end



end