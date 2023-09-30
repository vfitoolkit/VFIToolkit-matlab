function [VPath,PolicyPath]=ValueFnOnTransPath_Case1_FHorz(PricePath, ParamPath, T, V_final, Policy_final, Parameters, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, transpathoptions, vfoptions)
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
    N_e=1;
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
                simoptions.pi_z_J=pi_z_J;
                simoptions.z_grid_J=z_grid_J;
            elseif transpathoptions.zpathtrivial==0
                % z_grid_J and/or pi_z_J varies along the transition path (but only depending on ParamPath, not PricePath
                transpathoptions.pi_z_J_T=zeros(N_z,N_z,N_j,T,'gpuArray');
                transpathoptions.z_grid_J_T=zeros(sum(n_z),N_j,T,'gpuArray');
                pi_z_J=zeros(N_z,N_z,N_j,'gpuArray');
                z_grid_J=zeros(sum(n_z),N_j,'gpuArray');
                for ttr=1:T
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
                    transpathoptions.pi_z_J_T(:,:,:,ttr)=pi_z_J;
                    transpathoptions.z_grid_J_T(:,:,ttr)=z_grid_J;
                end
            end
        end
    end

    if vfoptions.lowmemory>0
        % Need z_gridvals_J instead of z_grid_J
        z_gridvals_J=zeros(N_z,l_z,N_j,'gpuArray');
        for jj=1:N_j
            z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid_J(:,jj),1);
        end
        vfoptions.z_gridvals_J=z_gridvals_J;
    end

    if transpathoptions.fastOLG==1 % Reshape grid and transtion matrix for use with fastOLG
        z_grid_J=z_grid_J'; % Give it the size required for CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(): N_j-by-N_z
        pi_z_J=permute(pi_z_J,[3,2,1]); % We want it to be (j,z',z) for value function 
        transpathoptions.pi_z_J_alt=permute(pi_z_J,[1,3,2]); % But is (j,z,z') for agent dist with fastOLG [note, this permute is off the previous one]
        if transpathoptions.zpathtrivial==0
            transpathoptions.z_grid_J_T=permute(transpathoptions.z_grid_J_T,[2,1,3]); % from (j,z,t) to (z,j,t)
            transpathoptions.pi_z_J_T=permute(transpathoptions.pi_z_J_T,[3,1,2,4]);  % We want it to be (j,z,z',t)
            transpathoptions.pi_z_J_T_alt=permute(transpathoptions.pi_z_J_T,[1,3,2,4]);  % We want it to be (j,z',z,t) [note, this permute is off the previous one]
        end
    end
    
end

%% If using e variables do the same for e as we just did for z
if N_e>0
    % Check if e_grid and/or pi_e depend on prices. If not then create pi_e_J and e_grid_J for the entire transition before we start

    transpathoptions.epathprecomputed=0;
    if isfield(vfoptions,'pi_e_J')
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
                % Now store them in vfoptions and simoptions
                vfoptions.pi_e_J=pi_e_J;
                vfoptions.e_grid_J=e_grid_J;
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
                        EiidShockFnParamsVec=CreateVectorFromParams(Parameters, vfoptions.EiidShockFnParamNames,jj);
                        EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                        for ii=1:length(ExogShockFnParamsVec)
                            EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                        end
                        [e_grid,pi_e]=vfoptions.ExogShockFn(EiidShockFnParamsCell{:});
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
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%% Setup for various objects
Policy_final=KronPolicyIndexes_FHorz_Case1(Policy_final,n_d,n_a,n_z,N_j);
if N_z==0
    V_final=reshape(V_final,[N_a,N_j]);
    % if transpathoptions.fastOLG==0
    %     V_final=reshape(V_final,[N_a,N_j]);
    % else % vfoptions.fastOLG==1
    %     V_final=reshape(V_final,[N_a,N_j]);
    % end
else
    if transpathoptions.fastOLG==0
        V_final=reshape(V_final,[N_a,N_z,N_j]);
    else % vfoptions.fastOLG==1
        V_final=reshape(permute(reshape(V_final,[N_a,N_z,N_j]),[1,3,2]),[N_a*N_j,N_z]);
        if N_d==0
            Policy_final=reshape(permute(Policy_final,[1,3,2]),[N_a*N_j,N_z]);
        else
            Policy_final2=shiftdim(Policy_final(1,:,:,:)+N_d*(Policy_final(2,:,:,:)-1),1);
            Policy_final=reshape(permute(Policy_final2,[1,3,2]),[N_a*N_j,N_z]);
        end
    end
end


%%
if N_z==0
    if transpathoptions.fastOLG==0
        %%
        VPath=zeros(N_a,N_j,T,'gpuArray');
        VPath(:,:,T)=V_final;
        if N_d>0
            PolicyPath=zeros(2,N_a,N_j,T,'gpuArray');
            PolicyPath(:,:,:,T)=Policy_final;
        else
            PolicyPath=zeros(N_a,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,T)=Policy_final;
        end
        
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

            if N_d>0
                PolicyPath(:,:,:,T-ttr)=Policy;
            else
                PolicyPath(:,:,T-ttr)=Policy;
            end
            VPath(:,:,T-ttr)=V;
        end

    else % transpathoptions.fastOLG==1
        %%
        VPath=zeros(N_a,N_j,T,'gpuArray');
        VPath(:,:,T)=V_final;
        if N_d>0
            PolicyPath=zeros(2,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,T)=Policy_final;
        else
            PolicyPath=zeros(N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,T)=Policy_final;
        end

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

            if N_d>0
                PolicyPath(:,:,:,T-ttr)=Policy;
            else
                PolicyPath(:,:,T-ttr)=Policy;
            end
            VPath(:,:,T-ttr)=V;
        end
    end

else % N_z>0
    if transpathoptions.fastOLG==0
        %%
        VPath=zeros(N_a,N_z,N_j,T,'gpuArray');
        VPath(:,:,:,T)=V_final;
        if N_d>0
            PolicyPath=zeros(2,N_a,N_z,N_j,T,'gpuArray');
            PolicyPath(:,:,:,:,T)=Policy_final;
        else
            PolicyPath=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1
            PolicyPath(:,:,:,T)=Policy_final;
        end
        
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
                z_grid_J=transpathoptions.z_grid_J_T(:,:,T-ttr);
            end
            % transpathoptions.zpathtrivial==1 % Does not depend on T, so is just in vfoptions already

            [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The VKron input is next period value fn, the VKron output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')

            if N_d>0
                PolicyPath(:,:,:,:,T-ttr)=Policy;
            else
                PolicyPath(:,:,:,T-ttr)=Policy;
            end
            VPath(:,:,:,T-ttr)=V;
        end

    else % transpathoptions.fastOLG==1
        %%
        VPath=zeros(N_a*N_j,N_z,T,'gpuArray');
        VPath(:,:,T)=V_final;
        PolicyPath=zeros(N_a*N_j,N_z,T,'gpuArray');
        PolicyPath(:,:,T)=Policy_final;

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
                z_grid_J=transpathoptions.z_grid_J_T(:,:,T-ttr);
            end

            [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            % The VKron input is next period value fn, the VKron output is this period.
            % Policy in fastOLG is [1,N_a*N_j*N_z] and contains the joint-index for (d,aprime)

            PolicyPath(:,:,T-ttr)=Policy; % fastOLG: so (a,j)-by-z
            VPath(:,:,T-ttr)=V;
        end
    end
end


%% Unkron to get into the shape for output
if N_z==0
    VPath=reshape(VPath,[n_a,N_j,T]);
    PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_noz(Policy, n_d, n_a,N_j,T);
else
    if transpathoptions.fastOLG==0
        VPath=reshape(VPath,[n_a,n_z,N_j,T]);
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath, n_d, n_a, n_z, N_j,T);
    else
        VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,T]),[1,3,2,4]),[n_a,n_z,N_j,T]);
        PolicyPath=permute(reshape(PolicyPath,[N_a,N_j,N_z,T]),[1,3,2,4]);
        if N_d==0
            PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath, n_d, n_a, n_z, N_j,T);
        else
            PolicyPath2=zeros(2,N_a,N_z,N_j,T,'gpuArray');
            PolicyPath2(1,:,:,:,:)=shiftdim(rem(PolicyPath-1,N_d)+1,-1);
            PolicyPath2(2,:,:,:,:)=shiftdim(ceil(PolicyPath/N_d),-1);
            PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath2, n_d, n_a, n_z, N_j,T);
        end
    end
end




end