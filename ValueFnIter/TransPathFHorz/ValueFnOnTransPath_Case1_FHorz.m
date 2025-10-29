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
    vfoptions.gridinterplayer=0;
    vfoptions.parallel=1+(gpuDeviceCount>0);
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
    if ~isfield(vfoptions,'gridinterplayer')
        vfoptions.gridinterplayer=0;
    elseif vfoptions.gridinterplayer==1
        if ~isfield(vfoptions,'ngridinterp')
            error('You have vfoptions.gridinterplayer, so must also set vfoptions.ngridinterp')
        end
    end
    if ~isfield(vfoptions,'parallel')
        vfoptions.parallel=1+(gpuDeviceCount>0);
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

%% Set up exogenous shock processes
[z_gridvals_J, pi_z_J, e_gridvals_J, pi_e_J, transpathoptions, vfoptions]=ExogShockSetup_TPath_FHorz(n_z,z_grid,pi_z,N_j,Parameters,PricePathNames,ParamPathNames,transpathoptions,vfoptions,3);
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

if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%% Setup for various objects
if N_e==0
    if N_z==0
        Policy_final=KronPolicyIndexes_FHorz_Case1_noz(Policy_final, n_d, n_a, N_j, vfoptions);
        if vfoptions.gridinterplayer==0
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:)+N_d*(Policy_final2(2,:,:)-1),1);
            end
        elseif vfoptions.gridinterplayer==1
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:)+N_d*(1+(vfoptions.ngridinterp+1)*(Policy_final2(2,:,:)-1)+(Policy_final2(3,:,:)-1)-1),1);
            else
                Policy_final2=Policy_final;
                Policy_final=shiftdim(1+(vfoptions.ngridinterp+1)*(Policy_final2(1,:,:)-1)+(Policy_final2(2,:,:)-1),1);
            end
        end
        V_final=reshape(V_final,[N_a,N_j]);
    else
        Policy_final=KronPolicyIndexes_FHorz_Case1(Policy_final,n_d,n_a,n_z,N_j, vfoptions);
        if vfoptions.gridinterplayer==0
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:,:)+N_d*(Policy_final2(2,:,:,:)-1),1);
            end
        elseif vfoptions.gridinterplayer==1
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:,:)+N_d*(1+(vfoptions.ngridinterp+1)*(Policy_final2(2,:,:,:)-1)+(Policy_final2(3,:,:,:)-1)-1),1);
            else
                Policy_final2=Policy_final;
                Policy_final=shiftdim(1+(vfoptions.ngridinterp+1)*(Policy_final2(1,:,:,:)-1)+(Policy_final2(2,:,:,:)-1),1);
            end
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
        Policy_final=KronPolicyIndexes_FHorz_Case1(Policy_final,n_d,n_a,n_e,N_j, vfoptions);
        if vfoptions.gridinterplayer==0
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:,:)+N_d*(Policy_final2(2,:,:,:)-1),1);
            end
        elseif vfoptions.gridinterplayer==1
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:,:)+N_d*(1+(vfoptions.ngridinterp+1)*(Policy_final2(2,:,:,:)-1)+(Policy_final2(3,:,:,:)-1)-1),1);
            else
                Policy_final2=Policy_final;
                Policy_final=shiftdim(1+(vfoptions.ngridinterp+1)*(Policy_final2(1,:,:,:)-1)+(Policy_final2(2,:,:,:)-1),1);
            end
        end
        if transpathoptions.fastOLG==0
            V_final=reshape(V_final,[N_a,N_e,N_j]);
        else % vfoptions.fastOLG==1
            V_final=reshape(permute(reshape(V_final,[N_a,N_e,N_j]),[1,3,2]),[N_a*N_j,N_e]);
            Policy_final=reshape(permute(Policy_final,[1,3,2]),[N_a,N_j,N_e]);
        end
    else
        Policy_final=KronPolicyIndexes_FHorz_Case1_e(Policy_final,n_d,n_a,n_z,n_e,N_j, vfoptions);
        if vfoptions.gridinterplayer==0
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:,:,:)+N_d*(Policy_final2(2,:,:,:,:)-1),1);
            end
        elseif vfoptions.gridinterplayer==1
            if N_d>0
                Policy_final2=Policy_final;
                Policy_final=shiftdim(Policy_final2(1,:,:,:,:)+N_d*(1+(vfoptions.ngridinterp+1)*(Policy_final2(2,:,:,:,:)-1)+(Policy_final2(3,:,:,:,:)-1)-1),1);
            else
                Policy_final2=Policy_final;
                Policy_final=shiftdim(1+(vfoptions.ngridinterp+1)*(Policy_final2(1,:,:,:,:)-1)+(Policy_final2(2,:,:,:,:)-1),1);
            end
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
                
                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG(V,n_d,n_a,n_z,N_j,d_grid, a_grid, z_gridvals_J, pi_z_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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

                [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e(V,n_d,n_a,n_z,n_e, N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
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
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_noz(PolicyPath, n_d, n_a,N_j,T,vfoptions);
    else
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_z,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,T]),[1,3,2,4]),[n_a,n_z,N_j,T]);
        end
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath, n_d, n_a, n_z, N_j, T,vfoptions);
    end
else
    if N_z==0
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_e,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_e,T]),[1,3,2,4]),[n_a,n_e,N_j,T]);
        end
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz(PolicyPath, n_d, n_a, n_e, N_j, T,vfoptions);
    else
        if transpathoptions.fastOLG==0
            VPath=reshape(VPath,[n_a,n_z,n_e,N_j,T]);
        else
            VPath=reshape(permute(reshape(VPath,[N_a,N_j,N_z,N_e,T]),[1,3,4,2,5]),[n_a,n_z,n_e,N_j,T]);
        end
        PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_e(PolicyPath, n_d, n_a, n_z, n_e, N_j,T,vfoptions);
    end
end



end