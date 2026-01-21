function AggVarsPath=TransitionPath_FHorz_PType_singlepath_e_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d,n_d,N_a,n_a,N_z,n_z,N_e,n_e,N_j,d_grid,a_grid,daprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ReturnFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, vfoptions, simoptions)
% When doing shooting alogrithm on TPath FHorz PType, this is for a given ptype, and does the steps of back-iterate to get policy, then forward to get agent dist and agg vars.
% The only output is the agg vars path.

AggVarsPath=zeros(T-1,length(FnsToEvaluateCell),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_aprime=length(n_a);

%%
% Shapes:
% V is [N_a,N_z,N_e,N_j]
% AgentDist for basic is [N_a*N_z*N_e,N_j]
% AgentDist for fastOLG is [N_a*N_j*N_z,N_e]

if vfoptions.gridinterplayer==0
    PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
elseif vfoptions.gridinterplayer==1
    PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
end
if simoptions.gridinterplayer==0
    if simoptions.fastOLG==0
        II1=1:1:N_a*N_z*N_e;
        II2=ones(N_a*N_z*N_e,1);
    elseif simoptions.fastOLG==1
        II1=1:1:N_a*(N_j-1)*N_z*N_e;
        II2=ones(N_a*(N_j-1)*N_z*N_e,1);
        exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
        exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
        justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
    end
elseif simoptions.gridinterplayer==1
    N_probs=2;
    if simoptions.fastOLG==0
        error('Cannot use simoptions.fastOLG=0 with grid interpolation layer')
    elseif simoptions.fastOLG==1
        PolicyProbsPath=zeros(N_a*(N_j-1)*N_z*N_e,N_probs,T-1,'gpuArray'); % preallocate
        II=repelem((1:1:N_a*(N_j-1)*N_z*N_e)',1,N_probs);
        exceptlastj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(0:1:N_j-2)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
        exceptfirstj=repmat((1:1:N_a)',(N_j-1)*N_z*N_e,1)+repmat(repelem(N_a*(1:1:N_j-1)',N_a,1),N_z*N_e,1)+repelem(N_a*N_j*(0:1:N_z*N_e-1)',N_a*(N_j-1),1);
        justfirstj=repmat((1:1:N_a)',N_z*N_e,1)+N_a*N_j*repelem((0:1:N_z*N_e-1)',N_a,1);
    end
end
%% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
% Since we won't need to keep the value functions for anything later we just store the current one in V
V=V_final;
for ttr=1:T-1 %so tt=T-ttr

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if transpathoptions.zpathtrivial==0
        z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,T-ttr);
        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr);
    end
    if transpathoptions.epathtrivial==0
        e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
        pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr);
    end

    [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_e(V,n_d,n_a,n_z,n_e,N_j,d_gridvals, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The V input is next period value fn, the V output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')

    PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy;
end

%% Modify PolicyIndexesPath into forms needed for forward iteration
% Create version of PolicyIndexesPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
% For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
% When using grid interpolation layer also PolicyProbsPath
if isscalar(n_a)
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:),[N_a*N_z*N_e,N_j-1,T-1]); % aprime index
elseif length(n_a)==2
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
elseif length(n_a)==3
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
elseif length(n_a)==4
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,:,:,1:N_j-1,:)-1),[N_a*N_z*N_e,N_j-1,T-1]);
end
if simoptions.fastOLG==0
    PolicyaprimezPath_slowOLG=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1),N_e,1);
elseif simoptions.fastOLG==1
    PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_z*N_e,N_j-1,T-1]),[1,3,2,4]),[N_a*(N_j-1)*N_z*N_e,T-1]);
    PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
    if simoptions.gridinterplayer==1
        L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:,:,1:N_j-1,:),[1,N_a,N_z*N_e,N_j-1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_z,N_e,N_j,T]
        L2index=reshape(permute(L2index,[2,4,3,1,5]),[N_a*(N_j-1)*N_z*N_e,1,T-1]);
        PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
        PolicyProbsPath(:,2,:)=L2index; % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
    end
end
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],N_j,T-1,d_grid,a_grid,vfoptions,1,0);
PolicyValuesPath=permute(PolicyValuesPath,[2,4,3,1,5]); %[N_a,N_j,N_z*N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars

%% Iterate forward over t: iterate agent dist, calculate aggvars, evaluate general eqm
% Call AgentDist the current periods distn and AgentDistnext the next periods distn which we must calculate
AgentDist=AgentDist_initial;
for tt=1:T-1

    % Get t-1 PricePath and ParamPath before we update them
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            if tt>1
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            else
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
    end
    % Get t-1 AggVars before we update them
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                % The AggVars have not yet been updated, so they still contain previous period values
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    % Update current PricePath and ParamPath
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    % Get t+1 PricePath
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
        end
    end

    %% Get the current optimal policy, and iterate the agent dist
    if transpathoptions.zepathtrivial==0
        ze_gridvals_J_fastOLG=transpathoptions.ze_gridvals_J_T_fastOLG(:,:,:,:,tt);
    end
    if transpathoptions.zpathtrivial==0
        if simoptions.fastOLG==0
            pi_z_J=transpathoptions.pi_z_J_T(:,:,:,tt);
        else
            pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
        end
    end
    if transpathoptions.epathtrivial==0
        if simoptions.fastOLG==0
            pi_e_J=transpathoptions.pi_e_J_T(:,:,tt);
        else
            pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt);
        end
    end

    AgeWeights=AgeWeights_T(:,:,tt); % by coincidence this is for fastOLG=0,1

    jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
    
    if simoptions.fastOLG==0
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_Iteration_e_raw(AgentDist,PolicyaprimezPath_slowOLG(:,:,tt),N_a,N_z,N_e,N_j,pi_z_J,pi_e_J,II1,II2,jequalOneDist);
    else % simoptions.fastOLG==1
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,PolicyaprimejzPath(:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim, pi_e_J_sim,II1,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_e_raw(AgentDist,PolicyaprimejzPath(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,II,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
        end
    end

    %% AggVars
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath(:,:,:,1:l_d,tt), PolicyValuesPath(:,:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_a,l_a,l_ze,N_a,N_ze,a_gridvals,ze_gridvals_J_fastOLG,1);
    for ii=1:length(AggVarNames)
        Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
    end
    % Keep AggVars in the AggVarsPath
    for ii=1:length(AggVarNames)
        AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
    end

    AgentDist=AgentDistnext;
end


end