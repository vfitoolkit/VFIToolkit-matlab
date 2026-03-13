function AggVarsPath=TransitionPath_FHorz_PType_singlepath_fastOLG_e_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d,n_d,N_a,n_a,N_z,n_z,N_e,n_e,N_j,d_grid,a_grid,d_gridvals,aprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ze_gridvals_J_fastOLG,ReturnFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions)
% When doing shooting alogrithm on TPath FHorz PType, this is for a given ptype, and does the steps of back-iterate to get policy, then forward to get agent dist and agg vars.
% The only output is the agg vars path.

AggVarsPath=zeros(length(FnsToEvaluateCell),T-1,'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect

if N_d==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_aprime=length(n_a);
l_z=length(n_z);

%%
% fastOLG so everything is (a,j,z,e)
% Shapes:
% V is [N_a,N_j,N_z,N_e]
% AgentDist for fastOLG is [N_a*N_j*N_z,N_e]

if vfoptions.gridinterplayer==0
    PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
elseif vfoptions.gridinterplayer==1
    N_probs=2;
    PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
    PolicyProbsPath=zeros(N_a*(N_j-1)*N_z*N_e,N_probs,T-1,'gpuArray'); % preallocate
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
        pi_z_J=transpathoptions.pi_z_J_T(:,:,:,T-ttr); % fastOLG value function uses (j,z',z)
        z_gridvals_J=transpathoptions.z_gridvals_J(:,:,T-ttr);
    end
    if transpathoptions.epathtrivial==0
        pi_e_J=transpathoptions.pi_e_J_T(:,1,:,T-ttr);
        e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,T-ttr);
    end

    [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e(V,n_d,n_a,n_z,n_e,N_j,d_grid, a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The V input is next period value fn, the V output is this period.
    % Policy in fastOLG is [N_a,N_j,N_z,N_e] and contains the joint-index for (d,aprime)

    PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e

end

%% Modify PolicyIndexesPath into forms needed for forward iteration
% Create version of PolicyIndexesPath in form we want for the agent distribution iteration
% Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
if isscalar(n_a)
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,T-1]); % aprime index
elseif length(n_a)==2
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
elseif length(n_a)==3
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
elseif length(n_a)==4
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
end
PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
if simoptions.gridinterplayer==1
    L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_z,N_e,T-1]
    PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
    PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
    PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
    PolicyProbsPath(:,2,:)=L2index; % L2 index
    PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
    PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
end
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,[n_z,n_e],N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_z*N_e,T]
PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_z*N_e,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars


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
        pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
    end
    if transpathoptions.epathtrivial==0
        pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt); % (a,j,z)-by-e
    end

    jequalOneDist=jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist
    
    AgeWeights=AgeWeights_T(:,:,tt); % by coincidence this is for fastOLG=0,1

    % simoptions.fastOLG=1 is hardcoded
    if simoptions.gridinterplayer==0
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,PolicyaprimejzPath(:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim, pi_e_J_sim,II1orII,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % II1orII is II1
    elseif simoptions.gridinterplayer==1
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_e_raw(AgentDist,PolicyaprimejzPath(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,II1orII,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % II1orII is II
    end
    
    %% AggVars
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights, PolicyValuesPath(:,:,:,1:l_d,tt), PolicyValuesPath(:,:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_a,l_a,l_ze,N_a,N_ze,a_gridvals,ze_gridvals_J_fastOLG,1);
    for ff=1:length(AggVarNames)
        Parameters.(AggVarNames{ff})=AggVars.(AggVarNames{ff}).Mean;
    end
    % Keep AggVars in the AggVarsPath
    for ff=1:length(AggVarNames)
        AggVarsPath(ff,tt)=AggVars.(AggVarNames{ff}).Mean;
    end

    AgentDist=AgentDistnext;
end


end
