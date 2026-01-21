function AggVarsPath=TransitionPath_FHorz_PType_singlepath_fastOLG_noz_raw(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,V_final,AgentDist_initial,jequalOneDist_T,AgeWeights_T,l_d,N_d,n_d,N_a,n_a,N_j,d_grid,a_grid,daprime_gridvals,a_gridvals,ReturnFn, FnsToEvaluateCell, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions)
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
% fastOLG so everything is (a,j)
% Shapes:
% V is [N_a,N_j]
% AgentDist for fastOLG is [N_a*N_j,1]

if vfoptions.gridinterplayer==0
    PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
elseif vfoptions.gridinterplayer==1
    N_probs=2;
    PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
    PolicyProbsPath=zeros(N_a*(N_j-1),N_probs,T-1,'gpuArray'); % preallocate
end


%% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
% Since we won't need to keep the value functions for anything later we just store the current one in V
V=V_final;
for ttr=1:T-1 % so tt=T-ttr
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz(V,n_d,n_a,N_j,d_grid, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The V input is next period value fn, the V output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')

    PolicyIndexesPath(:,:,:,T-ttr)=Policy;
end

%% Modify PolicyIndexesPath into forms needed for forward iteration
% Create version of PolicyIndexesPath in form we want for the agent distribution iteration
% Creates PolicyaprimejPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath
if isscalar(n_a)
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:),[N_a*(N_j-1),T-1]); % aprime index
elseif length(n_a)==2
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
elseif length(n_a)==3
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
elseif length(n_a)==4
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:)-1),[N_a*(N_j-1),T-1]);
end
PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
if simoptions.gridinterplayer==1
    L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:),[N_a*(N_j-1),1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,T-1]
    PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T-1]); % reinterpret this as lower grid index
    PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
    PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
    PolicyProbsPath(:,2,:)=L2index; % L2 index
    PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
    PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
end
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,0,N_j,T-1,d_gridvals,aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,T]
PolicyValuesPath=permute(PolicyValuesPath,[2,3,1,4]); %[N_a,N_j,l_d+l_aprime,T-1] % fastOLG ordering is needed for AggVars

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
    jequalOneDist=jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist

    AgeWeights=AgeWeights_T(:,tt);

    % simoptions.fastOLG=1 is hardcoded
    if simoptions.gridinterplayer==0
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_raw(AgentDist,PolicyaprimejPath(:,tt),N_a,N_j,II1orII,II2,jequalOneDist); % II1orII is II1
    elseif simoptions.gridinterplayer==1
        AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_raw(AgentDist,PolicyaprimejPath(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_j,II1orII,jequalOneDist); % II1orII is II
    end

    %% AggVars
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist.*AgeWeights,PolicyValuesPath(:,:,1:l_d,tt), PolicyValuesPath(:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_a,l_a,N_a,a_gridvals,1);
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
