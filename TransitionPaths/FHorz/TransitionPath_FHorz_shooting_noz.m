function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_shooting_noz(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d, n_a, N_j, N_d,N_a, l_d,l_aprime,l_a, d_gridvals,d_grid, a_gridvals,a_grid, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions, transpathoptions)
% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=cell(1,2*length(PricePathNames));
    for ii=1:length(PricePathNames)
        pathnametitles{ii}={['Old ',PricePathNames{ii}]};
        pathnametitles{ii+length(PricePathNames)}={['New ',PricePathNames{ii}]};
    end
end

%%
% interpret jequalOneDist input
if transpathoptions.trivialjequalonedist==0
    jequalOneDist_T=jequalOneDist;
    jequalOneDist=jequalOneDist_T(:,1);
end

%%
% Shapes:
% V is [N_a,N_j]
% AgentDist for basic is [N_a,N_j]
% AgentDist for fastOLG is [N_a*N_j,1]

PricePathNew=zeros(size(PricePathOld),'gpuArray'); 
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(FnsToEvaluateCell),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnPath=zeros(T-1,length(GeneralEqmEqnsCell),'gpuArray');

if vfoptions.gridinterplayer==0
    PolicyIndexesPath=zeros(l_d+l_aprime,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
elseif vfoptions.gridinterplayer==1
    PolicyIndexesPath=zeros(l_d+l_aprime+1,N_a,N_j,T-1,'gpuArray'); %Periods 1 to T-1
end
if simoptions.gridinterplayer==0
    if simoptions.fastOLG==0
        II1=1:1:N_a;
        II2=ones(N_a,1);
    elseif simoptions.fastOLG==1
        II1=1:1:N_a*(N_j-1);
        II2=ones(N_a*(N_j-1),1);
    end
elseif simoptions.gridinterplayer==1
    N_probs=2;
    if simoptions.fastOLG==0
        error('Cannot use simoptions.fastOLG=0 with grid interpolation layer')
    elseif simoptions.fastOLG==1
        PolicyProbsPath=zeros(N_a*(N_j-1),N_probs,T-1,'gpuArray'); % preallocate
        II=repelem((1:1:N_a*(N_j-1))',1,N_probs);
    end
end

%%
PricePathDist=Inf;
pathcounter=1;
while PricePathDist>transpathoptions.tolerance && pathcounter<=transpathoptions.maxiter
    %% First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step. 
    % Since we won't need to keep the value functions for anything later we just store the current one in V
    V=V_final;
    for tt=1:T-1 %so t=T-i   
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        
        [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_noz(V,n_d,n_a,N_j,d_gridvals, a_grid, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')
        
        PolicyIndexesPath(:,:,:,T-tt)=Policy;
    end

    %% Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyPath called PolicyaprimePath, which only tracks aprime and has j=1:N_j-1 as we don't use N_j to iterate agent dist (there is no N_j+1)
    % For fastOLG we use PolicyaprimejPath, if there is z then PolicyaprimejzPath
    % When using grid interpolation layer also PolicyProbsPath
    if isscalar(n_a)
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:),[N_a,N_j-1,T-1]); % aprime index
    elseif length(n_a)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
    elseif length(n_a)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
    elseif length(n_a)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,1:N_j-1,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,1:N_j-1,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,1:N_j-1,:)-1),[N_a,N_j-1,T-1]);
    end
    if simoptions.fastOLG==0
        PolicyaprimePath_slowOLG=PolicyaprimePath;
    elseif simoptions.fastOLG==1
        PolicyaprimePath=reshape(permute(reshape(PolicyaprimePath,[N_a,N_j-1,T-1]),[1,2,3]),[N_a*(N_j-1),T-1]);
        PolicyaprimejPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:(N_j-1)-1)',N_a,1);
        if simoptions.gridinterplayer==1
            L2index=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,1:N_j-1,:),[1,N_a,N_j-1,T-1]); % PolicyPath is of size [l_d+l_aprime+1,N_a,N_j,T]
            L2index=reshape(permute(L2index,[2,3,1,4]),[N_a*(N_j-1),1,T-1]);
            PolicyaprimejPath=reshape(PolicyaprimejPath,[N_a*(N_j-1),1,T-1]); % reinterpret this as lower grid index
            PolicyaprimejPath=repelem(PolicyaprimejPath,1,2,1); % create copy that will be the upper grid index
            PolicyaprimejPath(:,2,:)=PolicyaprimejPath(:,2,:)+1; % upper grid index
            PolicyProbsPath(:,2,:)=L2index; % L2 index
            PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
        end
    end
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,n_d,n_a,0,N_j,T-1,d_grid,a_grid,vfoptions,1,0);
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
        if simoptions.fastOLG==0
            AgeWeights=AgeWeights_T(:,:,tt);
        else % simoptions.fastOLG==1
            AgeWeights=AgeWeights_T(:,tt);
        end

        if transpathoptions.trivialjequalonedist==0
            jequalOneDist=jequalOneDist_T(:,tt+1); % Note: t+1 as we are about to create the next period AgentDist
        end

        if simoptions.fastOLG==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_Iteration_noz_raw(AgentDist,PolicyaprimePath_slowOLG(:,:,tt),N_a,N_j,II1,II2,jequalOneDist);
        else % simoptions.fastOLG==1
            if simoptions.gridinterplayer==0
                AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_noz_raw(AgentDist,PolicyaprimejPath(:,tt),N_a,N_j,II1,II2,jequalOneDist);
            elseif simoptions.gridinterplayer==1
                AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_noz_raw(AgentDist,PolicyaprimejPath(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_j,II,jequalOneDist);   
            end
        end

        %% AggVars
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG_noz(AgentDist.*AgeWeights,PolicyValuesPath(:,:,1:l_d,tt), PolicyValuesPath(:,:,l_d+1:end,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_a,l_a,N_a,a_gridvals,1);
        for ii=1:length(AggVarNames)
            Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
        end
        
        %% General Eqm Eqns
        % Evaluate the general eqm conditions, and based on them create PricePathNew (interpretation depends on transpathoptions)
        [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(tt,:),transpathoptions);
        PricePathNew(tt,:)=PricePathNew_tt;
        GEcondnPath(tt,:)=GEcondnPath_tt;
        
        % Sometimes, want to keep the AggVars to plot them
        if transpathoptions.graphaggvarspath==1
            for ii=1:length(AggVarNames)
                AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
            end
        end
        
        AgentDist=AgentDistnext;
    end
    

    %% Now update prices, give verbose feedback, and check for convergence

    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1     
        disp('Old, New')
        % Would be nice to have a way to get the iteration count without having the whole printout of path values (I think that would be useful?)
        pathnametitles{:}
        [PricePathOld,PricePathNew]
    end
    
    % Create plots of the transition path (before we update pricepath)
    createTPathFeedbackPlots(PricePathNames,AggVarNames,GEeqnNames,PricePathOld,AggVarsPath,GEcondnPath,transpathoptions);
    
    % Set price path to be 9/10ths the old path and 1/10th the new path (but making sure to leave prices in periods 1 & T unchanged).
    if transpathoptions.weightscheme==0
        PricePathOld=PricePathNew; % The update weights are already in GEnewprice setup
    elseif transpathoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight.*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight).*PricePathNew(1:T-1,:);
    elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        % I should precalculate these weighting vectors
        Ttheta=transpathoptions.Ttheta;
        PricePathOld(1:Ttheta,:)=transpathoptions.oldpathweight*PricePathOld(1:Ttheta,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:Ttheta,:);
        PricePathOld(Ttheta:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-Ttheta)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(Ttheta:T-1,:)+((exp(linspace(0,log(0.2),T-Ttheta)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(Ttheta:T-1,:);
    elseif transpathoptions.weightscheme==3 % A gradually opening window.
        if (pathcounter*3)<T-1
            PricePathOld(1:(pathcounter*3),:)=transpathoptions.oldpathweight*PricePathOld(1:(pathcounter*3),:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:(pathcounter*3),:);
        else
            PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
        end
    elseif transpathoptions.weightscheme==4 % Combines weightscheme 2 & 3
        if (pathcounter*3)<T-1
            PricePathOld(1:(pathcounter*3),:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),pathcounter*3)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:(pathcounter*3),:)+((exp(linspace(0,log(0.2),pathcounter*3)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:(pathcounter*3),:);
        else
            PricePathOld(1:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
        end
    end
    
    TransPathConvergence=PricePathDist/transpathoptions.tolerance; %So when this gets to 1 we have convergence
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path: %i \n',pathcounter)
        fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', PricePathDist)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',TransPathConvergence)
    end

    if transpathoptions.historyofpricepath==1
        % Store the whole history of the price path and save it every ten iterations
        PricePathHistory{pathcounter,1}=PricePathDist;
        PricePathHistory{pathcounter,2}=PricePathOld;        
        if rem(pathcounter,10)==1
            save ./SavedOutput/TransPath_Internal.mat PricePathHistory
        end
    end

    pathcounter=pathcounter+1;
    

end

end
