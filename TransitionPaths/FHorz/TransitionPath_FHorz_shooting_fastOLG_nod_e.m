function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_nod_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_a,n_z,n_e,N_j, N_a,N_z,N_e,N_ze, l_aprime,l_a,l_z,l_e,l_ze, aprime_gridvals,a_gridvals,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J_fastOLG, pi_z_J,pi_e_J,pi_z_J_sim,pi_e_J_sim, ReturnFn, FnsToEvaluateCell, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, N_probs,II1orII,II2,exceptlastj,exceptfirstj,justfirstj, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions, transpathoptions)
% fastOLG: fastOLG uses (a,j,z,e) instead of the standard (a,z,e,j)
% This (a,j,z,e) is important for ability to implement codes based on matrix
% multiplications (especially for Tan improvement)

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
% fastOLG so everything is (a,j,z,e)
% Shapes:
% V is [N_a,N_j,N_z,N_e]
% AgentDist for fastOLG is [N_a*N_j*N_z,N_e]

PricePathNew=zeros(size(PricePathOld),'gpuArray'); 
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(FnsToEvaluateCell),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnPath=zeros(T-1,length(GeneralEqmEqnsCell),'gpuArray');

if vfoptions.gridinterplayer==0
    PolicyIndexesPath=zeros(l_aprime,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
elseif vfoptions.gridinterplayer==1
    PolicyIndexesPath=zeros(l_aprime+1,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
    PolicyProbsPath=zeros(N_a*(N_j-1)*N_z*N_e,N_probs,T-1,'gpuArray'); % preallocate
end

%%
PricePathDist=Inf;
pathcounter=1;
while PricePathDist>transpathoptions.tolerance && pathcounter<=transpathoptions.maxiter    
    
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
        
        [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_e(V,0,n_a,n_z,n_e,N_j,[], a_grid, z_gridvals_J, e_gridvals_J, pi_z_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy in fastOLG is [N_a,N_j,N_z,N_e] and contains the joint-index for (d,aprime)

        PolicyIndexesPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e

    end
    
    %% Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
    % Creates PolicyaprimejzPath (omits j=N_j), and when using grid interpolation layer also PolicyProbsPath 
    if isscalar(n_a)
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,T-1]); % aprime index
    elseif length(n_a)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(2,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
    elseif length(n_a)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(2,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(3,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
    elseif length(n_a)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,1:N_j-1,:,:,:)+n_a(1)*(PolicyIndexesPath(2,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(3,:,1:N_j-1,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(4,:,1:N_j-1,:,:,:)-1),[N_a*(N_j-1)*N_z*N_e,T-1]);
    end
    PolicyaprimejzPath=PolicyaprimePath+repmat(repelem(N_a*gpuArray(0:1:(N_j-1)*N_z-1)',N_a,1),N_e,1);
    if simoptions.gridinterplayer==1
        L2index=reshape(PolicyIndexesPath(l_aprime+1,:,1:N_j-1,:,:,:),[N_a*(N_j-1)*N_z*N_e,1,T-1]); % PolicyIndexesPath is of size [l_d+l_aprime+1,N_a,N_j,N_z,N_e,T-1]
        PolicyaprimejzPath=reshape(PolicyaprimejzPath,[N_a*(N_j-1)*N_z*N_e,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimejzPath=repelem(PolicyaprimejzPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimejzPath(:,2,:)=PolicyaprimejzPath(:,2,:)+1; % upper grid index
        PolicyProbsPath(:,2,:)=L2index; % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
    end
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_FHorz_TPath(PolicyIndexesPath,0,n_a,[n_z,n_e],N_j,T-1,[],aprime_gridvals,vfoptions,1,1); % [size(PolicyValuesPath,1),N_a,N_j,N_z*N_e,T]
    PolicyValuesPath=permute(PolicyValuesPath,[2,3,4,1,5]); %[N_a,N_j,N_z*N_e,l_aprime,T-1] % fastOLG ordering is needed for AggVars

    
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

        if transpathoptions.trivialjequalonedist==0
            jequalOneDist=jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist
        end

        AgeWeights=AgeWeights_T(:,:,tt); % by coincidence this is for fastOLG=0,1

        % simoptions.fastOLG=1 is hardcoded
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,PolicyaprimejzPath(:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim, pi_e_J_sim,II1orII,II2,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % II1orII is II1
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_FHorz_TPath_SingleStep_IterFast_nProbs_e_raw(AgentDist,PolicyaprimejzPath(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,II1orII,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % II1orII is II
        end

        %% AggVars
        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights,[], PolicyValuesPath(:,:,:,:,tt), FnsToEvaluateCell,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,0,l_a,l_a,l_ze,N_a,N_ze,a_gridvals,ze_gridvals_J_fastOLG,1);
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
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
    elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
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
    
    TransPathConvergence=PricePathDist/transpathoptions.tolerance; %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
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
