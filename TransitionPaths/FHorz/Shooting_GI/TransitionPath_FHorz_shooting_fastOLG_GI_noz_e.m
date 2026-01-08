function [PricePathOld,GEcondnPath]=TransitionPath_FHorz_shooting_fastOLG_GI_noz_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n2short, n_d, n_a, n_e, N_j, N_d,N_a,N_e, l_d,l_a,l_e, d_gridvals,aprimefine_gridvals,a_gridvals,d_grid,a_grid,e_gridvals_J, pi_e_J,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GEeqnNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, vfoptions, simoptions, transpathoptions)
% fastOLG: fastOLG uses (a,j,e) instead of the standard (a,e,j)
% This (a,j,e) is important for ability to implement codes based on matrix multiplications (especially for Tan improvement)

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
PricePathDist=Inf;
pathcounter=1;

% fastOLG so everything is (a,j,e)
% Shapes:
% V is [N_a,N_j,N_e]
% AgentDist for fastOLG is [N_a*N_j,N_e]

PricePathNew=zeros(size(PricePathOld),'gpuArray');
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(FnsToEvaluate),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnPath=zeros(T-1,length(GeneralEqmEqnsCell),'gpuArray');

% preallocate
PolicyPath_aprime=zeros([N_a*(N_j-1)*N_e,2,T-1],'gpuArray');
PolicyProbsPath=zeros([N_a*(N_j-1)*N_e,2,T-1],'gpuArray');

if transpathoptions.trivialjequalonedist==0
    jequalOneDist_T=jequalOneDist;
    jequalOneDist=jequalOneDist_T(:,1);
end

%%
while PricePathDist>transpathoptions.tolerance && pathcounter<=transpathoptions.maxiter
    
    PolicyPath=zeros(3,N_a,N_j,N_e,T-1,'gpuArray'); %Periods 1 to T-1
    
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
            e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,T-ttr);
            pi_e_J=transpathoptions.pi_e_J_T(:,:,T-ttr); % fastOLG value function uses (j,z',z)
        end
        
        [V, Policy]=ValueFnIter_FHorz_TPath_SingleStep_fastOLG_noz_e(V,n_d,n_a,n_e,N_j,d_grid, a_grid, e_gridvals_J, pi_e_J, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The V input is next period value fn, the V output is this period.
        % Policy in fastOLG is [N_a,N_j,N_e] and contains the joint-index for (d,aprime)
        
        PolicyPath(:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z

    end


    %% Now we have the PolicyPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path
    
    % PolicyPath_d=PolicyPath(1,:,:,:,:);
    PolicyPath_aprimefine=(n2short+1)*(PolicyPath(2,:,:,:,:)-1)+PolicyPath(3,:,:,:,:);
    PolicyPath_aprime(:,1,:)=reshape(PolicyPath(2,:,1:end-1,:,:),[N_a*(N_j-1)*N_e,1,T-1]); % lower grid point index
    PolicyPath_aprime(:,2,:)=PolicyPath_aprime(:,1,:)+1; % upper grid point index
    PolicyProbsPath(:,2,:)=reshape((PolicyPath(3,:,1:end-1,:,:)-1)/n2short,[N_a*(N_j-1)*N_e,1,T-1]); % prob of upper grid point
    PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % prob of lower grid point

    % Call AgentDist the current periods distn
    AgentDist=AgentDist_initial;
    for tt=1:T-1
        
        % Get the current optimal policy (not needed, only use PolicyPath_parimefine, PolicyPath_aprime, and PolicyProbPath)
        % Policy=PolicyPath(:,:,:,:,tt); % fastOLG: so a-by-j-by-z
                
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
        
        if transpathoptions.zpathtrivial==0
            e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,tt);
            pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt); % fastOLG value function uses (j,z,z')
        end

        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights_T(:,:,tt),PolicyPath(1,:,:,:,tt), PolicyPath_aprimefine(:,:,:,:,tt), FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_a,l_a,l_e,N_a,N_e,d_gridvals,aprimefine_gridvals,a_gridvals,e_gridvals_J,1);
        for ii=1:length(AggVarNames)
            Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
        end

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

        if transpathoptions.trivialjequalonedist==0
            jequalOneDist=jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist
        end
        % if simoptions.fastOLG=1 is hardcoded
        AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_TwoProbs_noz_e_raw(AgentDist,PolicyPath_aprime(:,:,tt),PolicyProbsPath(:,:,tt),N_a,N_e,N_j,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist); % Policy for jj=1:N_j-1
    end


    %% Now update prices, give verbose feedback, and check for convergence

    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1     
        pathcounter
        disp('Old, New')
        % Would be nice to have a way to get the iteration count without having the whole
        % printout of path values (I think that would be useful?)
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
