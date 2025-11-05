function PricePathOld=TransitionPath_FHorz_shooting_fastOLG_e(PricePathOld, PricePathNames, PricePathSizeVec, l_p, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, jequalOneDist, n_d,n_a,n_z,n_e,N_j, N_d,N_a,N_z,N_e,N_ze, l_d,l_a,l_z,l_e,l_ze, d_gridvals,aprime_gridvals,a_gridvals,d_grid,a_grid,z_gridvals_J,e_gridvals_J,ze_gridvals_J, pi_z_J,pi_e_J,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj, ReturnFn, FnsToEvaluate, AggVarNames, FnsToEvaluateParamNames, GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters, DiscountFactorParamNames, AgeWeights_T, ReturnFnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames,  vfoptions, simoptions, transpathoptions)
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

PricePathDist=Inf;
pathcounter=1;

% fastOLG so everything is (a,j,z,e)
% Shapes:
% V is [N_a,N_j,N_z,N_e]
% AgentDist for fastOLG is [N_a*N_j*N_z,N_e]

PricePathNew=zeros(size(PricePathOld),'gpuArray');
PricePathNew(T,:)=PricePathOld(T,:);
AggVarsPath=zeros(T-1,length(FnsToEvaluate),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
% reshape pi_e_J and e_grid_J for use in fastOLG value fn

if transpathoptions.trivialjequalonedist==0
    jequalOneDist_T=jequalOneDist;
    jequalOneDist=jequalOneDist_T(:,1);
end


%%
while PricePathDist>transpathoptions.tolerance && pathcounter<=transpathoptions.maxiter
    
    PolicyPath=zeros(2,N_a,N_j,N_z,N_e,T-1,'gpuArray'); %Periods 1 to T-1
    
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

        PolicyPath(:,:,:,:,:,T-ttr)=Policy; % fastOLG: so a-by-j-by-z-by-e

    end
    
    
    %% Now we have the PolicyPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path
    
    % Call AgentDist the current periods distn
    AgentDist=AgentDist_initial;
    for tt=1:T-1
        
        % Get the current optimal policy
        Policy=PolicyPath(:,:,:,:,:,tt); % fastOLG: so (a,j)-by-z-by-e
                
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
            z_gridvals_J=transpathoptions.z_gridvals_J_T(:,:,:,tt);
            pi_z_J_sim=transpathoptions.pi_z_J_sim_T(:,:,:,tt);
        end
        if transpathoptions.epathtrivial==0
            e_gridvals_J=transpathoptions.e_gridvals_J_T(:,:,:,:,tt);
            pi_e_J_sim=transpathoptions.pi_e_J_sim_T(:,:,tt); % (a,j,z)-by-e
        end

        AggVars=EvalFnOnAgentDist_AggVars_FHorz_fastOLG(AgentDist.*AgeWeights_T(:,:,tt),Policy(1,:,:,:,:), Policy(2,:,:,:,:), FnsToEvaluate,FnsToEvaluateParamNames,AggVarNames,Parameters,N_j,l_d,l_a,l_a,l_ze,N_a,N_ze,d_gridvals,aprime_gridvals,a_gridvals,ze_gridvals_J,1);
        
        % An easy way to get the new prices is just to call GeneralEqmConditions_Case1 and then adjust it for the current prices
            % When using negative powers matlab will often return complex numbers, even if the solution is actually a real number. I
            % force converting these to real, albeit at the risk of missing problems created by actual complex numbers.
        if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
            AggVarNames=fieldnames(AggVars);
            for ii=1:length(AggVarNames)
                Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
            end
            PricePathNew(tt,:)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell,GeneralEqmEqnParamNames,Parameters));
        % Note there is no GEnewprice==2, it uses a completely different code
        elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
            AggVarNames=fieldnames(AggVars);
            for ii=1:length(AggVarNames)
                Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
            end
            p_i=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell,GeneralEqmEqnParamNames,Parameters));
            p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;
            PricePathNew(tt,:)=PricePathOld(tt,:)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
        end
        
        % Sometimes, want to keep the AggVars to plot them
        if transpathoptions.graphaggvarspath==1
            for ii=1:length(AggVarNames)
                AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
            end
        end

        if transpathoptions.trivialjequalonedist==0
            jequalOneDist=jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist
        end
        % simoptions.fastOLG=1 is hardcoded
        AgentDist=AgentDist_FHorz_TPath_SingleStep_IterFast_e_raw(AgentDist,reshape(Policy(2,:,1:end-1,:,:),[1,N_a*(N_j-1)*N_z*N_e]),N_a,N_z,N_e,N_j,pi_z_J_sim,pi_e_J_sim,exceptlastj,exceptfirstj,justfirstj,jequalOneDist);
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
    
    if transpathoptions.graphpricepath==1
        if length(PricePathNames)>12
            ncolumns=4;
        elseif length(PricePathNames)>6
            ncolumns=3;
        else
            ncolumns=2;
        end
        nrows=ceil(length(PricePathNames)/ncolumns);
        fig1=figure(1);
        for pp=1:length(PricePathNames)
            subplot(nrows,ncolumns,pp); plot(PricePathOld(:,pp))
            title(PricePathNames{pp})
        end
    end
    if transpathoptions.graphaggvarspath==1
        % Do an additional graph, this one of the AggVars
        if length(AggVarNames)>12
            ncolumns=4;
        elseif length(AggVarNames)>6
            ncolumns=3;
        else
            ncolumns=2;
        end
        nrows=ceil(length(AggVarNames)/ncolumns);
        fig2=figure(2);
        for pp=1:length(AggVarNames)
            subplot(nrows,ncolumns,pp); plot(AggVarsPath(:,pp))
            title(AggVarNames{pp})
        end
    end
    
    %Set price path to be 9/10ths the old path and 1/10th the new path (but
    %making sure to leave prices in periods 1 & T unchanged).
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
        fprintf('Current distance to convergence: %.2f (convergence when reaches 1) \n',TransPathConvergence) %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
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
