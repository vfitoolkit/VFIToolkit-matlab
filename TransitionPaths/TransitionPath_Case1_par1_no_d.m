function [PricePathOld]=TransitionPath_Case1_par1_no_d(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions)

%% 
% % HAVE NOT YET IMPLEMENTED lowmemory with par1
if transpathoptions.lowmemory==1
    fprintf('ERROR: Have not yet implemented lowmemory with parallel==1 options, contact robertdkirkby@gmail.com if you want to use this')
    dbstack
    return
end

% if transpathoptions.lowmemory==1
%     PricePathOld=TransitionPath_Case1_par1_no_d_lowmem(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_a, n_z, pi_z, a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions);
%     return
% end


%%
if transpathoptions.verbose==1
    transpathoptions
end

unkronoptions.parallel=1;

N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePathOld,2);

PricePathDist=Inf;
pathcounter=1;

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(StationaryDist_init,[N_a*N_z,1]);
V=zeros(size(V_final));
PricePathNew=zeros(size(PricePathOld)); PricePathNew(T,:)=PricePathOld(T,:);
Policy=zeros(N_a,N_z);

if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end
    
beta=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
IndexesForPathParamsInDiscountFactor=CreateParamVectorIndexes(DiscountFactorParamNames, ParamPathNames);
ReturnFnParamsVec=CreateVectorFromParams(Parameters, ReturnFnParamNames);
IndexesForPricePathInReturnFnParams=CreateParamVectorIndexes(ReturnFnParamNames, PricePathNames);
IndexesForPathParamsInReturnFnParams=CreateParamVectorIndexes(ReturnFnParamNames, ParamPathNames);

while PricePathDist>transpathoptions.tolerance && pathcounter<transpathoptions.maxiterations
    PolicyIndexesPath=zeros(N_a,N_z,T-1); %Periods 1 to T-1
        
    %First, go from T-1 to 1 calculating the Value function and Optimal
    %policy function at each step. Since we won't need to keep the value
    %functions for anything later we just store the next period one in
    %Vnext, and the current period one to be calculated in V
    Vnext=V_final;
    for i=1:T-1 %so t=T-i
        
        if ~isnan(IndexesForPathParamsInDiscountFactor)
            beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-i,:); % This step could be moved outside all the loops
        end
        if ~isnan(IndexesForPricePathInReturnFnParams)
            ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePathOld(T-i,:);
        end
        if ~isnan(IndexesForPathParamsInReturnFnParams)
            ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-i,:); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
        end
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid,1,ReturnFnParamsVec);

        parfor z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=Vnext.*(ones(N_a,1)*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,z_c)=Vtemp;
            Policy(:,z_c)=maxindex;
            
        end
        
        PolicyIndexesPath(:,:,T-i)=Policy;
        Vnext=V;
    end
    % Free up space on GPU by deleting things no longer needed
    clear ReturnMatrix ReturnMatrix_z entireRHS EV_z Vtemp maxindex V Vnext    
    
    %Now we have the full PolicyIndexesPath, we go forward in time from 1
    %to T using the policies to update the agents distribution generating a
    %new price path
    %Call SteadyStateDist the current periods distn and SteadyStateDistnext
    %the next periods distn which we must calculate
    AgentDist=AgentDist_initial;
    for i=1:T-1
        %Get the current optimal policy
        Policy=PolicyIndexesPath(:,:,i);

        optaprime=reshape(Policy,[1,N_a*N_z]);
        
        Ptemp=zeros(N_a,N_a*N_z);
        Ptemp(optaprime+N_a*(0:1:N_a*N_z-1))=1;
        Ptran=(kron(pi_z',ones(N_a,N_a))).*(kron(ones(N_z,1),Ptemp));
        AgentDistnext=Ptran*AgentDist;
        
        p=PricePathOld(i,:);
        
        for nn=1:length(ParamPathNames)
            Parameters.(ParamPathNames{nn})=ParamPath(i,nn);
        end
        for nn=1:length(PricePathNames)
            Parameters.(PricePathNames{nn})=PricePathOld(i,nn);
        end
        
        PolicyTemp=UnKronPolicyIndexes_Case1(Policy, 0, n_a, n_z,unkronoptions);
        AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, PolicyTemp, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, 0, n_a, n_z, 0, a_grid, z_grid, 1);
        for ii=1:length(transpathoptions.AggVarsNames)
            Parameters.(transpathoptions.AggVarsNames{ii})=AggVars(ii);
        end

        % When using negative powers matlab will often return complex numbers, even if the solution is actually a real number. I
        % force converting these to real, albeit at the risk of missing problems created by actual complex numbers.
        if transpathoptions.GEnewprice==1
            if transpathoptions.oldGE==1
                PricePathNew(i,:)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnInputNames, 1));
            else
                PricePathNew(i,:)=real(GeneralEqmConditions_Case1_new(GeneralEqmEqns, GeneralEqmEqnInputNames, Parameters, 1));
            end
        elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
            fprintf('ERROR: transpathoptions.GEnewprice==0 NOT YET IMPLEMENTED (TransitionPath_Case1_no_d.m)')
            return
%             for j=1:length(MarketPriceEqns)
%                 GEeqn_temp=@(p) real(MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParamsVec));
%                 PricePathNew(i,j)=fzero(GEeqn_temp,p);
%             end
        % Note there is no GEnewprice==2, I did this to make it harder to make a mistake
        elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
            if heteroagentoptions.oldGE==1
                p_i=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnInputNames, 1));
            else
                p_i=real(GeneralEqmConditions_Case1_new(GeneralEqmEqns, GeneralEqmEqnInputNames, Parameters, 1));
            end
%             GEcondnspath(i,:)=p_i;
            p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            PricePathNew(i,:)=PricePathOld(i,:)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
        end
        
        AgentDist=AgentDistnext;
    end
    % Free up space on GPU by deleting things no longer needed
    clear Ptemp Ptran AgentDistnext AgentDist
        
    %See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    %Notice that the distance is always calculated ignoring the time t=1 &
    %t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1
        disp('Old, New')
        [PricePathOld,PricePathNew]
    end
    
    %Set price path to be 9/10ths the old path and 1/10th the new path (but
    %making sure to leave prices in periods 1 & T unchanged).
    if transpathoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
    elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        % I should precalculate these weighting vectors
        Ttheta=transpathoptions.Ttheta;
        PricePathOld(1:Ttheta,:)=transpathoptions.oldpathweight*PricePathOld(1:Ttheta)+(1-transpathoptions.oldpathweight)*PricePathNew(1:Ttheta,:);
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
        fprintf('Current distance to convergence: %.2f (convergence when reaches 1) \n',TransPathConvergence) %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
    end
%     save ./SavedOutput/TransPathConv.mat TransPathConvergence pathcounter
    
%     if pathcounter==1
%         save ./SavedOutput/FirstTransPath.mat V_final V PolicyIndexesPath PricePathOld PricePathNew
%     end
    
    pathcounter=pathcounter+1;
end


end