function [PricePathNew]=TransitionPath_Case2(PricePathOld, PriceParamNames, ParamPath, PathParamNames, Parameters, DiscountFactorParamNames, Phi_aprimeKron_final, Case2_Type, T, V_final, StationaryDist_init, ReturnFn, ReturnFnParamNames, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, SSvaluesFn,SSvalueParamNames, MarketPriceEqns, MarketPriceParamNames,transpathoptions)

%% Check which transpathoptions have been used, set all others to defaults 
if nargin<25
    disp('No transpathoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-4);
    transpathoptions.parallel=2;
    transpathoptions.exoticpreferences=0;
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.maxiterations=1000;
    transpathoptions.verbose=1;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    eval('fieldexists=1;transpathoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.tolerance=10^(-4);
    end
    eval('fieldexists=1;transpathoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.parallel=2;
    end
    eval('fieldexists=1;transpathoptions.exoticpreferences;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.exoticpreferences=0;
    end
    eval('fieldexists=1;transpathoptions.oldpathweight;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.oldpathweight=0.9;
    end
    eval('fieldexists=1;transpathoptions.weightscheme;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.weightscheme=1;
    end
    eval('fieldexists=1;transpathoptions.maxiterations;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.maxiterations=1000;
    end
    eval('fieldexists=1;transpathoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.verbose=1;
    end
end

%%
%%
if transpathoptions.exoticpreferences~=0
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
    dbstack
else
    if length(DiscountFactorParamNames)~=1
        disp('WARNING: DiscountFactorParamNames should be of length one')
        dbstack
    end
end

if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePathOld=gpuArray(PricePathOld);
end
unkronoptions.parallel=2;

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePathOld,2);

if N_d==0
    PricePathNew=TransitionPath_Case1_no_d(PricePathOld, PriceParamNames, ParamPath, PathParamNames, Parameters, DiscountFactorParamNames, T, V_final, StationaryDist_init, ReturnFn,ReturnFnParamNames, n_a, n_z, pi_z, a_grid,z_grid, SSvaluesFn,SSvalueParamNames, MarketPriceEqns, MarketPriceParamNames,transpathoptions);
    return
end

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(StationaryDist_init,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);
Policy=zeros(N_a,N_z,'gpuArray');
Phi_aprimeKron=Phi_aprimeKron_final; % Might want to change this so that Phi_aprimeKron can change along the transition path.

if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    PathParamNames
    PriceParamNames
end

beta=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
IndexesForPathParamsInDiscountFactor=CreateParamVectorIndexes(DiscountFactorParamNames, PathParamNames);
IndexesForDiscountFactorInPathParams=CreateParamVectorIndexes(PathParamNames,DiscountFactorParamNames);
ReturnFnParamsVec=gpuArray(CreateVectorFromParams(Parameters, ReturnFnParamNames));
IndexesForPricePathInReturnFnParams=CreateParamVectorIndexes(ReturnFnParamNames, PriceParamNames);
IndexesForReturnFnParamsInPricePath=CreateParamVectorIndexes(PriceParamNames, ReturnFnParamNames);
IndexesForPathParamsInReturnFnParams=CreateParamVectorIndexes(ReturnFnParamNames, PathParamNames);
IndexesForReturnFnParamsInPathParams=CreateParamVectorIndexes(PathParamNames,ReturnFnParamNames);
SSvalueParamsVec=gpuArray(CreateVectorFromParams(Parameters, SSvalueParamNames));
IndexesForPricePathInSSvalueParams=CreateParamVectorIndexes(SSvalueParamNames, PriceParamNames);
IndexesForSSvalueParamsInPricePath=CreateParamVectorIndexes(PriceParamNames,SSvalueParamNames);
IndexesForPathParamsInSSvalueParams=CreateParamVectorIndexes(SSvalueParamNames, PathParamNames);
IndexesForSSvalueParamsInPathParams=CreateParamVectorIndexes(PathParamNames,SSvalueParamNames);
MarketPriceParamsVec=gpuArray(CreateVectorFromParams(Parameters, MarketPriceParamNames));
IndexesForPricePathInMarketPriceParams=CreateParamVectorIndexes(MarketPriceParamNames, PriceParamNames);
IndexesForMarketPriceParamsInPricePath=CreateParamVectorIndexes(PriceParamNames, MarketPriceParamNames);
IndexesForPathParamsInMarketPriceParams=CreateParamVectorIndexes(MarketPriceParamNames, PathParamNames);
IndexesForMarketPriceParamsInPathParams=CreateParamVectorIndexes(PathParamNames,MarketPriceParamNames);


% if Case2_Type==1
%     Phi_aprimeKron=reshape(Phi_aprimeKron, [N_d,N_a*N_z,N_z]);
% end
% 
% if Case2_Type==1
%     while PricePathDist>PricePathTolerance
%         PolicyIndexesPath=zeros(N_a,N_z,T-1); %Periods 1 to T-1
%         
%         PricePathOld
%         
%         %First, go from T-1 to 1 calculating the Value function and Optimal
%         %policy function at each step. Since we won't need to keep the value
%         %functions for anything later we just store the next period one in
%         %Vnext, and the current period one to be calculated in V
%         Vnext=V_final;
%         for i=1:T-1 %so t=T-i
%             params=ParamPath(T-i,:);
%             p=PricePathOld(T-i,:);
%             if transpathoptions.returnmatrix==1
%                 Fmatrix=ReturnFn(p,params);
%             elseif transpathoptions.returnmatrix==0
%                 disp('Transition path does not yet support vfoptions.returnmatrix==0')
%             end
% %             Fmatrix=reshape(ReturnFn(p,params),[N_d,N_a,N_z]);
%             
%             for z_c=1:N_z
%                 for a_c=1:N_a
%                     %first calc the second half of the RHS (except beta)
%                     RHSpart2=zeros(N_d,1);
%                     for zprime_c=1:N_z
%                         if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                             az_c=sub2ind_homemade([a_c,z_c],[N_a,N_z]);
%                             RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:,az_c,zprime_c)],zprime_c)*pi_z(z_c,zprime_c);
% %                             RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:,a_c,z_c,zprime_c)],zprime_c)*pi_z(z_c,zprime_c);
%                         end
%                     end
%                     entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
%                     
%                     %then maximizing d indexes
%                     [V(a_c,z_c),PolicyIndexes(a_c,z_c)]=max(entireRHS,[],1);
%                 end
%             end
%             
%             PolicyIndexesPath(:,:,T-i)=PolicyIndexes;
%             Vnext=V;
%         end
%         
%         
%         %Now we have the full PolicyIndexesPath, we go forward in time from 1
%         %to T using the policies to update the agents distribution generating a
%         %new price path
%         %Call SteadyStateDist the current periods distn and SteadyStateDistnext
%         %the next periods distn which we must calculate
%         SteadyStateDist=SteadyStateDist_initial;
%         for i=1:T-1
%             %Get the current optimal policy
%             PolicyIndexes=reshape(PolicyIndexesPath(:,:,i),[N_a*N_z,1]);
%             %Use this to calculate the steady state distn
%             P=zeros(N_a*N_z,N_a*N_z);
% %             P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%             
% 
%             parfor az_c=1:N_a*N_z
%                 optd=PolicyIndexes(az_c);
%                 Prow=zeros(1,N_a*N_z);
%                 for zprime_c=1:N_z
%                     optaprime=Phi_aprimeKron(optd,az_c,zprime_c);
%                     optaprimezprime_c=sub2ind_homemade([optaprime,zprime_c],[N_a,N_z]);
%                     Prow(optaprimezprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%                 end
%                 P(az_c,:)=Prow;
%             end
%             Ptransposed=P';
%             SteadyStateDistnext=Ptransposed*SteadyStateDist;
%             
%             p=PricePathOld(i,:);
%             SSvalues_AggVars=SSvalues_AggVars_Case2_raw(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, n_z, d_grid, a_grid,s_grid,pi_z,p);
%             %An easy way to get the new prices is just to call MarketClearance
%             %and then adjust it for the current prices
%             for j=1:length(MarketPriceEqns)
%                 PricePathNew(i,j)=MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParams);
%             end
%             
%             SteadyStateDist=SteadyStateDistnext;
%         end
%         
%         %See how far apart the price paths are
%         PricePathDist=sum(abs(reshape(PricePathNew-PricePathOld,[numel(PricePathOld),1])));
%         
%         %Set price path to be 9/10ths the old path and 1/10th the new path (but
%         %making sure to leave prices in periods 1 & T unchanged).
%         PricePathOld(2:T-1)=0.9*PricePathOld(2:T-1)+0.1*PricePathNew(2:T-1);
%         
%         pathcounter=pathcounter+1;
%         TransPathConvergence=PricePathDist/PricePathTolerance %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence is going)
%         save ./SavedOutput/TransPathConv.mat PricePathOld TransPathConvergence pathcounter
%     end
% end



if Case2_Type==2
    aaa=kron(pi_z,ones(N_d,1,'gpuArray'));
    
    while PricePathDist>transpathoptions.tolerance
        PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
                
        %First, go from T-1 to 1 calculating the Value function and Optimal
        %policy function at each step. Since we won't need to keep the value
        %functions for anything later we just store the next period one in
        %Vnext, and the current period one to be calculated in V
        Vnext=V_final;
        for i=1:T-1 %so t=T-i

            if ~isnan(IndexesForPathParamsInDiscountFactor)
                beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-i,IndexesForDiscountFactorInPathParams); % This step could be moved outside all the loops
            end
            if ~isnan(IndexesForPricePathInReturnFnParams)
                ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePathOld(T-i,IndexesForReturnFnParamsInPricePath);
            end
            if ~isnan(IndexesForPathParamsInReturnFnParams)
                ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-i,IndexesForReturnFnParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
            end
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
          
            
            EV=zeros(N_d*N_z,N_z,'gpuArray');
            for zprime_c=1:N_z
                EV(:,zprime_c)=Vnext(Phi_aprimeKron(:,:,zprime_c),zprime_c); %(d,z')
            end
            EV=EV.*aaa;
            EV(isnan(EV))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV=reshape(sum(EV,2),[N_d,1,N_z]);
            
            for z_c=1:N_z % Can probably eliminate this loop and replace with a matrix multiplication operation thereby making it faster
                entireRHS=ReturnMatrix(:,:,z_c)+beta*EV(:,z_c)*ones(1,N_a,1,'gpuArray');
                
                %Calc the max and it's index
                [Vtemp,maxindex]=max(entireRHS,[],1);
                V(:,z_c)=Vtemp;
                Policy(:,z_c)=maxindex;
            end
            
            PolicyIndexesPath(:,:,T-i)=Policy;
            Vnext=V;
        end
        
        
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution generating a
        %new price path
        %Call AgentDist the current periods distn and AgentDistnext
        %the next periods distn which we must calculate
        AgentDist=AgentDist_initial;
        for i=1:T-1
            %Get the current optimal policy
            Policy=PolicyIndexesPath(:,:,i);
                        
            % optaprime is here replaced by Phi_of_Policy, which is a different shape
            Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
            for z_c=1:N_z
                Phi_of_Policy(:,:,z_c)=Phi_aprimeKron(Policy(:,z_c),:,z_c);
            end
            Ptemp=zeros(N_a,N_a*N_z*N_z,'gpuArray');
            Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a*N_z*N_z])+N_a*(gpuArray(0:1:N_a*N_z*N_z-1)))=1;
            Ptran=kron(pi_z',ones(N_a,N_a,'gpuArray')).*reshape(Ptemp,[N_a*N_z,N_a*N_z]);
            AgentDistnext=Ptran*AgentDist;

            p=PricePathOld(i,:);
        
            if ~isnan(IndexesForPricePathInSSvalueParams)
                SSvalueParamsVec(IndexesForPricePathInSSvalueParams)=PricePathOld(i,IndexesForSSvalueParamsInPricePath);
            end
            if ~isnan(IndexesForPathParamsInSSvalueParams)
                SSvalueParamsVec(IndexesForPathParamsInSSvalueParams)=ParamPath(i,IndexesForSSvalueParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
            end
            PolicyTemp=UnKronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,unkronoptions);
            SSvalues_AggVars=SSvalues_AggVars_Case2_vec(AgentDist, PolicyTemp, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p, 2);

            
            if ~isnan(IndexesForPricePathInMarketPriceParams)
                MarketPriceParamsVec(IndexesForPricePathInMarketPriceParams)=PricePathOld(i,IndexesForMarketPriceParamsInPricePath);
            end
            if ~isnan(IndexesForPathParamsInMarketPriceParams)
                MarketPriceParamsVec(IndexesForPathParamsInMarketPriceParams)=ParamPath(i,IndexesForMarketPriceParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
            end
            %An easy way to get the new prices is just to call MarketClearance
            %and then adjust it for the current prices
            for j=1:length(MarketPriceEqns)
                
                % When using negative powers matlab will often return complex
                % numbers, even if the solution is actually a real number. I
                % force converting these to real, albeit at the risk of missing problems
                % created by actual complex numbers.
                PricePathNew(i,j)=real(MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParamsVec));
                
            end
            
            AgentDist=AgentDistnext;
        end
        
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
            PricePathOld(1:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
        elseif transpathoptions.weightscheme==3
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



% if Case2_Type==3
%     while PricePathDist>PricePathTolerance
%         PolicyIndexesPath=zeros(N_a,N_z,T-1); %Periods 1 to T-1
%         
%         PricePathOld
%         
%         %First, go from T-1 to 1 calculating the Value function and Optimal
%         %policy function at each step. Since we won't need to keep the value
%         %functions for anything later we just store the next period one in
%         %Vnext, and the current period one to be calculated in V
%         Vnext=V_final;
%         for i=1:T-1 %so t=T-i
%             params=ParamPath(T-i,:);
%             p=PricePathOld(T-i,:);
%             if transpathoptions.returnmatrix==1
%                 Fmatrix=ReturnFn(p,params);
%             elseif transpathoptions.returnmatrix==0
%                 disp('Transition path does not yet support vfoptions.returnmatrix==0')
%             end
% %             Fmatrix=reshape(ReturnFn(p,params),[N_d,N_a,N_z]);
%             
%             for z_c=1:N_z
%                 %first calc the second half of the RHS (except beta)
%                 RHSpart2=zeros(N_d,1);
%                 for zprime_c=1:N_z
%                     if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                         RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:)],zprime_c)*pi_z(z_c,zprime_c);
%                     end
%                 end
%                 for a_c=1:N_a
%                     entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
%                     
%                     %then maximizing d indexes
%                     [V(a_c,z_c),PolicyIndexes(a_c,z_c)]=max(entireRHS,[],1);
%                 end
%             end
%             
%             PolicyIndexesPath(:,:,T-i)=PolicyIndexes;
%             Vnext=V;
%         end
%         
%         
%         %Now we have the full PolicyIndexesPath, we go forward in time from 1
%         %to T using the policies to update the agents distribution generating a
%         %new price path
%         %Call SteadyStateDist the current periods distn and SteadyStateDistnext
%         %the next periods distn which we must calculate
%         SteadyStateDist=SteadyStateDist_initial;
%         for i=1:T-1
%             %Get the current optimal policy
%             PolicyIndexes=PolicyIndexesPath(:,:,i);
%             %Use this to calculate the steady state distn
%             P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%             
%             for z_c=1:N_z
%                 for a_c=1:N_a
%                     optd=PolicyIndexes(a_c,z_c);
%                     for zprime_c=1:N_z
%                         optaprime=Phi_aprimeKron(optd);
%                         P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%                     end
%                 end
%             end
%             P=reshape(P,[N_a*N_z,N_a*N_z]);
%             P=P';
%             SteadyStateDistnext=P*SteadyStateDist;
%             
%             p=PricePathOld(i,:);
%             SSvalues_AggVars=SSvalues_AggVars_Case2_raw(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, n_z, d_grid, a_grid,s_grid,pi_z,p);
%             %An easy way to get the new prices is just to call MarketClearance
%             %and then adjust it for the current prices
%             for j=1:length(MarketPriceEqns)
%                 PricePathNew(i,j)=MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParams);
%             end
%             
%             SteadyStateDist=SteadyStateDistnext;
%         end
%         
%         %See how far apart the price paths are
%         PricePathDist=sum(abs(reshape(PricePathNew-PricePathOld,[numel(PricePathOld),1])));
%         
%         %Set price path to be 9/10ths the old path and 1/10th the new path (but
%         %making sure to leave prices in periods 1 & T unchanged).
%         PricePathOld(2:T-1)=0.9*PricePathOld(2:T-1)+0.1*PricePathNew(2:T-1);
%         
%         pathcounter=pathcounter+1;
%         TransPathConvergence=PricePathDist/PricePathTolerance %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
%         save ./SavedOutput/TransPathConv.mat PricePathOld TransPathConvergence pathcounter
%     end
% end

end