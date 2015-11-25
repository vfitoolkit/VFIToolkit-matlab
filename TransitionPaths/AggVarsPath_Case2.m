function [AggVarsPath]=AggVarsPath_Case2(SSvaluesFn, SSvalueParamNames,PricePath,PriceParamNames, ParamPath, PathParamNames, Parameters, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, Phi_aprimeKron, Case2_Type,T, V_final, AgentDist_initial, ReturnFn, ReturnFnParamNames, SSvalues_AggVars_final)
%AggVarsPath is T+1 periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray');
%Policy=zeros(1,N_a,N_z);

unkronoptions.parallel=2;

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

%PolicyIndexesPath=zeros(N_d,N_a,N_z,T-1); %Periods 1 to T-1

PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;


% 
% if Case2_Type==1
%     %First, go from T-1 to 1 calculating the Value function and Optimal
%     %policy function at each step. Since we won't need to keep the value
%     %functions for anything later we just store the next period one in
%     %Vnext, and the current period one to be calculated in V
%     Vnext=V_final;
%     for i=1:T-1 %so t=T-i
%         params=ParamPath(T-i,:);
%         p=PricePath(T-i,:);
%         if transpathoptions.returnmatrix==1
%             Fmatrix=ReturnFn(p,params);
%         elseif transpathoptions.returnmatrix==0
%             disp('Transition path does not yet support vfoptions.returnmatrix==0')
%         end
% %        Fmatrix=reshape(FmatrixFn(p,params),[N_d,N_a,N_s]);
%         
%         for s_c=1:N_s
%             for a_c=1:N_a
%                 %first calc the second half of the RHS (except beta)
%                 RHSpart2=zeros(N_d,1);
%                 for sprime_c=1:N_s
%                     if pi_s(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                         RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:,a_c,s_c,sprime_c)],sprime_c)*pi_s(s_c,sprime_c);
%                     end
%                 end
%                 entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %d by 1
%                 
%                 %then maximizing d indexes
%                 [V(a_c,s_c),PolicyIndexes(a_c,s_c)]=max(entireRHS,[],1);
%             end
%         end
%         
%         PolicyIndexesPath(:,:,:,T-i)=PolicyIndexes;
%         Vnext=V;
%     end
%     
%     %Now we have the full PolicyIndexesPath, we go forward in time from 1
%     %to T using the policies to generate the AggVarsPath. First though we
%     %put in it's initial and final values.
%     AggVarsPath=zeros(length(SSvaluesFn),T+1);
%     AggVarsPath(:,1)=SSvalues_AggVars_initial; AggVarsPath(:,T+1)=SSvalues_AggVars_final;
%     %Call SteadyStateDist the current periods distn and SteadyStateDistnext
%     %the next periods distn which we must calculate
%     SteadyStateDist=SteadyStateDist_initial;
%     for i=1:T-1
%         %Get the current optimal policy
%         PolicyIndexes=PolicyIndexesPath(:,:,:,i);
%         %Use this to calculate the steady state distn
%         P=zeros(N_a,N_s,N_a,N_s); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%         for s_c=1:N_s
%             for a_c=1:N_a
%                 optd=PolicyIndexes(a_c,s_c);
%                 for sprime_c=1:N_s
%                     optaprime=Phi_aprimeKron(optd,a_c,s_c,sprime_c);
%                     P(a_c,s_c,optaprime,sprime_c)=pi_s(s_c,sprime_c)/sum(pi_s(s_c,:));
%                 end
%             end
%         end
%         P=reshape(P,[N_a*N_s,N_a*N_s]);
%         P=P';
%         SteadyStateDistnext=P*SteadyStateDist;
%         
%         AggVarsPath(:,i)=SSvalues_AggVars_Case2_raw(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, N_s, d_grid, a_grid,s_grid,pi_s,p); %the two zeros represent the d variables
%         
%         SteadyStateDist=SteadyStateDistnext;
%     end
%     %i=T
%     params=ParamPath(T,:);
%     p=PricePath(T,:);
%     Fmatrix=reshape(FmatrixFn(p,params),[N_d,N_a,N_s]);
%     for s_c=1:N_s
%         for a_c=1:N_a
%             %first calc the second half of the RHS (except beta)
%             RHSpart2=zeros(N_d,1);
%             for sprime_c=1:N_s
%                 if pi_s(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     RHSpart2=RHSpart2+V_final([Phi_aprimeKron(:,a_c,s_c,sprime_c)],sprime_c)*pi_s(s_c,sprime_c);
%                 end
%             end
%             entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %d by 1
%             
%             %then maximizing d indexes
%             [V(a_c,s_c),PolicyIndexes(a_c,s_c)]=max(entireRHS,[],1);
%         end
%     end
%     
%     AggVarsPath(:,T)=SSvalues_AggVars_Case2_raw(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, N_s, d_grid, a_grid,s_grid,pi_s,p); %the two zeros represent the d variables
%     %end
%     
% end






if Case2_Type==2
    aaa=kron(pi_z,ones(N_d,1,'gpuArray'));

    PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
    %First, go from T-1 to 1 calculating the Value function and Optimal
    %policy function at each step. Since we won't need to keep the value
    %functions for anything later we just store the next period one in
    %Vnext, and the current period one to be calculated in V
    Vnext=V_final;
    for ii=1:T-1 %so t=T-i
        if ~isnan(IndexesForPathParamsInDiscountFactor)
            beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-ii,IndexesForDiscountFactorInPathParams); % This step could be moved outside all the loops
        end
        if ~isnan(IndexesForPricePathInReturnFnParams)
            ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePath(T-ii,IndexesForReturnFnParamsInPricePath);
        end
        if ~isnan(IndexesForPathParamsInReturnFnParams)
            ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-ii,IndexesForReturnFnParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
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
        
        PolicyIndexesPath(:,:,T-ii)=Policy;
        Vnext=V;
%         params=ParamPath(T-i,:);
%         p=PricePath(T-i,:);
%         if transpathoptions.returnmatrix==1
%             Fmatrix=ReturnFn(p,params);
%         elseif transpathoptions.returnmatrix==0
%             disp('Transition path does not yet support vfoptions.returnmatrix==0')
%         end
% %         Fmatrix=reshape(FmatrixFn(p,params),[N_d,N_a,N_s]);
%         
%         for s_c=1:N_z
%             %first calc the second half of the RHS (except beta)
%             RHSpart2=zeros(N_d,1);
%             for sprime_c=1:N_z
%                 if pi_z(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:,s_c,sprime_c)],sprime_c)*pi_z(s_c,sprime_c);
%                 end
%             end
%             for a_c=1:N_a
%                 entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %d by 1
%                 
%                 %then maximizing d indexes
%                 [V(a_c,s_c),Policy(a_c,s_c)]=max(entireRHS,[],1);
%             end
%         end
%         
%         PolicyIndexesPath(:,:,:,T-i)=Policy;
%         Vnext=V;
    end
    
    %Now we have the full PolicyIndexesPath, we go forward in time from 1
    %to T using the policies to generate the AggVarsPath. First though we
    %put in it's initial and final values.
    AggVarsPath=zeros(T,length(SSvaluesFn),'gpuArray');
    AggVarsPath(T,:)=SSvalues_AggVars_final;
    %Call AgentDist the current periods distn and AgentDistnext
    %the next periods distn which we must calculate
    AgentDist=AgentDist_initial;
    %Now we have the full PolicyIndexesPath, we go forward in time from 1
    %to T using the policies to update the agents distribution generating a
    %new price path
    for ii=1:T-1
        %Get the current optimal policy
        Policy=PolicyIndexesPath(:,:,ii);
        
        % optaprime is here replaced by Phi_of_Policy, which is a different shape
        Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
        for z_c=1:N_z
            Phi_of_Policy(:,:,z_c)=Phi_aprimeKron(Policy(:,z_c),:,z_c);
        end
        Ptemp=zeros(N_a,N_a*N_z*N_z,'gpuArray');
        Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a*N_z*N_z])+N_a*(gpuArray(0:1:N_a*N_z*N_z-1)))=1;
        Ptran=kron(pi_z',ones(N_a,N_a,'gpuArray')).*reshape(Ptemp,[N_a*N_z,N_a*N_z]);
        AgentDistnext=Ptran*AgentDist;
        
        p=PricePath(ii,:);
        
        if ~isnan(IndexesForPricePathInSSvalueParams)
            SSvalueParamsVec(IndexesForPricePathInSSvalueParams)=PricePath(ii,IndexesForSSvalueParamsInPricePath);
        end
        if ~isnan(IndexesForPathParamsInSSvalueParams)
            SSvalueParamsVec(IndexesForPathParamsInSSvalueParams)=ParamPath(ii,IndexesForSSvalueParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
        end
        PolicyTemp=UnKronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,unkronoptions);
        SSvalues_AggVars=SSvalues_AggVars_Case2_vec(AgentDist, PolicyTemp, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p, 2);
        
        AggVarsPath(ii,:)=SSvalues_AggVars;
        
        AgentDist=AgentDistnext;
    end
    
%     %Now we have the full PolicyIndexesPath, we go forward in time from 1
%     %to T using the policies to generate the AggVarsPath. First though we
%     %put in it's initial and final values.
%     AggVarsPath=zeros(length(SSvaluesFn),T+1);
%     AggVarsPath(:,1)=SSvalues_AggVars_initial; AggVarsPath(:,T+1)=SSvalues_AggVars_final;
%     %Call SteadyStateDist the current periods distn and SteadyStateDistnext
%     %the next periods distn which we must calculate
%     SteadyStateDist=AgentDist_initial;
%     for i=1:T-1
%         %Get the current optimal policy
%         Policy=PolicyIndexesPath(:,:,:,i);
%         %Use this to calculate the steady state distn
%         P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%         for s_c=1:N_z
%             for a_c=1:N_a
%                 optd=Policy(a_c,s_c);
%                 for sprime_c=1:N_z
%                     optaprime=Phi_aprimeKron(optd,s_c,sprime_c);
%                     P(a_c,s_c,optaprime,sprime_c)=pi_z(s_c,sprime_c)/sum(pi_z(s_c,:));
%                 end
%             end
%         end
%         P=reshape(P,[N_a*N_z,N_a*N_z]);
%         P=P';
%         SteadyStateDistnext=P*SteadyStateDist;
%         
%         AggVarsPath(:,i)=SSvalues_AggVars_Case2_raw(SteadyStateDist, Policy, SSvaluesFn, n_d, n_a, N_z, d_grid, a_grid,z_grid,pi_z,p); %the two zeros represent the d variables
%         
%         SteadyStateDist=SteadyStateDistnext;
%     end
    
%     %i=T
%     params=ParamPath(T,:);
%     p=PricePath(T,:);
%     Fmatrix=reshape(ReturnFn(p,params),[N_d,N_a,N_z]);
%     for s_c=1:N_z
%         %first calc the second half of the RHS (except beta)
%         RHSpart2=zeros(N_d,1);
%         for sprime_c=1:N_z
%             if pi_z(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                 RHSpart2=RHSpart2+V_final([Phi_aprimeKron(:,s_c,sprime_c)],sprime_c)*pi_z(s_c,sprime_c);
%             end
%         end
%         for a_c=1:N_a
%             entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %d by 1
%             
%             %then maximizing d indexes
%             [V(a_c,s_c),Policy(a_c,s_c)]=max(entireRHS,[],1);
%         end
%     end
%     
%     AggVarsPath(:,T)=SSvalues_AggVars_Case2_raw(SteadyStateDist, Policy, SSvaluesFn, n_d, n_a, N_z, d_grid, a_grid,z_grid,pi_z,p); %the two zeros represent the d variables
%     %end
    
end










% if Case2_Type==3
%     %First, go from T-1 to 1 calculating the Value function and Optimal
%     %policy function at each step. Since we won't need to keep the value
%     %functions for anything later we just store the next period one in
%     %Vnext, and the current period one to be calculated in V
%     Vnext=V_final;
%     for i=1:T-1 %so t=T-i
%         params=ParamPath(T-i,:);
%         p=PricePath(T-i,:);p=PricePath(ii,:);
%         if transpathoptions.returnmatrix==1
%             Fmatrix=ReturnFn(p,params);
%         elseif transpathoptions.returnmatrix==0
%             disp('Transition path does not yet support vfoptions.returnmatrix==0')
%         end
% %         Fmatrix=reshape(FmatrixFn(p,params),[N_d,N_a,N_s]);
%         
%         for s_c=1:N_s
%             %first calc the second half of the RHS (except beta)
%             RHSpart2=zeros(N_d,1);
%             for sprime_c=1:N_s
%                 if pi_s(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:)],sprime_c)*pi_s(s_c,sprime_c);
%                 end
%             end
%             for a_c=1:N_a
%                 entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %d by 1
%                 
%                 %then maximizing d indexes
%                 [V(a_c,s_c),PolicyIndexes(a_c,s_c)]=max(entireRHS,[],1);
%             end
%         end
%         
%         PolicyIndexesPath(:,:,:,T-i)=PolicyIndexes;
%         Vnext=V;
%     end
%     
%     %Now we have the full PolicyIndexesPath, we go forward in time from 1
%     %to T using the policies to generate the AggVarsPath. First though we
%     %put in it's initial and final values.
%     AggVarsPath=zeros(length(SSvaluesFn),T+1);
%     AggVarsPath(:,1)=SSvalues_AggVars_initial; AggVarsPath(:,T+1)=SSvalues_AggVars_final;
%     %Call SteadyStateDist the current periods distn and SteadyStateDistnext
%     %the next periods distn which we must calculate
%     SteadyStateDist=SteadyStateDist_initial;
%     for i=1:T-1
%         %Get the current optimal policy
%         PolicyIndexes=PolicyIndexesPath(:,:,:,i);
%         %Use this to calculate the steady state distn
%         P=zeros(N_a,N_s,N_a,N_s); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%         for s_c=1:N_s
%             for a_c=1:N_a
%                 optd=PolicyIndexes(a_c,s_c);
%                 optaprime=Phi_aprimeKron(optd);
%                 for sprime_c=1:N_s
%                     P(a_c,s_c,optaprime,sprime_c)=pi_s(s_c,sprime_c)/sum(pi_s(s_c,:));
%                 end
%             end
%         end
%         P=reshape(P,[N_a*N_s,N_a*N_s]);
%         P=P';
%         SteadyStateDistnext=P*SteadyStateDist;
%         
%         AggVarsPath(:,i)=SSvalues_AggVars_Case2_raw(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, N_s, d_grid, a_grid,s_grid,pi_s,p); %the two zeros represent the d variables
%         
%         SteadyStateDist=SteadyStateDistnext;
%     end
%     %i=T
%     params=ParamPath(T,:);
%     p=PricePath(T,:);
%     Fmatrix=reshape(FmatrixFn(p,params),[N_d,N_a,N_s]);
%     for s_c=1:N_s
%         %first calc the second half of the RHS (except beta)
%         RHSpart2=zeros(N_d,1);
%         for sprime_c=1:N_s
%             if pi_s(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                 RHSpart2=RHSpart2+V_final([Phi_aprimeKron(:)],sprime_c)*pi_s(s_c,sprime_c);
%             end
%         end
%         for a_c=1:N_a
%             entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %d by 1
%             
%             %then maximizing d indexes
%             [V(a_c,s_c),PolicyIndexes(a_c,s_c)]=max(entireRHS,[],1);
%         end
%     end
%     
%     AggVarsPath(:,T)=SSvalues_AggVars_Case2_raw(SteadyStateDist, PolicyIndexes, SSvaluesFn, n_d, n_a, N_s, d_grid, a_grid,s_grid,pi_s,p); %the two zeros represent the d variables
%     %end
%     
% end


end