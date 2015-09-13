function [PricePathNew]=TransitionPath_Case2(PricePathOld,IndexesForPricesInReturnFnParams, ParamPath, IndexesForPathParamsInReturnFnParams, beta, Phi_aprimeKron_final, Case2_Type, T, V_final, StationaryDist_init, ReturnFn, ReturnFnParams, n_d, n_a, n_z, pi_z, d_grid,a_grid,z_grid, SSvaluesFn, MarketPriceEqns, MarketPriceParams,transpathoptions)
                      %=TransitionPath_Case2(PricePathTolerance, PricePathOld, n_d, n_a, n_z, pi_z, d_grid,a_grid,s_grid,beta, Phi_aprimeKron, Case2_Type, T, ParamPath, V_final, SteadyStateDist_initial, ReturnFn, SSvaluesFn, MarketPriceEqns, MarketPriceParams, transpathoptions)
%Even though this function will only ever likely be called for heterogenous
%agent models (and so we will have s instead of z) I have left the notation
%here as z.

%transpathoptions.tolerance
%transpathoptions.parallel
%transpathoptions.oldpathweight % default =0.9
%transpathoptions.weightscheme % default =1

Phi_aprimeKron=Phi_aprimeKron_final; % Might want to change this so that Phi_aprimeKron can change along the transition path.

if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case2')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
%     PricePathOld=gpuArray(PricePathOld);
end
unkronoptions.parallel=2;

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z]);
SteadyStateDist_initial=reshape(StationaryDist_init,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T)=PricePathOld(T);
PolicyIndexes=zeros(N_a,N_z,'gpuArray');

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
        
%         PricePathOld
        
        %First, go from T-1 to 1 calculating the Value function and Optimal
        %policy function at each step. Since we won't need to keep the value
        %functions for anything later we just store the next period one in
        %Vnext, and the current period one to be calculated in V
        Vnext=V_final;
        for i=1:T-1 %so t=T-i
%            params=ParamPath(T-i,:);
%            p=PricePathOld(T-i,:);
            ReturnFnParams(IndexesForPricesInReturnFnParams)=PricePathOld(T-i,:);
            ReturnFnParams(IndexesForPathParamsInReturnFnParams)=ParamPath(T-i,:);
            
            ReturnMatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParams);
            
%             if transpathoptions.returnmatrix==1
%                 Fmatrix=ReturnFn(p,params);
%             elseif transpathoptions.returnmatrix==0
%                 disp('Transition path does not yet support vfoptions.returnmatrix==0')
%             end
%             Fmatrix=reshape(ReturnFn(p,params),[N_d,N_a,N_z]);

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
                PolicyIndexes(:,z_c)=maxindex;
                
%                 tempmaxindex=maxindex+(0:1:N_a-1)*(N_d)+(z_c-1)*N_d*N_a;
%                 Ftemp(:,z_c)=ReturnMatrix(tempmaxindex);
            end
                        
%             for z_c=1:N_z
%                 %first calc the second half of the RHS (except beta)
%                 RHSpart2=zeros(N_d,1);
%                 for zprime_c=1:N_z
%                     if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                         RHSpart2=RHSpart2+Vnext([Phi_aprimeKron(:,z_c,zprime_c)],zprime_c)*pi_z(z_c,zprime_c);
%                     end
%                 end
%                 for a_c=1:N_a
%                     entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %d by 1
%                     
%                     %then maximizing d indexes
%                     [V(a_c,z_c),PolicyIndexes(a_c,z_c)]=max(entireRHS,[],1);
%                 end
%             end
            
            PolicyIndexesPath(:,:,T-i)=PolicyIndexes;
            Vnext=V;
        end
        
        
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution generating a
        %new price path
        %Call SteadyStateDist the current periods distn and SteadyStateDistnext
        %the next periods distn which we must calculate
        SteadyStateDist=SteadyStateDist_initial;
        for i=1:T-1
            %Get the current optimal policy
            PolicyIndexes=PolicyIndexesPath(:,:,i);
%             %Use this to calculate the steady state distn
%             P=zeros(N_a*N_z,N_a*N_z);
%             P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%             parfor az_c=1:N_a*N_z
%                 z_c=ceil(az_c/N_a);
%         
%                 optd=PolicyIndexes(az_c);
%                 Prow=zeros(1,N_a*N_z);
%                 for zprime_c=1:N_z
%                     optaprime=Phi_aprimeKron(optd,z_c,zprime_c);
%                     optaprimezprime_c=optaprime+(zprime_c-1)*N_a;
%                     Prow(optaprimezprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%                 end
%                 P(az_c,:)=Prow;
%             end
%             Ptransposed=P';
%             SteadyStateDistnext=Ptransposed*SteadyStateDist;
            
            % optaprime is here replaced by Phi_of_Policy, which is a different shape
            Phi_of_Policy=zeros(N_a,N_z,N_z,'gpuArray'); %a'(a,z',z)
            for z_c=1:N_z
                Phi_of_Policy(:,:,z_c)=Phi_aprimeKron(PolicyIndexes(:,z_c),:,z_c);
            end
            %        Phi_aprimeKron % aprime(d,zprime,z)
            Ptemp=zeros(N_a,N_a*N_z*N_z,'gpuArray');
            Ptemp(reshape(permute(Phi_of_Policy,[2,1,3]),[1,N_a*N_z*N_z])+N_a*(gpuArray(0:1:N_a*N_z*N_z-1)))=1;
            %        Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
            Ptran=kron(pi_z',ones(N_a,N_a,'gpuArray')).*reshape(Ptemp,[N_a*N_z,N_a*N_z]);
            SteadyStateDistnext=Ptran*SteadyStateDist;

            
            
            p=PricePathOld(i,:);
            Policy=UnKronPolicyIndexes_Case2(PolicyIndexes, n_d, n_a, n_z,unkronoptions);
            SSvalues_AggVars=SSvalues_AggVars_Case2(SteadyStateDist, Policy, SSvaluesFn, n_d, n_a, n_z, d_grid, a_grid,z_grid,pi_z,p);
            %An easy way to get the new prices is just to call MarketClearance
            %and then adjust it for the current prices
            for j=1:length(MarketPriceEqns)
                PricePathNew(i,j)=MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParams);
            end
            
            SteadyStateDist=SteadyStateDistnext;
        end
        
        %See how far apart the price paths are
        PricePathDist=max(abs(reshape(PricePathNew-PricePathOld,[numel(PricePathOld),1])));        
%         PricePathDist=sum(abs(reshape(PricePathNew-PricePathOld,[numel(PricePathOld),1])));
        
        disp('Old, New')
        [PricePathOld,gather(PricePathNew)]
        %Set price path to be 9/10ths the old path and 1/10th the new path (but
        %making sure to leave prices in periods 1 & T unchanged).
        
        if transpathoptions.weightscheme==1 % Just a constant weighting
            PricePathOld(1:T-1)=transpathoptions.oldpathweight*PricePathOld(1:T-1)+(1-transpathoptions.oldpathweight)*gather(PricePathNew(1:T-1));
        elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        
        
        PricePathOld(1:T-1)=(transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.1),T-1)))*(1-transpathoptions.oldpathweight))*PricePathOld(1:T-1)+exp(linspace(0,log(0.1),T-1))*(1-transpathoptions.oldpathweight)*gather(PricePathNew(1:T-1));
        
        pathcounter=pathcounter+1;
        TransPathConvergence=PricePathDist/transpathoptions.tolerance %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
        save ./SavedOutput/TransPathConv.mat PricePathOld TransPathConvergence pathcounter
    end
    save ./SavedOutput/TransPath_Policy.mat PolicyIndexesPath
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