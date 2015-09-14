function [PricePathNew]=TransitionPath_Case1_no_d(PricePathOld,IndexesForPricesInReturnFnParams, ParamPath, IndexesForPathParamsInReturnFnParams, beta, T, V_final, StationaryDist_init, ReturnFn, ReturnFnParams, n_a, n_z, pi_z, a_grid,z_grid, SSvaluesFn, MarketPriceEqns, MarketPriceParams,transpathoptions)

%% Check which transpathoptions have been used, set all others to defaults 
if nargin<19
    disp('No transpathoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-5);
    transpathoptions.parallel=2;
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.verbose=1;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    eval('fieldexists=1;transpathoptions.tolerance;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.tolerance=10^(-5);
    end
    eval('fieldexists=1;transpathoptions.parallel;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.parallel=2;
    end
    eval('fieldexists=1;transpathoptions.oldpathweight;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.oldpathweight=0.9;
    end
    eval('fieldexists=1;transpathoptions.weightscheme;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.weightscheme=1;
    end
    eval('fieldexists=1;transpathoptions.verbose;','fieldexists=0;')
    if fieldexists==0
        transpathoptions.verbose=1;
    end
end

%%

if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case2')
else
    a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
%     PricePathOld=gpuArray(PricePathOld);
end
unkronoptions.parallel=2;

N_z=prod(n_z);
N_a=prod(n_a);

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z]);
SteadyStateDist_initial=reshape(StationaryDist_init,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T)=PricePathOld(T);
PolicyIndexes=zeros(N_a,N_z,'gpuArray');


while PricePathDist>transpathoptions.tolerance
    PolicyIndexesPath=zeros(N_a,N_z,T-1); %Periods 1 to T-1
    
    PricePathOld
    
    %First, go from T-1 to 1 calculating the Value function and Optimal
    %policy function at each step. Since we won't need to keep the value
    %functions for anything later we just store the next period one in
    %Vnext, and the current period one to be calculated in V
    Vnext=V_final;
    for i=1:T-1 %so t=T-i
        
%         p=PricePathOld(T-i,:);
        ReturnFnParams(IndexesForPricesInReturnFnParams)=PricePathOld(T-i,:);
%         params=ParamPath(T-i,:);
        ReturnFnParams(IndexesForPathParamsInReturnFnParams)=ParamPath(T-i,:);
%         Fmatrix=reshape(FmatrixFn(p,params),[N_a,N_a,N_z]);
        ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid,ReturnFnParams);
        
        for z_c=1:N_z
            ReturnMatrix_z=ReturnMatrix(:,:,z_c);
            %Calc the condl expectation term (except beta), which depends on z but
            %not on control variables
            EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:)); %kron(ones(N_a,1),pi_z(z_c,:));
            EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
            EV_z=sum(EV_z,2);
            
            entireRHS=ReturnMatrix_z+beta*EV_z*ones(1,N_a,1); %aprime by 1
            
            %Calc the max and it's index
            [Vtemp,maxindex]=max(entireRHS,[],1);
            V(:,z_c)=Vtemp;
            PolicyIndexes(:,z_c)=maxindex;
            
        end
        
%         for z_c=1:N_z
%             %first calc the second half of the RHS (except beta)
%             RHSpart2=zeros(N_a,1); %aprime by kprime
%             for zprime_c=1:N_z
%                 if pi_z(z_c,zprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%                     RHSpart2=RHSpart2+Vnext(:,zprime_c)*pi_z(z_c,zprime_c)';
%                 end
%             end
%             for a_c=1:N_a
%                 
%                 entireRHS=Fmatrix(:,a_c,z_c)+beta*RHSpart2; %aprime by 1
%                 %calculate in order, the maximizing aprime indexes
%                 [V(a_c,z_c),PolicyIndexes(a_c,z_c)]=max(entireRHS,[],1);
%             end
%         end
        
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
        %Use this to calculate the steady state distn
%         P=zeros(N_a,N_z,N_a,N_z); %P(a,z,aprime,zprime)=proby of going to (a',z') given in (a,z)
%         for z_c=1:N_z
%             for a_c=1:N_a
%                 optaprime=PolicyIndexes(a_c,z_c);
%                 for zprime_c=1:N_z
%                     P(a_c,z_c,optaprime,zprime_c)=pi_z(z_c,zprime_c)/sum(pi_z(z_c,:));
%                 end
%             end
%         end
%         P=reshape(P,[N_a*N_z,N_a*N_z]);
%         P=P';
%         SteadyStateDistnext=P*SteadyStateDist;

        optaprime=reshape(PolicyIndexes,[1,N_a*N_z]);
        
        Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
        Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
        Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
        SteadyStateDistnext=Ptran*SteadyStateDist;
        
        p=PricePathOld(i,:);
        Policy=UnKronPolicyIndexes_Case1(PolicyIndexes, 0, n_a, n_z,unkronoptions);
        SSvalues_AggVars=SSvalues_AggVars_Case1_raw(SteadyStateDist, Policy, SSvaluesFn, 0, n_a, n_z, 0, a_grid,s_grid,pi_z,p); %the two zeros represent the d variables
        %An easy way to get the new prices is just to call MarketClearance
        %and then adjust it for the current prices
        for j=1:length(MarketPriceEqns)
            PricePathNew(i,j)=MarketPriceEqns{j}(SSvalues_AggVars,p, MarketPriceParams);
        end
        
        SteadyStateDist=SteadyStateDistnext;
    end
    
    %See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    %Notice that the distance is always calculated ignoring the time t=1 &
    %t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1
        disp('Old, New')
        [PricePathOld,gather(PricePathNew)]
    end
    %Set price path to be 9/10ths the old path and 1/10th the new path (but
    %making sure to leave prices in period T unchanged).
    PricePathOld(1:T-1)=transpathoptions.oldpathweight*PricePathOld(1:T-1)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1);
 
    pathcounter=pathcounter+1;
    TransPathConvergence=PricePathDist/PricePathTolerance; %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transtion path: %i \n',pathcounter)
        fprintf('Current distance to convergence: %.2f (convergence when reaches 1) \n',TransPathConvergence) %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
    end
    
    save ./SavedOutput/TransPathConv.mat TransPathConvergence pathcounter
end


end