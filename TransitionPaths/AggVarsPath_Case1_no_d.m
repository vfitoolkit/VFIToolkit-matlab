function AggVarsPath=AggVarsPath_Case1_no_d(SSvaluesFn, SSvalueParamNames,PricePath,PriceParamNames, ParamPath, PathParamNames, Parameters, n_a, n_z, pi_z, a_grid,z_grid, DiscountFactorParamNames, T, V_final, AgentDist_initial, ReturnFn, ReturnFnParamNames,SSvalues_AggVars_final)
%AggVarsPath is T periods long (period 0 would be before the reforms are announced). Period 1 is thus once the whole reforms 
% path (prices and params) is know, but with the agents distribution still being it's inital value. T is imposed as the final values.

N_z=prod(n_z);
N_a=prod(n_a);

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray');

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

PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for ii=1:T-1 %so t=T-i
    
    if ~isnan(IndexesForPathParamsInDiscountFactor)
        beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-i,IndexesForDiscountFactorInPathParams); % This step could be moved outside all the loops
    end
    if ~isnan(IndexesForPricePathInReturnFnParams)
    ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePath(T-ii,IndexesForReturnFnParamsInPricePath);
    end
    if ~isnan(IndexesForPathParamsInReturnFnParams)
        ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-ii,IndexesForReturnFnParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
    end
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, 0, n_a, n_z, 0, a_grid, z_grid,ReturnFnParamsVec);
    
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
        Policy(:,z_c)=maxindex;
        
    end
    
    PolicyIndexesPath(:,:,T-ii)=Policy;
    Vnext=V;
end


%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to generate the AggVarsPath. First though we
%put in it's initial and final values.
AggVarsPath=zeros(T,length(SSvaluesFn),'gpuArray');
AggVarsPath(T,:)=SSvalues_AggVars_final;
%Call AgentDist the current periods distn and SteadyStateDistnext
%the next periods distn which we must calculate
AgentDist=AgentDist_initial;
%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to update the agents distribution generating a
%new price path
for ii=1:T-1
    %Get the current optimal policy
    Policy=PolicyIndexesPath(:,:,ii);
    
    optaprime=reshape(Policy,[1,N_a*N_z]);
    
    Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
    Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
    AgentDistnext=Ptran*AgentDist;
    
    p=PricePath(ii,:);
    
    if ~isnan(IndexesForPricePathInSSvalueParams)
        SSvalueParamsVec(IndexesForPricePathInSSvalueParams)=PricePathOld(ii,IndexesForSSvalueParamsInPricePath);
    end
    if ~isnan(IndexesForPathParamsInSSvalueParams)
        SSvalueParamsVec(IndexesForPathParamsInSSvalueParams)=ParamPath(ii,IndexesForSSvalueParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
    end
    PolicyTemp=UnKronPolicyIndexes_Case1(Policy, 0, n_a, n_z,unkronoptions);
    SSvalues_AggVars=SSvalues_AggVars_Case1_vec(AgentDist, PolicyTemp, SSvaluesFn, SSvalueParamsVec, 0, n_a, n_z, 0, a_grid, z_grid, pi_z,p, 2);
    
    AggVarsPath(ii,:)=SSvalues_AggVars;
    
    AgentDist=AgentDistnext;
end
% %i=T
% p=PricePath(T,:);
% if ~isnan(IndexesForPricesInSSvalueParamsVec)
%     SSvalueParamsVec(IndexesForPricesInSSvalueParamsVec)=PricePathOld(T,:);
% end
% if ~isnan(IndexesForPathParamsInSSvalueParamsVec)
%     SSvalueParamsVec(IndexesForPathParamsInSSvalueParamsVec)=ParamPath(T,:); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
% end
% AggVarsPath(:,T)=SSvalues_AggVars_Case1_vec(AgentDist, Policy_final, SSvaluesFn, SSvalueParamsVec, 0, n_a, n_z, 0, a_grid, z_grid, pi_z,p, 2);



end