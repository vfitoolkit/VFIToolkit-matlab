function AggVarsPath=EvalFnOnTransPath_AggVars_Case1(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,ParamPath, Parameters, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames,transpathoptions)
%AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values).

if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=2;
    transpathoptions.lowmemory=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'lowmemory')==0
        transpathoptions.lowmemory=0;
    end
end

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);

% Internally PricePathOld is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathNames=fieldnames(PricePathOld);
PricePathStruct=PricePathOld; 
PricePathOld=zeros(T,length(PricePathNames));
for ii=1:length(PricePathNames)
    PricePathOld(:,ii)=PricePath.(PricePathNames{ii});
end
ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath; 
ParamPath=zeros(T,length(ParamPathNames));
for ii=1:length(ParamPathNames)
    ParamPath(:,ii)=ParamPath.(PricePathNames{ii});
end

if N_d==0
    AggVarsPath=EvalFnOnTransPath_AggVars_Case1_no_d(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames, ParamPath, ParamPathNames, Parameters, T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames);
    return
end

if transpathoptions.lowmemory==1
    % The lowmemory option is going to use gpu (but loop over z instead of
    % parallelize) for value fn, and then use sparse matrices on cpu when iterating on agent dist.
    AggVarsPath=EvalFnOnTransPath_AggVars_Case1_lowmem(FnsToEvaluate, FnsToEvaluateParamNames,PricePath,PricePathNames, ParamPath, ParamPathNames, Parameters, T, V_final, AgentDist_initial, n_d, n_a, n_z, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames,transpathoptions);
    return
end

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,1]);
V=zeros(size(V_final),'gpuArray');
Policy=zeros(N_a,N_z,'gpuArray');

unkronoptions.parallel=2;

beta=CreateVectorFromParams(Parameters, DiscountFactorParamNames);
IndexesForPathParamsInDiscountFactor=CreateParamVectorIndexes(DiscountFactorParamNames, ParamPathNames);
IndexesForDiscountFactorInPathParams=CreateParamVectorIndexes(ParamPathNames,DiscountFactorParamNames);
ReturnFnParamsVec=gpuArray(CreateVectorFromParams(Parameters, ReturnFnParamNames));
IndexesForPricePathInReturnFnParams=CreateParamVectorIndexes(ReturnFnParamNames, PricePathNames);
IndexesForReturnFnParamsInPricePath=CreateParamVectorIndexes(PricePathNames, ReturnFnParamNames);
IndexesForPathParamsInReturnFnParams=CreateParamVectorIndexes(ReturnFnParamNames, ParamPathNames);
IndexesForReturnFnParamsInPathParams=CreateParamVectorIndexes(ParamPathNames,ReturnFnParamNames);

PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for ii=0:T-1 %so t=T-i
    
    if ~isnan(IndexesForPathParamsInDiscountFactor)
        beta(IndexesForPathParamsInDiscountFactor)=ParamPath(T-ii,IndexesForDiscountFactorInPathParams); % This step could be moved outside all the loops
    end
    if ~isnan(IndexesForPricePathInReturnFnParams)
    ReturnFnParamsVec(IndexesForPricePathInReturnFnParams)=PricePath(T-ii,IndexesForReturnFnParamsInPricePath);
    end
    if ~isnan(IndexesForPathParamsInReturnFnParams)
        ReturnFnParamsVec(IndexesForPathParamsInReturnFnParams)=ParamPath(T-ii,IndexesForReturnFnParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
    end    
    ReturnMatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec);
    
    for z_c=1:N_z
        ReturnMatrix_z=ReturnMatrix(:,:,z_c);
        %Calc the condl expectation term (except beta), which depends on z but
        %not on control variables
        EV_z=Vnext.*(ones(N_a,1,'gpuArray')*pi_z(z_c,:));
        EV_z(isnan(EV_z))=0; %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
        EV_z=sum(EV_z,2);
        
        entireEV_z=kron(EV_z,ones(N_d,1));
        entireRHS=ReturnMatrix_z+beta*entireEV_z*ones(1,N_a,1);
        
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
AggVarsPath=zeros(T,length(FnsToEvaluate),'gpuArray');
% AggVarsPath(T,:)=SSvalues_AggVars_final;
%Call AgentDist the current periods distn and AgentDistnext
%the next periods distn which we must calculate
AgentDist=AgentDist_initial;
%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to update the agents distribution generating a
%new price path
for ii=1:T%-1
    %Get the current optimal policy
    Policy=PolicyIndexesPath(:,:,ii);
    
    optaprime=shiftdim(ceil(Policy/N_d),-1); % This shipting of dimensions is probably not necessary
    optaprime=reshape(optaprime,[1,N_a*N_z]);
    
    Ptemp=zeros(N_a,N_a*N_z,'gpuArray');
    Ptemp(optaprime+N_a*(gpuArray(0:1:N_a*N_z-1)))=1;
    Ptran=(kron(pi_z',ones(N_a,N_a,'gpuArray'))).*(kron(ones(N_z,1,'gpuArray'),Ptemp));
    AgentDistnext=Ptran*AgentDist;
    
%     p=PricePath(ii,:);
%     
%     if ~isnan(IndexesForPricePathInFnsToEvaluateParams)
%         FnsToEvaluateParamsVec(IndexesForPricePathInFnsToEvaluateParams)=PricePath(ii,IndexesForFnsToEvaluateParamsInPricePath);
%     end
%     if ~isnan(IndexesForPathParamsInFnsToEvaluateParams)
%         FnsToEvaluateParamsVec(IndexesForPathParamsInFnsToEvaluateParams)=ParamPath(ii,IndexesForFnsToEvaluateParamsInPathParams); % This step could be moved outside all the loops by using BigReturnFnParamsVec idea
%     end
%     
%     % The next five lines should really be replaced with a custom
%     % alternative version of SSvalues_AggVars_Case1_vec that can
%     % operate directly on Policy, rather than present messing around
%     % with converting to PolicyTemp and then using
%     % UnKronPolicyIndexes_Case1.
%     % Current approach is likely way suboptimal speedwise.
    PolicyTemp=zeros(2,N_a,N_z,'gpuArray'); %NOTE: this is not actually in Kron form
    PolicyTemp(1,:,:)=shiftdim(rem(Policy-1,N_d)+1,-1);
    PolicyTemp(2,:,:)=shiftdim(ceil(Policy/N_d),-1);
    
%     SSvalues_AggVars=SSvalues_AggVars_Case1_vec(AgentDist, PolicyTemp, SSvaluesFn, FnsToEvaluateParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p, 2);

    for jj=1:size(PricePath,2)
        Parameters.(PricePathNames{jj})=PricePath(ii,jj);
    end
    for jj=1:size(ParamPath,2)
        Parameters.(ParamPathNames{jj})=ParamPath(ii,jj);
    end
    
    PolicyTemp=UnKronPolicyIndexes_Case1(PolicyTemp, n_d, n_a, n_z,unkronoptions);
    AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, PolicyTemp, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, 2);

    AggVarsPath(ii,:)=AggVars;
    
    AgentDist=AgentDistnext;
end
%i=T
% params=ParamPath(T,:);
% p=PricePath(T,:);
% Fmatrix=reshape(ReturnFn(p,params),[N_a,N_a,N_z]);
% for s_c=1:N_z
%     %first calc the second half of the RHS (except beta)
%     RHSpart2=zeros(N_a,1); %aprime by kprime
%     for sprime_c=1:N_z
%         if pi_z(s_c,sprime_c)~=0 %multilications of -Inf with 0 gives NaN, this replaces them with zeros (as the zeros come from the transition probabilites)
%             RHSpart2=RHSpart2+V_final(:,sprime_c)*pi_z(s_c,sprime_c)';
%         end
%     end
%     for a_c=1:N_a
%         entireRHS=Fmatrix(:,a_c,s_c)+beta*RHSpart2; %aprime by 1
%         
%         %calculate in order, the maximizing aprime indexes
%         [V(a_c,s_c),Policy(1,a_c,s_c)]=max(entireRHS,[],1);
%     end
% end
% AggVarsPath(:,T)=SSvalues_AggVars_Case1_raw(AgentDist, Policy, SSvaluesFn, 0, n_a, N_z, 0, a_grid,z_grid,pi_z,p); %the two zeros represent the d variables
%end




end