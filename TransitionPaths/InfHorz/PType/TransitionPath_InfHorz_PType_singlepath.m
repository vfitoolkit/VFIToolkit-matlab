function AggVarsPath=TransitionPath_InfHorz_PType_singlepath(PricePathOld, ParamPath, PricePathNames, ParamPathNames, T, V_final, AgentDist_initial, ...
    l_d,N_d,n_d,N_a,n_a,N_z,n_z,d_grid,a_grid,d_gridvals,aprime_gridvals,a_gridvals,z_grid,pi_z,ReturnFn, ...
    FnsToEvaluate, ...
    Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, ...
    ... % use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, II1orII, II2, ...
    transpathoptions, vfoptions, simoptions)
% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% For this agent type, first go back through the value & policy fns.
% Then forwards through agent dist and agg vars.
if N_d>0
    PolicyIndexesPath=zeros(2,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
else
    PolicyIndexesPath=zeros(N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for tt=1:T-1 %so t=T-i
    % The following digs deeper into PricePathOld and ParamPath in
    % FHorz case--check it
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(T-tt,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-tt,kk);
    end
    
    [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The VKron input is next period value fn, the VKron output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')

    if N_d>0
        PolicyIndexesPath(:,:,:,T-tt)=Policy;
    else
        PolicyIndexesPath(:,:,T-tt)=Policy;
    end
    Vnext=V;
    
end
% Free up space on GPU by deleting things no longer needed
clear V Vnext

%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to update the agents distribution generating a
%new price path
%Call AgentDist the current periods distn
AgentDist=AgentDist_initial;
AggVarsPath=zeros(length(FnsToEvaluate),T-1);

% Precompute for later
if N_d>0
    II1=gpuArray(repmat(1:1:N_a*N_z,size(PolicyIndexesPath,1),1));
    IIones=ones(size(PolicyIndexesPath,1),N_a*N_z,1,'gpuArray');
else
    II1=gpuArray(1:1:N_a*N_z);
    IIones=ones(N_a*N_z,1,'gpuArray');
end

for tt=1:T-1
    
    %Get the current optimal policy
    if N_d>0
        Policy=PolicyIndexesPath(:,:,:,tt);
    else
        Policy=PolicyIndexesPath(:,:,tt);
    end
    
    GEprices=PricePathOld(tt,:);
    
    % Again, since we don't do tminus1, don't dig as deep as FHorz
    for nn=1:length(ParamPathNames)
        Parameters.(ParamPathNames{nn})=ParamPath(tt,nn);
    end
    for nn=1:length(PricePathNames)
        Parameters.(PricePathNames{nn})=PricePathOld(tt,nn);
    end
    
    PolicyUnKron=UnKronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,vfoptions);
    AggVars=EvalFnOnAgentDist_AggVars_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions); % The 2 is for Parallel (use GPU)
    
    AgentDist=AgentDist_InfHorz_TPath_SingleStep(AgentDist,Policy,II1,IIones,N_a,N_z,sparse(pi_z));

    AggVarsPath(:,tt)=AggVars;
end


end
