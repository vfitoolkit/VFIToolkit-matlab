function WeightedSumSq_GeneralEqmCondnPath=TransitionPath_Case1_FHorz_no_d_subfn(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

PricePathOld=reshape(PricePathOld,[T,length(PricePathNames)]); % Had to be inputed as a vector to allow use of fminsearch.

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePathOld,2);

PricePathDist=Inf;
pathcounter=1;

V_final=reshape(V_final,[N_a,N_z,N_j]);
AgentDist_initial=reshape(StationaryDist_init,[N_a*N_z,N_j]);
V=zeros(size(V_final),'gpuArray'); %preallocate space
GeneralEqmCondnPath=nan(size(PricePathOld),'gpuArray'); GeneralEqmCondnPath(T,:)=0;
Policy=zeros(N_a,N_z,N_j,'gpuArray');

PolicyIndexesPath=zeros(N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
    
%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for i=1:T-1 %so t=T-i
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(T-i,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-i,kk);
    end
    
    [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The VKron input is next period value fn, the VKron output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')
    
    if N_d>0
        PolicyIndexesPath(:,:,:,:,T-i)=Policy;
    else
        PolicyIndexesPath(:,:,:,T-i)=Policy;
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
for i=1:T-1
    
    %Get the current optimal policy
    if N_d>0
        Policy=PolicyIndexesPath(:,:,:,:,i);
    else
        Policy=PolicyIndexesPath(:,:,:,i);
    end
    
    p=PricePathOld(i,:);
    
    for nn=1:length(ParamPathNames)
        Parameters.(ParamPathNames{nn})=ParamPath(i,nn);
    end
    for nn=1:length(PricePathNames)
        Parameters.(PricePathNames{nn})=PricePathOld(i,nn);
    end
    
    PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 2); % The 2 is for Parallel (use GPU)
    
    
    % When using negative powers matlab will often return complex
    % numbers, even if the solution is actually a real number. I
    % force converting these to real, albeit at the risk of missing problems
    % created by actual complex numbers.
    GeneralEqmCondnPath(i,:)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));
    
    AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
end
%     % Free up space on GPU by deleting things no longer needed
%     clear AgentDist
    

%     if transpathoptions.historyofpricepath==1
%         PricePathHistory{pathcounter,1}=PricePathDist;
%         PricePathHistory{pathcounter,2}=PricePathOld;
%         
%         if rem(pathcounter,5)==1
%             save ./SavedOutput/TransPath_Internal.mat PricePathHistory
%         end
%     end
% 
%     pathcounter=pathcounter+1;

WeightedSumSq_GeneralEqmCondnPath=sum(sum(transpathoptions.weightsforpath.*(GeneralEqmCondnPath).^2));    

% end

WeightedSumSq_GeneralEqmCondnPath=gather(WeightedSumSq_GeneralEqmCondnPath);

if transpathoptions.verbose==1
    fprintf('Current PricePath: \n')
    PricePathOld
    fprintf('Current GeneralEqmCondnPath: \n')
    GeneralEqmCondnPath
    fprintf('Current WeightedSumSq_GeneralEqmCondnPath: \n')
    WeightedSumSq_GeneralEqmCondnPath
end

end