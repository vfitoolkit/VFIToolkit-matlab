function AggVarsPath=EvalFnOnTransPath_AggVars_Case1_FHorz_noz(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, Parameters, T, n_d, n_a, N_j, d_grid, a_grid, transpathoptions, simoptions)
% AggVarsPath is T periods long (periods 0 (before the reforms are announced) & T are the initial and final values.
% 
% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

%%
N_d=prod(n_d);
N_a=prod(n_a);
if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+1:end}}; % the first inputs will always be (d,aprime,a)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
if transpathoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   if N_d>0
       d_grid=gpuArray(d_grid);
   end
   a_grid=gpuArray(a_grid);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   if N_d>0
       d_grid=gather(d_grid);
   end
   a_grid=gather(a_grid);
end

if ~strcmp(transpathoptions.exoticpreferences,'None')
    error('Only transpathoptions.exoticpreferences==None is supported by TransitionPath_Case1')
end


l_p=size(PricePath,2);

if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
    ParamPathNames
    PricePathNames
end

AgentDistPath=reshape(AgentDistPath,[N_a,N_j,T]);
PolicyPath=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j,T);

if transpathoptions.parallel==2
    
    if FnsToEvaluateStruct==0
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        AggVarsPath=zeros(T,length(FnsToEvaluate),'gpuArray');
        for tt=1:T
            AgentDist=AgentDistPath(:,:,tt);
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyPath(:,:,:,tt);
            else
                Policy=PolicyPath(:,:,tt);
            end
                        
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz_noz(Policy, n_d, n_a, N_j,simoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_noz(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, N_j, d_grid, a_grid, 2, simoptions); % The 2 is for Parallel (use GPU)
            
            AggVarsPath(tt,:)=AggVars;
        end
    else % FnsToEvaluateStruct==1
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        for ff=1:length(AggVarNames)
            AggVarsPath.(AggVarNames{ff}).Mean=zeros(T,1,'gpuArray');
        end

        for tt=1:T
            AgentDist=AgentDistPath(:,:,tt);
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyPath(:,:,:,tt);
            else
                Policy=PolicyPath(:,:,tt);
            end
                        
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz_noz(Policy, n_d, n_a, N_j,simoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_noz(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, N_j, d_grid, a_grid, 2, simoptions); % The 2 is for Parallel (use GPU)

            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars(ff);
            end
        end
    end

else
    
    if FnsToEvaluateStruct==0
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        AggVarsPath=zeros(T,length(FnsToEvaluate));

        for tt=1:T
            AgentDist=AgentDistPath(:,:,tt);
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyPath(:,:,:,tt);
            else
                Policy=PolicyPath(:,:,tt);
            end
                        
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz_noz(Policy, n_d, n_a, N_j,simoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_noz(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, N_j, d_grid, a_grid, 1, simoptions); % The 1 is for Parallel (use CPU)
            
            AggVarsPath(tt,:)=AggVars;
        end
    else % FnsToEvaluateStruct==1
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        for ff=1:length(AggVarNames)
            AggVarsPath.(AggVarNames{ff}).Mean=zeros(T,1);
        end

        for tt=1:T
            AgentDist=AgentDistPath(:,:,tt);
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyPath(:,:,:,tt);
            else
                Policy=PolicyPath(:,:,tt);
            end
                        
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz_noz(Policy, n_d, n_a, N_j,simoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1_noz(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, N_j, d_grid, a_grid, 1, simoptions); % The 1 is for Parallel (use CPU)
            
            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars(tt,ff);
            end
        end
    end
end

end