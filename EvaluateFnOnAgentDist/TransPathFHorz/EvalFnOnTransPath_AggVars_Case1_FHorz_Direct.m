function AggVarsPath=EvalFnOnTransPath_AggVars_Case1_FHorz_Direct(FnsToEvaluate, PricePath, ParamPath, Parameters, T, V_final, Policy_final, AgentDist_initial, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, AgeWeightsParamNames, transpathoptions)
%AggVarsPath is T-1 periods long (periods 0 (before the reforms are announced) & T are the initial and final values; they are not created by this command and instead can be used to provide double-checks of the output (the T-1 and the final should be identical if convergence has occoured).
% To fix idea of the size/shape: AggVarsPath=nan(T,length(FnsToEvaluate));

% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePath is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    transpathoptions.exoticpreferences='None';
    transpathoptions.lowmemory=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=1+(gpuDeviceCount>0); % GPU where available, otherwise parallel CPU.
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences='None';
    end
    if isfield(transpathoptions,'lowmemory')==0
        transpathoptions.lowmemory=0;
    end
end

%%
N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
if isempty(n_d)
    l_d=0;
elseif n_d(1)==0
    l_d=0;
else
    l_d=1;
end
l_a=length(n_a);
l_z=length(n_z);

%%

% Note: Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'. 
PricePathNames=fieldnames(PricePath);
PricePathStruct=PricePath; 
PricePathSizeVec=zeros(1,length(PricePathNames)); % Allows for a given price param to depend on age (or permanent type)
for tt=1:length(PricePathNames)
    temp=PricePathStruct.(PricePathNames{tt});
    tempsize=size(temp);
    PricePathSizeVec(tt)=tempsize(tempsize~=T); % Get the dimension which is not T
end
PricePathSizeVec=cumsum(PricePathSizeVec);
if length(PricePathNames)>1
    PricePathSizeVec=[[1,PricePathSizeVec(1:end-1)+1];PricePathSizeVec];
else
    PricePathSizeVec=[1;PricePathSizeVec];
end
PricePath=zeros(T,PricePathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for tt=1:length(PricePathNames)
    if size(PricePathStruct.(PricePathNames{tt}),1)==T
        PricePath(:,PricePathSizeVec(1,tt):PricePathSizeVec(2,tt))=PricePathStruct.(PricePathNames{tt});
    else % Need to transpose
        PricePath(:,PricePathSizeVec(1,tt):PricePathSizeVec(2,tt))=PricePathStruct.(PricePathNames{tt})';
    end
end

ParamPathNames=fieldnames(ParamPath);
ParamPathStruct=ParamPath;
ParamPathSizeVec=zeros(1,length(ParamPathNames)); % Allows for a given price param to depend on age (or permanent type)
for tt=1:length(ParamPathNames)
    temp=ParamPathStruct.(ParamPathNames{tt});
    tempsize=size(temp);
    ParamPathSizeVec(tt)=tempsize(tempsize~=T); % Get the dimension which is not T
end
ParamPathSizeVec=cumsum(ParamPathSizeVec);
if length(ParamPathNames)>1
    ParamPathSizeVec=[[1,ParamPathSizeVec(1:end-1)+1];ParamPathSizeVec];
else
    ParamPathSizeVec=[1;ParamPathSizeVec];
end
ParamPath=zeros(T,ParamPathSizeVec(2,end));% Do this seperately afterwards so that can preallocate the memory
for tt=1:length(ParamPathNames)
    if size(ParamPathStruct.(ParamPathNames{tt}),1)==T
        ParamPath(:,ParamPathSizeVec(1,tt):ParamPathSizeVec(2,tt))=ParamPathStruct.(ParamPathNames{tt});
    else % Need to transpose
        ParamPath(:,ParamPathSizeVec(1,tt):ParamPathSizeVec(2,tt))=ParamPathStruct.(ParamPathNames{tt})';
    end
end

PricePath=struct();

%%
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

%% Check which vfoptions have been used, set all others to defaults 
if isfield(transpathoptions,'vfoptions')==1
    vfoptions=transpathoptions.vfoptions;
end

if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
    vfoptions.parallel=transpathoptions.parallel;
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.exoticpreferences='None';
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    end
    if vfoptions.parallel==2
        vfoptions.returnmatrix=2; % On GPU, must use this option
    end
    if isfield(vfoptions,'lowmemory')==0
        vfoptions.lowmemory=0;
    end
    if isfield(vfoptions,'verbose')==0
        vfoptions.verbose=0;
    end
    if isfield(vfoptions,'returnmatrix')==0
        if isa(ReturnFn,'function_handle')==1
            vfoptions.returnmatrix=0;
        else
            vfoptions.returnmatrix=1;
        end
    end
    if isfield(vfoptions,'exoticpreferences')==0
        vfoptions.exoticpreferences='None';
    end
    if isfield(vfoptions,'polindorval')==0
        vfoptions.polindorval=1;
    end
    if isfield(vfoptions,'policy_forceintegertype')==0
        vfoptions.policy_forceintegertype=0;
    end
end

%% Check which simoptions have been used, set all others to defaults 
if isfield(transpathoptions,'simoptions')==1
    simoptions=transpathoptions.simoptions;
end

if exist('simoptions','var')==0
    simoptions.nsims=10^4;
    simoptions.parallel=transpathoptions.parallel; % GPU where available, otherwise parallel CPU.
    simoptions.verbose=0;
    try 
        PoolDetails=gcp;
        simoptions.ncores=PoolDetails.NumWorkers;
    catch
        simoptions.ncores=1;
    end
    simoptions.iterate=1;
    simoptions.tolerance=10^(-9);
else
    %Check vfoptions for missing fields, if there are some fill them with
    %the defaults
    if isfield(simoptions,'tolerance')==0
        simoptions.tolerance=10^(-9);
    end
    if isfield(simoptions,'nsims')==0
        simoptions.nsims=10^4;
    end
    if isfield(simoptions,'parallel')==0
        simoptions.parallel=transpathoptions.parallel;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'ncores')==0
        try
            PoolDetails=gcp;
            simoptions.ncores=PoolDetails.NumWorkers;
        catch
            simoptions.ncores=1;
        end
    end
    if isfield(simoptions,'iterate')==0
        simoptions.iterate=1;
    end
end

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

% Create ReturnFnParamNames
temp=getAnonymousFnInputNames(ReturnFn);
if length(temp)>(l_d+l_a+l_a+l_z)
    ReturnFnParamNames={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
else
    ReturnFnParamNames={};
end


%%
if transpathoptions.parallel==2 
   % If using GPU make sure all the relevant inputs are GPU arrays (not standard arrays)
   pi_z=gpuArray(pi_z);
   if N_d>0
       d_grid=gpuArray(d_grid);
   end
   a_grid=gpuArray(a_grid);
   z_grid=gpuArray(z_grid);
%    PricePath=gpuArray(PricePath);
else
   % If using CPU make sure all the relevant inputs are CPU arrays (not standard arrays)
   % This may be completely unnecessary.
   pi_z=gather(pi_z);
   if N_d>0
       d_grid=gather(d_grid);
   end
   a_grid=gather(a_grid);
   z_grid=gather(z_grid);
%    PricePath=gather(PricePath);
end

if ~strcmp(transpathoptions.exoticpreferences,'None')
    disp('ERROR: Only transpathoptions.exoticpreferences==None is supported by TransitionPath_Case1')
    dbstack
end


l_p=size(PricePath,2);

if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

if transpathoptions.parallel==2
    
    V_final=reshape(V_final,[N_a,N_z,N_j]);
    AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]);
%     V=zeros(size(V_final),'gpuArray'); %preallocate space
    PricePathNew=zeros(size(PricePath),'gpuArray'); PricePathNew(T,:)=PricePath(T,:);
%     if N_d>0
%         Policy=zeros(2,N_a,N_z,N_j,'gpuArray');
%     else
%         Policy=zeros(N_a,N_z,N_j,'gpuArray');
%     end
    %%
    
    if N_d>0
        PolicyIndexesPath=zeros(2,N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1 will be calculated
        PolicyIndexesPath(:,:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
    else
        PolicyIndexesPath=zeros(N_a,N_z,N_j,T,'gpuArray'); %Periods 1 to T-1 will be calculated
        PolicyIndexesPath(:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
    end
    
    %First, go from T-1 to 1 calculating the Value function and Optimal
    %policy function at each step. Since we won't need to keep the value
    %functions for anything later we just store the next period one in
    %Vnext, and the current period one to be calculated in V
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i
        
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        
        [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')
                
        if N_d>0
            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
        else
            PolicyIndexesPath(:,:,:,T-ttr)=Policy;
        end
        Vnext=V;
    end
    % Free up space on GPU by deleting things no longer needed
    clear V Vnext

    
    if FnsToEvaluateStruct==0
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        AggVarsPath=zeros(T,length(FnsToEvaluate),'gpuArray');
        %Call AgentDist the current periods distn
        AgentDist=AgentDist_initial;
        for tt=1:T%-1
            
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyIndexesPath(:,:,:,:,tt);
            else
                Policy=PolicyIndexesPath(:,:,:,tt);
            end
            
            p=PricePath(tt,:);
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 2); % The 2 is for Parallel (use GPU)
            
            AggVarsPath(tt,:)=AggVars;
            
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
        end
    else % FnsToEvaluateStruct==1
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        for ff=1:length(AggVarNames)
            AggVarsPath.(AggVarNames{ff}).Mean=zeros(T,1,'gpuArray');
        end
        %Call AgentDist the current periods distn
        AgentDist=AgentDist_initial;
        for tt=1:T%-1
            
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyIndexesPath(:,:,:,:,tt);
            else
                Policy=PolicyIndexesPath(:,:,:,tt);
            end
            
            p=PricePath(tt,:);
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 2); % The 2 is for Parallel (use GPU)
            
            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars(ff);
            end
            
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
        end
    end

else
    V_final=reshape(V_final,[N_a,N_z,N_j]);
    AgentDist_initial=reshape(AgentDist_initial,[N_a*N_z,N_j]);
    V=zeros(size(V_final)); %preallocate space
    PricePathNew=zeros(size(PricePath)); PricePathNew(T,:)=PricePath(T,:);
    if N_d>0
        Policy=zeros(2,N_a,N_z,N_j);
    else
        Policy=zeros(N_a,N_z,N_j);
    end
    if N_d>0
        PolicyIndexesPath=zeros(2,N_a,N_z,N_j,T); %Periods 1 to T-1 will be calculated
        PolicyIndexesPath(:,:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
    else
        PolicyIndexesPath=zeros(N_a,N_z,N_j,T); %Periods 1 to T-1 will be calculated
        PolicyIndexesPath(:,:,:,T)=KronPolicyIndexes_FHorz_Case1(Policy_final, n_d, n_a, n_z,N_j);
    end
    
    %First, go from T-1 to 1 calculating the Value function and Optimal
    %policy function at each step. Since we won't need to keep the value
    %functions for anything later we just store the next period one in
    %Vnext, and the current period one to be calculated in V
    Vnext=V_final;
    for ttr=1:T-1 %so t=T-i
        
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePath(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        
        [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_Par1(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')

        if N_d>0
            PolicyIndexesPath(:,:,:,:,T-ttr)=Policy;
        else
            PolicyIndexesPath(:,:,:,T-ttr)=Policy;
        end
        Vnext=V;
    end
    % Free up memory by deleting things no longer needed
    clear V Vnext         
    
    if FnsToEvaluateStruct==0
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        AggVarsPath=zeros(T,length(FnsToEvaluate));
        %Call AgentDist the current periods distn
        AgentDist=AgentDist_initial;
        for tt=1:T%-1
            
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyIndexesPath(:,:,:,:,tt);
            else
                Policy=PolicyIndexesPath(:,:,:,tt);
            end
            
            p=PricePath(tt,:);
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 1); % The 1 is for Parallel (use CPU)
            
            AggVarsPath(tt,:)=AggVars;
            
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
        end
    else % FnsToEvaluateStruct==1
        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution and generate
        %the AggVarsPath.
        for ff=1:length(AggVarNames)
            AggVarsPath.(AggVarNames{ff}).Mean=zeros(T,1);
        end
        %Call AgentDist the current periods distn
        AgentDist=AgentDist_initial;
        for tt=1:T%-1
            
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyIndexesPath(:,:,:,:,tt);
            else
                Policy=PolicyIndexesPath(:,:,:,tt);
            end
            
            p=PricePath(tt,:);
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 1); % The 1 is for Parallel (use CPU)
            
            for ff=1:length(AggVarNames)
                AggVarsPath.(AggVarNames{ff}).Mean(tt)=AggVars(tt,ff);
            end
            
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
        end
    end
end

end
