function AggVarsPath=TransitionPath_InfHorz_PType_singlepath(PricePathOld, ParamPath, PricePathNames, ParamPathNames, T, V_final, AgentDist_initial, ...
    l_d,N_d,n_d,N_a,n_a,N_z,n_z,d_grid,a_grid,d_gridvals,aprime_gridvals,a_gridvals,z_grid,pi_z,ReturnFn, ...
    FnsToEvaluate, ...
    Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, ...
    use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, ...
    transpathoptions, vfoptions, simoptions)
% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

% For this agent type, first go back through the value & policy fns.
% Then forwards through agent dist and agg vars.

l_a=length(n_a);
PolicyIndexesPath=zeros(l_d+l_a,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1

% This and much of the rest of this code borrowed from TransitionPath_InfHorz_shooting.m
if simoptions.gridinterplayer==0
    II1=(1:1:N_a*N_z); % Index for this period (a,z)
    IIones=ones(N_a*N_z,1); % Next period 'probabilities'
elseif simoptions.gridinterplayer==1
    PolicyProbsPath=zeros(N_a*N_z,2,T-1,'gpuArray'); % preallocate
    II2=([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies
end

if size(V_final)~=[N_a,N_z]
    error("V_final wrong shape")
    V_final=reshape(V_final,[N_a,N_z]);
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for ttr=1:T-1 %so t=T-i
    % The following digs deeper into PricePathOld and ParamPath in
    % FHorz case--check it
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,kk);
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,kk);
    end
    
    [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
    % The VKron input is next period value fn, the VKron output is this period.
    % Policy is kept in the form where it is just a single-value in (d,a')

    PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    Vnext=V;
    
end
% Free up space on GPU by deleting things no longer needed
clear V Vnext

%% Modify PolicyIndexesPath into forms needed for forward iteration
% Create version of PolicyIndexesPath in form we want for the agent distribution iteration
% Creates PolicyaprimezPath, and when using grid interpolation layer also PolicyProbsPath 
if isscalar(n_a)
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:),[N_a*N_z,T-1]); % aprime index
elseif length(n_a)==2
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1),[N_a*N_z,T-1]);
elseif length(n_a)==3
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1),[N_a*N_z,T-1]);
elseif length(n_a)==4
    PolicyaprimePath=reshape(PolicyIndexesPath(l_d+1,:,:,:)+n_a(1)*(PolicyIndexesPath(l_d+2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(l_d+3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(l_d+4,:,:,:)-1),[N_a*N_z,T-1]);
end
PolicyaprimezPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);
if simoptions.gridinterplayer==1
    PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T-1]); % reinterpret this as lower grid index
    PolicyaprimezPath=repelem(PolicyaprimezPath,1,2,1); % create copy that will be the upper grid index
    PolicyaprimezPath(:,2,:)=PolicyaprimezPath(:,2,:)+1; % upper grid index
    PolicyProbsPath(:,2,:)=reshape(PolicyIndexesPath(l_d+l_aprime+1,:,:),[N_a*N_z,1,T-1]); % L2 index
    PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
    PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
end
% Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,n_d,n_a,n_z,T-1,d_grid,a_grid,vfoptions,1);
PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T-1]),[2,3,1,4]); %[N_a,N_z,l_d+l_a,T-1]

%Now we have the full PolicyIndexesPath, we go forward in time from 1
%to T using the policies to update the agents distribution generating a
%new price path
%Call AgentDist the current periods distn

AgentDist=sparse(gather(reshape(AgentDist_initial,[N_a*N_z,1])));
AggVarsPath=zeros(length(FnsToEvaluate),T-1);
pi_z_sparse=sparse(gather(pi_z)); % Need full pi_z for value fn, and sparse for agent dist

for tt=1:T-1
    %% Setup the Parameters for period tt

    % Get t-1 PricePath and ParamPath before we update them
    if use_tminus1price==1
        for pp=1:length(tminus1priceNames)
            if tt>1
                Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
            else
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
    end
    if use_tminus1params==1
        for pp=1:length(tminus1paramNames)
            if tt>1
                Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
            else
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
    end
    % Get t-1 AggVars before we update them
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                % The AggVars have not yet been updated, so they still contain previous period values
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    % Update current PricePath and ParamPath
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    % Get t+1 PricePath
    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
        end
    end
    
    %% Get the current optimal policy, and iterate the agent dist
    if simoptions.gridinterplayer==0
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep(AgentDist,PolicyaprimezPath(:,tt),II1,IIones,N_a,N_z,pi_z_sparse);
    elseif simoptions.gridinterplayer==1
        AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a,N_z,pi_z_sparse);
    end

    %% AggVars
    AggVars_Means=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(gpuArray(full(AgentDist)), PolicyValuesPath(:,:,:,tt), FnsToEvaluate, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, n_z, a_gridvals, z_grid,1);
    AggVars=zeros(length(AggVars_Means),1);
    if length(fieldnames(AggVars_Means))~=length(AggVarNames)
        error(["AggVar length disparity:";"---------";AggVarNames;"--- vs ---";fieldnames(AggVars_Means)]);
    end
    for ii=1:length(AggVarNames)
        AggVars(ii)=AggVars_Means.(AggVarNames{ii}).Mean;
        Parameters.(AggVarNames{ii})=AggVars(ii);
    end

    % Do nothing with IntermediateEqns and GeneralEqmEqns as they are outside PType scope (they are where all the PTypes meet).

    AgentDist=AgentDistnext;

    AggVarsPath(:,tt)=AggVars;
end


end
