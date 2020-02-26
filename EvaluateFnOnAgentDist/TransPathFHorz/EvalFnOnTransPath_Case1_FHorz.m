function AggVarsPath=EvalFnOnTransPath_Case1_FHorz(FnsToEvaluate, FnsToEvaluateParamNames,PricePath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, Parameters, n_d, n_a, n_z, N_j, pi_z, d_grid, a_grid,z_grid, DiscountFactorParamNames, ReturnFn, ReturnFnParamNames,AgeWeightsParamNames, transpathoptions)
%AggVarsPath is T-1 periods long (periods 0 (before the reforms are announced) & T are the initial and final values; they are not created by this command and instead can be used to provide double-checks of the output (the T-1 and the final should be identical if convergence has occoured).
AggVarsPath=nan(T-1,length(FnsToEvaluate),'gpuArray');

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
    transpathoptions.parallel=2;
    transpathoptions.exoticpreferences=0;
    transpathoptions.verbose=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences=0;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
end

%% Check which vfoptions have been used, set all others to defaults 
if isfield(transpathoptions,'vfoptions')==1
    vfoptions=transpathoptions.vfoptions;
end

if exist('vfoptions','var')==0
    disp('No vfoptions given, using defaults')
    %If vfoptions is not given, just use all the defaults
%     vfoptions.exoticpreferences=0;
    vfoptions.parallel=2;
    vfoptions.returnmatrix=2;
    vfoptions.verbose=0;
    vfoptions.lowmemory=0;
    vfoptions.polindorval=1;
    vfoptions.policy_forceintegertype=0;
else
    %Check vfoptions for missing fields, if there are some fill them with the defaults
    if isfield(vfoptions,'parallel')==0
        vfoptions.parallel=2;
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
    simoptions.parallel=2;
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
        simoptions.parallel=2;
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

%%
if transpathoptions.exoticpreferences~=0
    disp('ERROR: Only transpathoptions.exoticpreferences==0 is supported by TransitionPath_Case1')
    dbstack
else
    if length(DiscountFactorParamNames)~=1
        disp('WARNING: DiscountFactorParamNames should be of length one')
        dbstack
    end
end

if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePath=gpuArray(PricePath);
end
unkronoptions.parallel=2;

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePath,2);

if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    if N_d>0
        d_grid=gather(d_grid);
    end
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end


% if N_d==0
%     PricePathNew=TransitionPath_Case1_no_d(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_a, n_z, N_j, pi_z, a_grid,z_grid, ReturnFn, SSvaluesFn, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, SSvalueParamNames, GeneralEqmEqnParamNames,transpathoptions);
%     return
% end

if transpathoptions.verbose==1
    transpathoptions
end

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z,N_j]);
StationaryDist_initial=reshape(StationaryDist_init,[N_a*N_z,N_j]);
V=zeros(size(V_final),'gpuArray');
if N_d>0
    Policy=zeros(2,N_a,N_z,N_j,'gpuArray');
else
    Policy=zeros(N_a,N_z,N_j,'gpuArray');
end
if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%%

if N_d>0
    PolicyIndexesPath=zeros(2,N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
else
    PolicyIndexesPath=zeros(N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
end

%First, go from T-1 to 1 calculating the Value function and Optimal
%policy function at each step. Since we won't need to keep the value
%functions for anything later we just store the next period one in
%Vnext, and the current period one to be calculated in V
Vnext=V_final;
for i=1:T-1 %so t=T-i
    
    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(T-i,kk);
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
StationaryDist=StationaryDist_initial;
for i=1:T-1
    
    %Get the current optimal policy
    if N_d>0
        Policy=PolicyIndexesPath(:,:,:,:,i);
    else
        Policy=PolicyIndexesPath(:,:,:,i);
    end
        
    for nn=1:length(ParamPathNames)
        Parameters.(ParamPathNames{nn})=ParamPath(i,nn);
    end
    for nn=1:length(PricePathNames)
        Parameters.(PricePathNames{nn})=PricePath(i,nn);
    end
    
    PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
    AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(StationaryDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 2); % The 2 is for Parallel (use GPU)
    
    AggVarsPath(i,:)=AggVars;
    
    StationaryDist=StationaryDist_FHorz_Case1_TPath_SingleStep(StationaryDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
end

end