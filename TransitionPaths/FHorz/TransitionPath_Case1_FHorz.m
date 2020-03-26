function PricePath=TransitionPath_Case1_FHorz(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

%% Check which transpathoptions have been used, set all others to defaults 
if exist('transpathoptions','var')==0
    disp('No transpathoptions given, using defaults')
    %If transpathoptions is not given, just use all the defaults
    transpathoptions.tolerance=10^(-5);
    transpathoptions.parallel=2;
    transpathoptions.exoticpreferences=0;
    transpathoptions.oldpathweight=0.9; % default =0.9
    transpathoptions.weightscheme=1; % default =1
    transpathoptions.Ttheta=1;
    transpathoptions.maxiterations=1000;
    transpathoptions.verbose=0;
    transpathoptions.GEnewprice=2;
    transpathoptions.historyofpricepath=0;
else
    %Check transpathoptions for missing fields, if there are some fill them with the defaults
    if isfield(transpathoptions,'tolerance')==0
        transpathoptions.tolerance=10^(-5);
    end
    if isfield(transpathoptions,'parallel')==0
        transpathoptions.parallel=2;
    end
    if isfield(transpathoptions,'exoticpreferences')==0
        transpathoptions.exoticpreferences=0;
    end
    if isfield(transpathoptions,'oldpathweight')==0
        transpathoptions.oldpathweight=0.9;
    end
    if isfield(transpathoptions,'weightscheme')==0
        transpathoptions.weightscheme=1;
    end
    if isfield(transpathoptions,'Ttheta')==0
        transpathoptions.Ttheta=1;
    end
    if isfield(transpathoptions,'maxiterations')==0
        transpathoptions.maxiterations=1000;
    end
    if isfield(transpathoptions,'verbose')==0
        transpathoptions.verbose=0;
    end
    if isfield(transpathoptions,'GEnewprice')==0
        transpathoptions.GEnewprice=2;
    end
    if isfield(transpathoptions,'historyofpricepath')==0
        transpathoptions.historyofpricepath=0;
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
% else % HAVE NOW IMPLEMENTED THIS
%     if length(DiscountFactorParamNames)~=1
%         disp('WARNING: DiscountFactorParamNames should be of length one')
%         dbstack
%     end
end

if transpathoptions.parallel~=2
    disp('ERROR: Only transpathoptions.parallel==2 is supported by TransitionPath_Case1')
    dbstack
else
    d_grid=gpuArray(d_grid); a_grid=gpuArray(a_grid); z_grid=gpuArray(z_grid); pi_z=gpuArray(pi_z);
    PricePathOld=gpuArray(PricePathOld);
end
% unkronoptions.parallel=2;

if transpathoptions.GEnewprice==1
    PricePath=TransitionPath_Case1_Fhorz_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions);
    return
end

N_d=prod(n_d);
N_z=prod(n_z);
N_a=prod(n_a);
l_p=size(PricePathOld,2);

if transpathoptions.parallel==2
    % Make sure things are on gpu where appropriate.
    if N_d>0
        d_grid=gather(d_grid);
    end
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
end

if transpathoptions.verbose==1
    transpathoptions
end

if transpathoptions.verbose==1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end

%% Set up transition path as minimization of a function (default is to use as objective the weighted sum of squares of the general eqm conditions)
PricePathVec=gather(reshape(PricePathOld,[T*length(PricePathNames),1])); % Has to be vector of fminsearch. Additionally, provides a double check on sizes.

if transpathoptions.GEnewprice==2 % Function minimization
    if n_d(1)==0
        GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_no_d_subfn(pricepathPricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions);
    else
        GeneralEqmConditionsPathFn=@(pricepath) TransitionPath_Case1_FHorz_subfn(pricepath, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, n_d, n_a, n_z, N_j, pi_z, d_grid,a_grid,z_grid, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, AgeWeightsParamNames, FnsToEvaluateParamNames, GeneralEqmEqnParamNames,transpathoptions);
    end
end

% if transpathoptions.GEnewprice2algo==0
[PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathVec);
% else
%     [PricePath,~]=fminsearch(GeneralEqmConditionsPathFn,PricePathOld);
% end

% LOOK INTO USING 'SURROGATE OPTIMIZATION'

PricePath=gpuArray(reshape(PricePath,[T,length(PricePathNames)])); % Switch back to appropriate shape (out of the vector required to use fminsearch)


% PricePathDist=Inf;
% pathcounter=1;
% 
% V_final=reshape(V_final,[N_a,N_z,N_j]);
% AgentDist_initial=reshape(StationaryDist_init,[N_a*N_z,N_j]);
% V=zeros(size(V_final),'gpuArray'); %preallocate space
% PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);
% if N_d>0
%     Policy=zeros(2,N_a,N_z,N_j,'gpuArray');
% else
%     Policy=zeros(N_a,N_z,N_j,'gpuArray');
% end
% if transpathoptions.verbose==1
%     DiscountFactorParamNames
%     ReturnFnParamNames
%     ParamPathNames
%     PricePathNames
% end
% 
% while PricePathDist>transpathoptions.tolerance && pathcounter<transpathoptions.maxiterations
%     if N_d>0
%         PolicyIndexesPath=zeros(2,N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
%     else
%         PolicyIndexesPath=zeros(N_a,N_z,N_j,T-1,'gpuArray'); %Periods 1 to T-1
%     end
%     
%     %First, go from T-1 to 1 calculating the Value function and Optimal
%     %policy function at each step. Since we won't need to keep the value
%     %functions for anything later we just store the next period one in
%     %Vnext, and the current period one to be calculated in V
%     Vnext=V_final;
%     for i=1:T-1 %so t=T-i
%         
%         for kk=1:length(PricePathNames)
%             Parameters.(PricePathNames{kk})=PricePathOld(T-i,kk);
%         end
%         for kk=1:length(ParamPathNames)
%             Parameters.(ParamPathNames{kk})=ParamPath(T-i,kk);
%         end
%         
%         [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
%         % The VKron input is next period value fn, the VKron output is this period.
%         % Policy is kept in the form where it is just a single-value in (d,a')
% 
%         if N_d>0
%             PolicyIndexesPath(:,:,:,:,T-i)=Policy;
%         else
%             PolicyIndexesPath(:,:,:,T-i)=Policy;
%         end
%         Vnext=V;
%     end
%     % Free up space on GPU by deleting things no longer needed
%     clear V Vnext    
%     
%     %Now we have the full PolicyIndexesPath, we go forward in time from 1
%     %to T using the policies to update the agents distribution generating a
%     %new price path
%     %Call AgentDist the current periods distn
%     AgentDist=AgentDist_initial;
%     for i=1:T-1
%                 
%         %Get the current optimal policy
%         if N_d>0
%             Policy=PolicyIndexesPath(:,:,:,:,i);
%         else
%             Policy=PolicyIndexesPath(:,:,:,i);
%         end
%         
%         p=PricePathOld(i,:);
%         
%         for nn=1:length(ParamPathNames)
%             Parameters.(ParamPathNames{nn})=ParamPath(i,nn);
%         end
%         for nn=1:length(PricePathNames)
%             Parameters.(PricePathNames{nn})=PricePathOld(i,nn);
%         end
%         
%         PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz(Policy, n_d, n_a, n_z, N_j,vfoptions);
%         AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 2); % The 2 is for Parallel (use GPU)
%       
%         %An easy way to get the new prices is just to call GeneralEqmConditions_Case1
%         %and then adjust it for the current prices
%             % When using negative powers matlab will often return complex
%             % numbers, even if the solution is actually a real number. I
%             % force converting these to real, albeit at the risk of missing problems
%             % created by actual complex numbers.
%         if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
%             PricePathNew(i,:)=real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames));
%         elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
%             % Remark: following assumes that there is one'GeneralEqmEqnParameter' per 'GeneralEqmEqn'
%             for j=1:length(GeneralEqmEqns)
%                 GEeqn_temp=@(p) sum(real(GeneralEqmConditions_Case1(AggVars,p, GeneralEqmEqns, Parameters,GeneralEqmEqnParamNames)).^2);
%                 PricePathNew(i,j)=fminsearch(GEeqn_temp,p);
%             end
%         end
%         
%         AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
%     end
% %     % Free up space on GPU by deleting things no longer needed
% %     clear AgentDist
%     
%     %See how far apart the price paths are
%     PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
%     %Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
%     
%     if transpathoptions.verbose==1
%         pathcounter
%         disp('Old, New')
%         [PricePathOld,PricePathNew]
%     end
%     
%     %Set price path to be 9/10ths the old path and 1/10th the new path (but
%     %making sure to leave prices in periods 1 & T unchanged).
%     if transpathoptions.weightscheme==1 % Just a constant weighting
%         PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
%     elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
%         % I should precalculate these weighting vectors
% %         PricePathOld(1:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
%         Ttheta=transpathoptions.Ttheta;
%         PricePathOld(1:Ttheta,:)=transpathoptions.oldpathweight*PricePathOld(1:Ttheta,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:Ttheta,:);
%         PricePathOld(Ttheta:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-Ttheta)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(Ttheta:T-1,:)+((exp(linspace(0,log(0.2),T-Ttheta)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(Ttheta:T-1,:);
%     elseif transpathoptions.weightscheme==3 % A gradually opening window.
%         if (pathcounter*3)<T-1
%             PricePathOld(1:(pathcounter*3),:)=transpathoptions.oldpathweight*PricePathOld(1:(pathcounter*3),:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:(pathcounter*3),:);
%         else
%             PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
%         end
%     elseif transpathoptions.weightscheme==4 % Combines weightscheme 2 & 3
%         if (pathcounter*3)<T-1
%             PricePathOld(1:(pathcounter*3),:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),pathcounter*3)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:(pathcounter*3),:)+((exp(linspace(0,log(0.2),pathcounter*3)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:(pathcounter*3),:);
%         else
%             PricePathOld(1:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
%         end
%     end
%     
%     TransPathConvergence=PricePathDist/transpathoptions.tolerance; %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
%     if transpathoptions.verbose==1
%         fprintf('Number of iterations on transition path: %i \n',pathcounter)
%         fprintf('Current distance to convergence: %.2f (convergence when reaches 1) \n',TransPathConvergence) %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
%     end
% %     save ./SavedOutput/TransPathConv.mat TransPathConvergence pathcounter
%     
% %     if pathcounter==1
% %         save ./SavedOutput/FirstTransPath.mat V_final V PolicyIndexesPath PricePathOld PricePathNew
% %     end
% 
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
%     
% 
% end


end