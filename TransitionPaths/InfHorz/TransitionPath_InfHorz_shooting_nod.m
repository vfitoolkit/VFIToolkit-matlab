function [PricePathOld,GEcondnPath]=TransitionPath_InfHorz_shooting_nod(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec,  T, V_final, AgentDist_initial, n_a, n_z, pi_z, a_grid,z_gridvals, ReturnFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, GEeqnNames, vfoptions, simoptions,transpathoptions)


N_z=prod(n_z);
N_a=prod(n_a);

l_a=length(n_a);
l_z=length(n_z);
l_aprime=length(n_a);
l_daprime=l_a;  % This is the no_d code

%%
if transpathoptions.verbose>=1
    % Set up some things to be used later
    pathnametitles=cell(1,2*length(PricePathNames));
    for tt=1:length(PricePathNames)
        pathnametitles{tt}={['Old ',PricePathNames{tt}]};
        pathnametitles{tt+length(PricePathNames)}={['New ',PricePathNames{tt}]};
    end
end

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames);
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1AggVarsNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if ~isempty(tplus1priceNames)
    use_tplus1price=1;
end
use_tminus1price=0;
if ~isempty(tminus1priceNames)
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
        end
    end
end
use_tminus1AggVars=0;
if ~isempty(tminus1AggVarsNames)
    use_tminus1AggVars=1;
    for tt=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{tt})
            dbstack
            error('Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{tt})
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

if transpathoptions.verbose>1
    tplus1priceNames
    tminus1priceNames
    tminus1AggVarsNames
    tplus1pricePathkk
    use_tminus1price
    use_tminus1AggVars
end

%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_daprime+l_a+l_z)
        FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;
simoptions.AggVarNames=AggVarNames;


%%
l_p=size(PricePathOld,2);

PricePathDist=Inf;
pathcounter=1;

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=sparse(gather(reshape(AgentDist_initial,[N_a*N_z,1])));
pi_z_sparse=sparse(gather(pi_z)); % Need full pi_z for value fn, and sparse for agent dist

AggVarsPath=zeros(T-1,length(FnsToEvaluate),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnPath=zeros(T-1,length(GEeqnNames),'gpuArray');

PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);

a_gridvals=CreateGridvals(n_a,a_grid,1);
if vfoptions.gridinterplayer==0
    aprime_gridvals=CreateGridvals(n_a,a_grid,1);
    PolicyIndexesPath=zeros(l_aprime,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
elseif vfoptions.gridinterplayer==1
    if isscalar(n_a)
        aprime_grid=interp1(gpuArray(1:1:N_a)',a_grid,gpuArray(linspace(1,N_a,N_a+(N_a-1)*vfoptions.ngridinterp))');
        aprime_gridvals=CreateGridvals(n_a,aprime_grid,1);
    else
        a1_grid=a_grid(1:n_a(1));
        n_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
        a1prime_grid=interp1(gpuArray(1:1:n_a(1))',a1_grid,gpuArray(linspace(1,n_a(1),n_a1prime))');
        aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
        n_aprime=[n_a1prime,n_a(2:end)];
        aprime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
    end
    PolicyIndexesPath=zeros(l_aprime+1,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
end
if simoptions.gridinterplayer==0
    II1=(1:1:N_a*N_z); % Index for this period (a,z)
    IIones=ones(N_a*N_z,1); % Next period 'probabilities'
elseif simoptions.gridinterplayer==1
    PolicyProbs=zeros(N_a*N_z,2,'gpuArray'); % preallocate
    II2=([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies
end

while PricePathDist>transpathoptions.tolerance && pathcounter<transpathoptions.maxiter
    
    %% Iterate backwards from T-1 to 1 calculating the Value function and Optimal policy function at each step. 
    % Since we won't need to keep the value functions for anything later we just store the next period one in Vnext, and the current period one to be calculated in V
    V=V_final;
    for ttr=1:T-1 %so tt=T-ttr

        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep(V,0,n_a,n_z,[], a_grid, z_gridvals, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')
        
        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
    end
    
    %% Modify PolicyIndexesPath into forms needed for forward iteration
    % Create version of PolicyIndexesPath in form we want for the agent distribution iteration
    % Creates PolicyaprimezPath, and when using grid interpolation layer also PolicyProbsPath 
    if isscalar(n_a)
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,:,:),[N_a*N_z,T-1]); % aprime index
    elseif length(n_a)==2
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,:,:)+n_a(1)*(PolicyIndexesPath(2,:,:,:)-1),[N_a*N_z,T-1]);
    elseif length(n_a)==3
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,:,:)+n_a(1)*(PolicyIndexesPath(2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(3,:,:,:)-1),[N_a*N_z,T-1]);
    elseif length(n_a)==4
        PolicyaprimePath=reshape(PolicyIndexesPath(1,:,:,:)+n_a(1)*(PolicyIndexesPath(2,:,:,:)-1)+n_a(1)*n_a(2)*(PolicyIndexesPath(3,:,:,:)-1)+n_a(1)*n_a(2)*n_a(3)*(PolicyIndexesPath(4,:,:,:)-1),[N_a*N_z,T-1]);
    end
    PolicyaprimezPath=PolicyaprimePath+repelem(N_a*gpuArray(0:1:N_z-1)',N_a,1);
    if simoptions.gridinterplayer==1
        PolicyaprimezPath=reshape(PolicyaprimezPath,[N_a*N_z,1,T-1]); % reinterpret this as lower grid index
        PolicyaprimezPath=repelem(PolicyaprimezPath,1,2,1); % create copy that will be the upper grid index
        PolicyaprimezPath(:,2,:)=PolicyaprimezPath(:,2,:)+1; % upper grid index
        PolicyProbsPath(:,2,:)=reshape(PolicyIndexesPath(l_aprime+1,:,:),[N_a*N_z,1,T-1]); % L2 index
        PolicyProbsPath(:,2,:)=(PolicyProbsPath(:,2,:)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
        PolicyProbsPath(:,1,:)=1-PolicyProbsPath(:,2,:); % probability of lower grid point
    end
    % Create PolicyValuesPath from PolicyIndexesPath for use in calculating model stats
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(PolicyIndexesPath,0,n_a,n_z,T-1,[],a_grid,vfoptions,1);
    PolicyValuesPath=permute(reshape(PolicyValuesPath,[size(PolicyValuesPath,1),N_a,N_z,T-1]),[2,3,1,4]); %[N_a,N_z,l_a,T-1]

    %% Iterate forward over t: iterate agent dist, calculate aggvars, evaluate general eqm
    % Call AgentDist the current periods distn and AgentDistnext the next periods distn which we must calculate
    AgentDist=AgentDist_initial;
    for tt=1:T-1
        %% Setup the Parameters for period tt
        GEprices=PricePathOld(tt,:);
        
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                if tt>1
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                else
                    Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                end
            end
        end
        if use_tplus1price==1
            for pp=1:length(tplus1priceNames)
                kk=tplus1pricePathkk(pp);
                Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
            end
        end
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

        %% Get the current optimal policy, and iterate the agent dist
        if simoptions.gridinterplayer==0
            AgentDistnext=AgentDist_InfHorz_TPath_SingleStep(AgentDist,PolicyaprimezPath(:,tt),II1,IIones,N_a,N_z,pi_z_sparse);
        elseif simoptions.gridinterplayer==1
            AgentDistnext=AgentDist_InfHorz_TPath_SingleStep_nProbs(AgentDist,PolicyaprimezPath(:,:,tt),II2,PolicyProbsPath(:,:,tt),N_a,N_z,pi_z_sparse);
        end

        %% AggVars
        AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(gpuArray(full(AgentDist)), PolicyValuesPath(:,:,:,tt), FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, n_a, n_z, a_gridvals, z_gridvals,1);
        for ii=1:length(AggVarNames)
            Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
        end
        
        %% Intermediate Eqns
        if transpathoptions.useintermediateEqns==1
            % Note: intermediateEqns just take in things from the Parameters structure, as do GeneralEqmEqns (AggVars get put into structure), hence just use the GeneralEqmConditions_Case1_v3g().
            intEqnnames=fieldnames(transpathoptions.intermediateEqns);
            intermediateEqnsVec=zeros(1,length(intEqnnames));
            % Do the intermediateEqns, in order
            for gg=1:length(intEqnnames)
                intermediateEqnsVec(gg)=real(GeneralEqmConditions_Case1_v3g(transpathoptions.intermediateEqnsCell{gg}, transpathoptions.intermediateEqnParamNames(gg).Names, Parameters));
                Parameters.(intEqnnames{gg})=intermediateEqnsVec(gg);
            end
        end
        
        %% General Eqm Eqns
        % When using negative powers matlab will often return complex numbers, even if the solution is actually a real number. I
        % force converting these to real, albeit at the risk of missing problems created by actual complex numbers.
        if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
                PricePathNew(tt,:)=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
        elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
            % Remark: following assumes that there is one'GeneralEqmEqnParameter' per 'GeneralEqmEqn'
            for j=1:length(GeneralEqmEqns)
                GEeqn_temp=@(GEprices) sum(real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2)).^2);
                PricePathNew(tt,j)=fminsearch(GEeqn_temp,GEprices);
            end
        % Note there is no GEnewprice==2, it uses a completely different code
        elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
            p_i=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
            GEcondnPath(tt,:)=p_i; % Sometimes, want to keep the GE conditions to plot them
            p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;
            PricePathNew(tt,:)=(PricePathOld(tt,:).*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
        end

        % Sometimes, want to keep the AggVars to plot them
        if transpathoptions.graphaggvarspath==1
            for ii=1:length(AggVarNames)
                AggVarsPath(tt,ii)=AggVars.(AggVarNames{ii}).Mean;
            end
        end

        AgentDist=AgentDistnext;
    end
    % Free up space on GPU by deleting things no longer needed
    clear AgentDistnext AgentDist
        
    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=1 & t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1
        fprintf('Number of iteration on the path: %i \n',pathcounter)
        
        % Would be nice to have a way to get the iteration count without having the whole printout of path values (I think that would be useful?)
        pathnametitles{:}
        [PricePathOld,PricePathNew]
    end
    
    if transpathoptions.graphpricepath==1
        if length(PricePathNames)>12
            ncolumns=4;
        elseif length(PricePathNames)>6
            ncolumns=3;
        else
            ncolumns=2;
        end
        nrows=ceil(length(PricePathNames)/ncolumns);
        fig1=figure(1);
        for pp=1:length(PricePathNames)
            subplot(nrows,ncolumns,pp); plot(PricePathOld(:,pp))
            title(PricePathNames{pp})
        end
    end
    if transpathoptions.graphaggvarspath==1
        % Do an additional graph, this one of the AggVars
        if length(AggVarNames)>12
            ncolumns=4;
        elseif length(AggVarNames)>6
            ncolumns=3;
        else
            ncolumns=2;
        end
        nrows=ceil(length(AggVarNames)/ncolumns);
        fig2=figure(2);
        for pp=1:length(AggVarNames)
            subplot(nrows,ncolumns,pp); plot(AggVarsPath(:,pp))
            title(AggVarNames{pp})
        end
    end
    if transpathoptions.graphGEcondns==1
        % Assumes transpathoptions.GEnewprice==3
        % Do an additional graph, this one of the general eqm conditions
        if length(GEeqnNames)>12
            ncolumns=4;
        elseif length(GEeqnNames)>6
            ncolumns=3;
        else
            ncolumns=2;
        end
        nrows=ceil(length(GEeqnNames)/ncolumns);
        fig3=figure(3);
        for pp=1:length(GEeqnNames)
            subplot(nrows,ncolumns,pp); plot(GEcondnPath(:,pp))
            title(GEeqnNames{pp})
        end
    end

    
    
    %Set price path to be 9/10ths the old path and 1/10th the new path (but making sure to leave prices in periods 1 & T unchanged).
    if transpathoptions.weightscheme==0
        PricePathOld=PricePathNew; % The update weights are already in GEnewprice setup
    elseif transpathoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
    elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        % I should precalculate these weighting vectors
        Ttheta=transpathoptions.Ttheta;
        PricePathOld(1:Ttheta,:)=transpathoptions.oldpathweight*PricePathOld(1:Ttheta)+(1-transpathoptions.oldpathweight)*PricePathNew(1:Ttheta,:);
        PricePathOld(Ttheta:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-Ttheta)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(Ttheta:T-1,:)+((exp(linspace(0,log(0.2),T-Ttheta)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(Ttheta:T-1,:);
    elseif transpathoptions.weightscheme==3 % A gradually opening window.
        if (pathcounter*3)<T-1
            PricePathOld(1:(pathcounter*3),:)=transpathoptions.oldpathweight*PricePathOld(1:(pathcounter*3),:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:(pathcounter*3),:);
        else
            PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
        end
    elseif transpathoptions.weightscheme==4 % Combines weightscheme 2 & 3
        if (pathcounter*3)<T-1
            PricePathOld(1:(pathcounter*3),:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),pathcounter*3)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:(pathcounter*3),:)+((exp(linspace(0,log(0.2),pathcounter*3)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:(pathcounter*3),:);
        else
            PricePathOld(1:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
        end
    end
    
    TransPathConvergence=PricePathDist/transpathoptions.tolerance; %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path: %i \n',pathcounter)
        fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', PricePathDist)
        fprintf('Current distance to convergence: %.2f (convergence when reaches 1) \n',TransPathConvergence) %So when this gets to 1 we have convergence (uncomment when you want to see how the convergence isgoing)
    end
    
    if transpathoptions.historyofpricepath==1
        % Store the whole history of the price path and save it every ten iterations
        PricePathHistory{pathcounter,1}=PricePathDist;
        PricePathHistory{pathcounter,2}=PricePathOld;        
        if rem(pathcounter,10)==1
            save ./SavedOutput/TransPath_Internal.mat PricePathHistory
        end
    end
    
    pathcounter=pathcounter+1;
end


end
