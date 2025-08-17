function PricePath=TransitionPath_InfHorz_shooting_ExpAsset(PricePathOld, PricePathNames, PricePathSizeVec, ParamPath, ParamPathNames, ParamPathSizeVec, T, V_final, AgentDist_initial, n_d1, n_d2, n_a1, n_a2, n_z, pi_z, d1_grid, d2_grid,a1_grid, a2_grid, z_gridvals, ReturnFn, aprimeFn, FnsToEvaluate, GeneralEqmEqns, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, GEeqnNames, vfoptions, simoptions,transpathoptions)

N_a1=prod(n_a1);
N_a2=prod(n_a2);
N_a=N_a1*N_a2;
N_z=prod(n_z);

if n_d1(1)==0
    l_d1=0;
else
    l_d1=length(n_d1);
end
l_d2=length(n_d2); % expasset, so cannot have l_d2=0
l_d=l_d1+l_d2;
l_a1=length(n_a1);
l_a2=length(n_a2);
l_a=l_a1+l_a2;
l_z=length(n_z);

if transpathoptions.verbose==1
    transpathoptions
    ParamPathNames
    PricePathNames
end

%%
if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=cell(1,2*length(PricePathNames));
    for tt=1:length(PricePathNames)
        pathnametitles{tt}={['Old ',PricePathNames{tt}]};
        pathnametitles{tt+length(PricePathNames)}={['New ',PricePathNames{tt}]};
    end
end

%% Check if using _tminus1 and/or _tplus1 variables.
if isstruct(FnsToEvaluate) && isstruct(GeneralEqmEqns)
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk]=inputsFindtplus1tminus1(FnsToEvaluate,GeneralEqmEqns,PricePathNames);
    tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk
else
    tplus1priceNames=[];
    tminus1priceNames=[];
    tminus1AggVarsNames=[];
    tminus1paramNames=[];
    tplus1pricePathkk=[];
end

use_tplus1price=0;
if length(tplus1priceNames)>0
    use_tplus1price=1;
end
use_tminus1price=0;
if length(tminus1priceNames)>0
    use_tminus1price=1;
    for tt=1:length(tminus1priceNames)
        if ~isfield(transpathoptions.initialvalues,tminus1priceNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1priceNames{tt})
            dbstack
            break
        end
    end
end
use_tminus1AggVars=0;
if length(tminus1AggVarsNames)>0
    use_tminus1AggVars=1;
    for tt=1:length(tminus1AggVarsNames)
        if ~isfield(transpathoptions.initialvalues,tminus1AggVarsNames{tt})
            fprintf('ERROR: Using %s as an input (to FnsToEvaluate or GeneralEqmEqns) but it is not in transpathoptions.initialvalues \n',tminus1AggVarsNames{tt})
            dbstack
            break
        end
    end
end
% Note: I used this approach (rather than just creating _tplus1 and _tminus1 for everything) as it will be same computation.

use_tminus1price
use_tminus1AggVars
use_tplus1price

%% Change to FnsToEvaluate as cell so that it is not being recomputed all the time
AggVarNames=fieldnames(FnsToEvaluate);
for ff=1:length(AggVarNames)
    temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
    if length(temp)>(l_d+l_a1+l_a+l_z) % expasset
        FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a1+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
    else
        FnsToEvaluateParamNames(ff).Names={};
    end
    FnsToEvaluateCell{ff}=FnsToEvaluate.(AggVarNames{ff});
end
% Change FnsToEvaluate out of structure form, but want to still create AggVars as a structure
simoptions.outputasstructure=1;
simoptions.AggVarNames=AggVarNames;


%%

PricePathDist=Inf;
pathcounter=0;

V_final=reshape(V_final,[N_a,N_z]);
AgentDist_initial=sparse(gather(reshape(AgentDist_initial,[N_a*N_z,1])));
pi_z_sparse=sparse(pi_z); % Need full pi_z for value fn, and sparse for agent dist

PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);

AggVarsPath=zeros(T-1,length(AggVarNames),'gpuArray'); % Note: does not include the final AggVars, might be good to add them later as a way to make if obvious to user it things are incorrect
GEcondnsPath=zeros(T-1,length(GEeqnNames),'gpuArray');

if l_d1==0
    n_d=n_d2;
    d_gridvals=CreateGridvals(n_d2,d2_grid,1);
else
    n_d=[n_d1,n_d2];
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
end
a_gridvals=CreateGridvals([n_a1,n_a2],[a1_grid; a2_grid],1);
if l_d1==0
    if vfoptions.gridinterplayer==0
        a1prime_gridvals=CreateGridvals(n_a1,a1_grid,1);
        PolicyIndexesPath=zeros(2,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
    elseif vfoptions.gridinterplayer==1
        if isscalar(n_a)
            n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp;
            aprime_grid=interp1(gpuArray(1:1:N_a)',a_grid,gpuArray(linspace(1,N_a,n_aprime))');
            a1prime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
            PolicyIndexesPath=zeros(3,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        else
            a1_grid=a_grid(1:n_a(1));
            n_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
            n_aprime=[n_a1prime,n_a(2:end)];
            a1prime_grid=interp1(gpuArray(1:1:n_a(1))',a1_grid,gpuArray(linspace(1,n_a(1),n_a1prime))');
            aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
            a1prime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
            PolicyIndexesPath=zeros(4,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        end
    end
else % l_d1
    if vfoptions.gridinterplayer==0
        a1prime_gridvals=CreateGridvals(n_a1,a1_grid,1);
        PolicyIndexesPath=zeros(3,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
    elseif vfoptions.gridinterplayer==1
        if isscalar(n_a)
            n_aprime=n_a+(n_a-1)*vfoptions.ngridinterp;
            aprime_grid=interp1(gpuArray(1:1:N_a)',a_grid,gpuArray(linspace(1,N_a,n_aprime))');
            a1prime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
            PolicyIndexesPath=zeros(4,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        else
            a1_grid=a_grid(1:n_a(1));
            n_a1prime=n_a(1)+(n_a(1)-1)*vfoptions.ngridinterp;
            n_aprime=[n_a1prime,n_a(2:end)];
            a1prime_grid=interp1(gpuArray(1:1:n_a(1))',a1_grid,gpuArray(linspace(1,n_a(1),n_a1prime))');
            aprime_grid=[a1prime_grid; a_grid(n_a(1)+1:end)];
            a1prime_gridvals=CreateGridvals(n_aprime,aprime_grid,1);
            PolicyIndexesPath=zeros(5,N_a,N_z,T-1,'gpuArray'); %Periods 1 to T-1
        end
    end
end

if n_d1(1)==0
    d_gridvals=CreateGridvals(n_d2,d2_grid,1);
else
    d_gridvals=CreateGridvals([n_d1,n_d2],[d1_grid; d2_grid],1);
end
a1_gridvals=CreateGridvals(n_a1,a1_grid,1);

% Precompute for AgentDist
if l_d1==0
    ndvars=1;
else
    ndvars=2;
end
d_grid=[d1_grid; d2_grid];
Policy_a2prime=zeros(N_a,N_z,2,'gpuArray'); % the lower grid point
PolicyProbs=zeros(N_a,N_z,2,'gpuArray'); % preallocate
Policy_aprime=zeros(N_a,N_z,2,'gpuArray'); % preallocate
II2=gpuArray([1:1:N_a*N_z; 1:1:N_a*N_z]'); % Index for this period (a,z), note the 2 copies


%%
while PricePathDist>transpathoptions.tolerance && pathcounter<transpathoptions.maxiterations
        
    % First, go from T-1 to 1 calculating the Value function and Optimal policy function at each step. Since we won't need to keep the value
    % functions for anything later we just store the next period one in Vnext, and the current period one to be calculated in V
    Vnext=V_final;
    for ttr=1:T-1 %so tt=T-ttr
        
        for kk=1:length(PricePathNames)
            Parameters.(PricePathNames{kk})=PricePathOld(T-ttr,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
        for kk=1:length(ParamPathNames)
            Parameters.(ParamPathNames{kk})=ParamPath(T-ttr,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
        end
        
        [V, Policy]=ValueFnIter_InfHorz_TPath_SingleStep_ExpAsset(Vnext,n_d1,n_d2,n_a1,n_a2,n_z,d_gridvals,d2_grid, a1_gridvals, a2_grid, z_gridvals, pi_z, ReturnFn, aprimeFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, aprimeFnParamNames, vfoptions);
        % The VKron input is next period value fn, the VKron output is this period.
        % Policy is kept in the form where it is just a single-value in (d,a')
              
        PolicyIndexesPath(:,:,:,T-ttr)=Policy;
        Vnext=V;
    end
    % Free up space on GPU by deleting things no longer needed
    clear V Vnext

    % Now we have the full PolicyIndexesPath, we go forward in time from 1 to T using the policies to update the agents distribution generating a new price path.
    PolicyValuesPath=PolicyInd2Val_InfHorz_TPath(UnKronPolicyIndexes_InfHorz_TransPath_ExpAsset(PolicyIndexesPath,n_d,n_a1,[n_a1,n_a2],n_z,T-1,vfoptions),n_d,[n_a1,n_a2],n_z,T-1,d_gridvals,a1prime_gridvals,simoptions,1);
    % Call AgentDist the current periods distn and AgentDistnext the next periods distn which we must calculate
    AgentDist=AgentDist_initial;
    for tt=1:T-1
        % Get the current optimal policy, and iterate the agent dist
        Policy=PolicyIndexesPath(:,:,:,tt);

        %% Update the parameters
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

        %% Iterate Agent Dist
        if simoptions.gridinterplayer==0

            whichisdforexpasset=length(n_d);  % is just saying which is the decision variable that influences the experience asset (it is the 'last' decision variable)
            aprimeFnParamsVec=CreateVectorFromParams(Parameters, aprimeFnParamNames);
            [a2primeIndexes, a2primeProbs]=CreateaprimePolicyExperienceAsset_Case1(Policy,simoptions.aprimeFn, whichisdforexpasset, n_d, n_a1,n_a2, N_z, d_grid, a2_grid, aprimeFnParamsVec);
            % Note: aprimeIndexes and aprimeProbs are both [N_a,N_z]
            % Note: aprimeIndexes is always the 'lower' point (the upper points are just aprimeIndexes+1), and the aprimeProbs are the probability of this lower point (prob of upper point is just 1 minus this).
            Policy_a2prime(:,:,1)=a2primeIndexes; % lower grid point
            Policy_a2prime(:,:,2)=a2primeIndexes+1; % upper grid point
            PolicyProbs(:,:,1)=a2primeProbs; % probability of lower grid point
            PolicyProbs(:,:,2)=1-a2primeProbs; % probability of upper grid point

            if l_a1==0 % just experienceasset
                Policy_aprime=Policy_a2prime;
            elseif l_a1==1 % one other asset, then experience asset
                Policy_aprime(:,:,1)=reshape(Policy(ndvars+1,:,:),[N_a,N_z,1])+n_a1*(Policy_a2prime(:,:,1)-1);
                Policy_aprime(:,:,2)=reshape(Policy(ndvars+1,:,:),[N_a,N_z,1])+n_a1*Policy_a2prime(:,:,1); % Note: upper grid point minus 1 is anyway just lower grid point
            end
            Policy_aprimez=reshape(Policy_aprime+N_a*(0:1:N_z-1),[N_a*N_z,2]);
            AgentDistnext=StationaryDist_InfHorz_TPath_SingleStep_TwoProbs(AgentDist,Policy_aprimez,II2,PolicyProbs,N_a,N_z,pi_z_sparse);
        elseif simoptions.gridinterplayer==1
            error('not done yet')
            % if isscalar(n_a)
            %     Policy_aprime(:,1)=reshape(Policy(2,:,:),[N_a*N_z,1]); % lower grid point
            %     Policy_aprime(:,2)=Policy_aprime(:,1)+1; % upper grid point
            %     PolicyProbs(:,2)=reshape(Policy(3,:,:),[N_a*N_z,1]); % L2 index
            % else % length(n_a)>1
            %     Policy_aprime(:,1)=reshape(Policy(2,:,:),[N_a*N_z,1]); % lower grid point
            %     Policy_aprime=Policy_aprime+n_a(1)*(reshape(Policy(3,:,:),[N_a*N_z,1])-1); % a2
            %     Policy_aprime(:,2)=Policy_aprime(:,1)+1; % upper grid point
            %     PolicyProbs(:,2)=reshape(Policy(4,:,:),[N_a*N_z,1]); % L2 index
            % end
            % PolicyProbs(:,2)=(PolicyProbs(:,2)-1)/(1+simoptions.ngridinterp); % probability of upper grid point
            % PolicyProbs(:,1)=1-PolicyProbs(:,2); % probability of lower grid point
            Policy_aprimez=reshape(Policy_aprime+N_a*(0:1:N_z-1),[N_a*N_z,1]);
            AgentDistnext=StationaryDist_InfHorz_TPath_SingleStep_TwoProbs(AgentDist,Policy_aprimez,II2,PolicyProbs,N_a,N_z,pi_z_sparse);
        end
        
        %% AggVars and General Eqm
        AggVars=EvalFnOnAgentDist_InfHorz_TPath_SingleStep_AggVars(full(AgentDist), PolicyValuesPath(:,:,:,tt), FnsToEvaluateCell, Parameters, FnsToEvaluateParamNames, AggVarNames, [n_a1,n_a2], n_z, a_gridvals, z_gridvals,1);
        
        % [AggVars.L.Mean, AggVars.K.Mean,tt,full(sum(AgentDist))]
        % sum(a_grid.*sum(reshape(AgentDist,[N_a,N_z]),2))

        % When using negative powers matlab will often return complex numbers, even if the solution is actually a real number. I
        % force converting these to real, albeit at the risk of missing problems created by actual complex numbers.
        if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
                for ii=1:length(AggVarNames)
                    Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
                end
                PricePathNew(tt,:)=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
        elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
            % Remark: following assumes that there is one'GeneralEqmEqnParameter' per 'GeneralEqmEqn'
            for j=1:length(GeneralEqmEqns)
                for ii=1:length(AggVarNames)
                    Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
                end
                GEeqn_temp=@(GEprices) sum(real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2)).^2);
                PricePathNew(tt,j)=fminsearch(GEeqn_temp,GEprices);
            end
        % Note there is no GEnewprice==2, it uses a completely different code
        elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
            for ii=1:length(AggVarNames)
                Parameters.(AggVarNames{ii})=AggVars.(AggVarNames{ii}).Mean;
            end
            p_i=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
            GEcondnsPath(tt,:)=p_i; % Sometimes, want to keep the GE conditions to plot them
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
    
    %% Check convergence, and give feedback on progress

    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1
        fprintf('Number of iteration on the path: %i \n',pathcounter)
        
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
            subplot(nrows,ncolumns,pp); plot(GEcondnsPath(:,pp))
            title(GEeqnNames{pp})
        end
    end
    
    %Set price path to be 9/10ths the old path and 1/10th the new path (but making sure to leave prices in periods 1 & T unchanged).
    if transpathoptions.weightscheme==0
        PricePathOld=PricePathNew; % The update weights are already in GEnewprice setup
    elseif transpathoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:T-1,:);
    elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        % I should precalculate these weighting vectors
%         PricePathOld(1:T-1,:)=((transpathoptions.oldpathweight+(1-exp(linspace(0,log(0.2),T-1)))*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathOld(1:T-1,:)+((exp(linspace(0,log(0.2),T-1)).*(1-transpathoptions.oldpathweight))'*ones(1,l_p)).*PricePathNew(1:T-1,:);
        Ttheta=transpathoptions.Ttheta;
        PricePathOld(1:Ttheta,:)=transpathoptions.oldpathweight*PricePathOld(1:Ttheta,:)+(1-transpathoptions.oldpathweight)*PricePathNew(1:Ttheta,:);
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


for tt=1:length(PricePathNames)
    PricePath.(PricePathNames{tt})=PricePathOld(:,tt);
end


end
