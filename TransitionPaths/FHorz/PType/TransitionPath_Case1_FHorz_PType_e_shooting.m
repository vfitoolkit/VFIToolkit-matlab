function PricePathOld=TransitionPath_Case1_FHorz_PType_e_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, StationaryDist_init, FullFnsToEvaluate, GeneralEqmEqns, transpathoptions, PTypeStructure)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes) 

% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

l_p=size(PricePathOld,2);

if transpathoptions.verbose==1
    transpathoptions
end
if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=cell(1,2*length(PricePathNames));
    for tt=1:length(PricePathNames)
        pathnametitles{tt}={['Old ',PricePathNames{tt}]};
        pathnametitles{tt+length(PricePathNames)}={['New ',PricePathNames{tt}]};
    end
end

PricePathDist=Inf;
pathcounter=1;

AgentDist_initial.ptweights=StationaryDist_init.ptweights;
for ii=1:PTypeStructure.N_i
    iistr=PTypeStructure.Names_i{ii};

    N_a=prod(PTypeStructure.(iistr).n_a);
    N_z=prod(PTypeStructure.(iistr).n_z);
    N_e=prod(PTypeStructure.(iistr).vfoptions.n_e);
    N_j=PTypeStructure.(iistr).N_j;
    V_final.(iistr)=reshape(V_final.(iistr),[N_a,N_z,N_e,N_j]);
    AgentDist_initial.(iistr)=reshape(StationaryDist_init.(iistr),[N_a*N_z*N_e,N_j]);
end
PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);

if transpathoptions.verbose==1
    ParamPathNames
    PricePathNames
end

while PricePathDist>transpathoptions.tolerance && pathcounter<transpathoptions.maxiterations
    
    % For each agent type, first go back through the value & policy fns.
    % Then forwards through agent dist and agg vars.
    AggVarsFullPath=zeros(PTypeStructure.numFnsToEvaluate,T-1,PTypeStructure.N_i); % Does not include period T
    for ii=1:PTypeStructure.N_i
        iistr=PTypeStructure.Names_i{ii};
        
        % Grab everything relevant out of PTypeStructure
        n_d=PTypeStructure.(iistr).n_d; N_d=prod(n_d);
        n_a=PTypeStructure.(iistr).n_a; N_a=prod(n_a);
        n_z=PTypeStructure.(iistr).n_z; N_z=prod(n_z);
        n_e=PTypeStructure.(iistr).vfoptions.n_e; N_e=prod(n_e);
        N_j=PTypeStructure.(iistr).N_j;
        d_grid=PTypeStructure.(iistr).d_grid;
        a_grid=PTypeStructure.(iistr).a_grid;
        z_grid=PTypeStructure.(iistr).z_grid;
        % e_grid=PTypeStructure.(iistr).vfoptions.e_grid;
        pi_z=PTypeStructure.(iistr).pi_z;
        % pi_e=PTypeStructure.(iistr).vfoptions.pi_e;
        ReturnFn=PTypeStructure.(iistr).ReturnFn;
        Parameters=PTypeStructure.(iistr).Parameters;
        DiscountFactorParamNames=PTypeStructure.(iistr).DiscountFactorParamNames;
        ReturnFnParamNames=PTypeStructure.(iistr).ReturnFnParamNames;
        AgeWeightsParamNames=PTypeStructure.(iistr).AgeWeightsParamNames;
        vfoptions=PTypeStructure.(iistr).vfoptions;
        simoptions=PTypeStructure.(iistr).simoptions;
        FnsToEvaluate=PTypeStructure.(iistr).FnsToEvaluate;
        FnsToEvaluateParamNames=PTypeStructure.(iistr).FnsToEvaluateParamNames;
        
        if N_d>0
            PolicyIndexesPath=zeros(2,N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        else
            PolicyIndexesPath=zeros(N_a,N_z,N_e,N_j,T-1,'gpuArray'); %Periods 1 to T-1
        end
        
        if transpathoptions.verbose==1
            fprintf('Transition path: Backward iteration to compute policy of ptype %i \n',ii)
        end

        %First, go from T-1 to 1 calculating the Value function and Optimal
        %policy function at each step. Since we won't need to keep the value
        %functions for anything later we just store the next period one in
        %Vnext, and the current period one to be calculated in V
        Vnext=V_final.(iistr);
        for tt=1:T-1 % so T-tt goes T-1 to 1
            
            
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(T-tt,kk);
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(T-tt,kk);
            end
            
            if transpathoptions.fastOLG==0
                [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            else
                error('Have not yet implemented fastOLG with e variables. Email me if you want/need')
                % [V, Policy]=ValueFnIter_Case1_FHorz_TPath_SingleStep_fastOLG(Vnext,n_d,n_a,n_z,N_j,d_grid, a_grid, z_grid, pi_z, ReturnFn, Parameters, DiscountFactorParamNames, ReturnFnParamNames, vfoptions);
            end
            % The VKron input is next period value fn, the VKron output is this period.
            % Policy is kept in the form where it is just a single-value in (d,a')
            
            if N_d>0
                PolicyIndexesPath(:,:,:,:,:,T-tt)=Policy;
            else
                PolicyIndexesPath(:,:,:,:,T-tt)=Policy;
            end
            Vnext=V;
            
        end
        % Free up space on GPU by deleting things no longer needed
        clear V Vnext
        
        if transpathoptions.verbose==1
            fprintf('Transition path: Forward iteration to compute agent dist and agg var of ptype %i \n',ii)
        end

        %Now we have the full PolicyIndexesPath, we go forward in time from 1
        %to T using the policies to update the agents distribution generating a
        %new price path
        %Call AgentDist the current periods distn
        AgentDist=AgentDist_initial.(iistr);
        AggVarsPath=zeros(length(FnsToEvaluate),T-1);
        for tt=1:T-1
            
            %Get the current optimal policy
            if N_d>0
                Policy=PolicyIndexesPath(:,:,:,:,:,tt);
            else
                Policy=PolicyIndexesPath(:,:,:,:,tt);
            end
            
            GEprices=PricePathOld(tt,:);
            
            for nn=1:length(ParamPathNames)
                Parameters.(ParamPathNames{nn})=ParamPath(tt,nn);
            end
            for nn=1:length(PricePathNames)
                Parameters.(PricePathNames{nn})=PricePathOld(tt,nn);
            end
            
            PolicyUnKron=UnKronPolicyIndexes_Case1_FHorz_e(Policy, n_d, n_a, n_z,n_e,N_j,vfoptions);

            AggVars=EvalFnOnAgentDist_AggVars_FHorz_Case1(AgentDist, PolicyUnKron, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, 2, simoptions); % The 2 is for Parallel (use GPU)
            
            AgentDist=StationaryDist_FHorz_Case1_TPath_SingleStep(AgentDist,AgeWeightsParamNames,Policy,n_d,n_a,n_z,N_j,pi_z,Parameters,simoptions);
            
            AggVarsPath(:,tt)=AggVars;
        end
        AggVarsFullPath(:,:,ii)=AggVarsPath;
    end
    
    
    % Note: Cannot do transition paths in which the mass of each agent type changes.
    AggVarsPooledPath=reshape(PTypeStructure.FnsAndPTypeIndicator,[PTypeStructure.numFnsToEvaluate,1,PTypeStructure.N_i]).*sum(AggVarsFullPath(:,:,ii).*shiftdim(StationaryDist_init.ptweights,-2),3); % Weighted sum over agent type dimension
    AggVarNames=fieldnames(FullFnsToEvaluate);
    
    for tt=1:T-1
        % Note that the parameters that are relevant to the GeneralEqmEqns
        % (those in GeneralEqmEqnParamNames) must be independent of agent
        % type. So arbitrarilty use the last agent (current content of iistr)
        Parameters=PTypeStructure.(iistr).Parameters;

        %An easy way to get the new prices is just to call GeneralEqmConditions_Case1
        %and then adjust it for the current prices
            % When using negative powers matlab will often return complex
            % numbers, even if the solution is actually a real number. I
            % force converting these to real, albeit at the risk of missing problems
            % created by actual complex numbers.
        if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
            for ff=1:length(AggVarNames)
                Parameters.(AggVarNames{ff})=AggVarsPooledPath(ff,tt);
            end
            PricePathNew(tt,:)=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
        elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
            % Remark: following assumes that there is one'GeneralEqmEqnParameter' per 'GeneralEqmEqn'
            for j=1:length(GeneralEqmEqns)
                for ff=1:length(AggVarNames)
                    Parameters.(AggVarNames{ff})=AggVarsPooledPath(ff,tt);
                end
                GEeqn_temp=@(GEprices) sum(real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2)).^2);
                PricePathNew(tt,j)=fminsearch(GEeqn_temp,GEprices);
            end
        % Note there is no GEnewprice==2, it uses a completely different code
        elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
            for ff=1:length(AggVarNames)
                Parameters.(AggVarNames{ff})=AggVarsPooledPath(ff,tt);
            end
            p_i=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
%             GEcondnspath(i,:)=p_i;
            p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;
            PricePathNew(tt,:)=(PricePathOld(tt,:).*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
        end
    end
    
%     % Free up space on GPU by deleting things no longer needed
%     clear AgentDist
    
    %See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    %Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1
        fprintf('Number of iteration on the path: %i \n',pathcounter)
        
        % Would be nice to have a way to get the iteration count without having the whole
        % printout of path values (I think that would be useful?)
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
            subplot(nrows,ncolumns,pp); plot(AggVarsPooledPath(:,pp))
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
    
    % Set price path to be 9/10ths the old path and 1/10th the new path (but making sure to leave prices in periods 1 & T unchanged).
    if transpathoptions.weightscheme==0
        PricePathOld=PricePathNew; % The update weights are already in GEnewprice setup
    elseif transpathoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight.*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight).*PricePathNew(1:T-1,:);
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
    
    TransPathConvergence=PricePathDist/transpathoptions.tolerance; %So when this gets to 1 we have convergence
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