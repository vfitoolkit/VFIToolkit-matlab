function PricePathOld=TransitionPath_Case1_FHorz_PType_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_init, AgeWeights_T, FullFnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, PTypeStructure)
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
    for ii=1:length(PricePathNames)
        pathnametitles{ii}={['Old ',PricePathNames{ii}]};
        pathnametitles{ii+length(PricePathNames)}={['New ',PricePathNames{ii}]};
    end
end
if transpathoptions.verbose>1
    DiscountFactorParamNames
    ReturnFnParamNames
    ParamPathNames
    PricePathNames
end


%% Set up GEnewprice==3 (if relevant)
if transpathoptions.GEnewprice==3
    transpathoptions.weightscheme=0;
    
    if isstruct(GeneralEqmEqns) 
        % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
        % Is same as order of fields in GeneralEqmEqns
        % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
        temp=transpathoptions.GEnewprice3.howtoupdate;
        GEeqnNames=fieldnames(GeneralEqmEqns);
        for ii=1:length(GEeqnNames)
            for jj=1:size(temp,1)
                if strcmp(temp{jj,1},GEeqnNames{ii}) % Names match
                    transpathoptions.GEnewprice3.howtoupdate{ii,1}=temp{jj,1};
                    transpathoptions.GEnewprice3.howtoupdate{ii,2}=temp{jj,2};
                    transpathoptions.GEnewprice3.howtoupdate{ii,3}=temp{jj,3};
                    transpathoptions.GEnewprice3.howtoupdate{ii,4}=temp{jj,4};
                end
            end
        end
        nGeneralEqmEqns=length(GEeqnNames);
    else
        nGeneralEqmEqns=length(GeneralEqmEqns);
    end
    transpathoptions.GEnewprice3.add=[transpathoptions.GEnewprice3.howtoupdate{:,3}];
    transpathoptions.GEnewprice3.factor=[transpathoptions.GEnewprice3.howtoupdate{:,4}];
    transpathoptions.GEnewprice3.keepold=ones(size(transpathoptions.GEnewprice3.factor));
    transpathoptions.GEnewprice3.keepold=ones(size(transpathoptions.GEnewprice3.factor));
    tempweight=transpathoptions.oldpathweight;
    transpathoptions.oldpathweight=zeros(size(transpathoptions.GEnewprice3.factor));
    for ii=1:length(transpathoptions.GEnewprice3.factor)
        if transpathoptions.GEnewprice3.factor(ii)==Inf
            transpathoptions.GEnewprice3.factor(ii)=1;
            transpathoptions.GEnewprice3.keepold(ii)=0;
            transpathoptions.oldpathweight(ii)=tempweight;
        end
    end
    if size(transpathoptions.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns && nGeneralEqmEqns==length(PricePathNames)
        % do nothing, this is how things should be
    else
        error('transpathoptions.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (different number of conditions/prices) \n')
    end
    transpathoptions.GEnewprice3.permute=zeros(size(transpathoptions.GEnewprice3.howtoupdate,1),1);
    for ii=1:size(transpathoptions.GEnewprice3.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for jj=1:length(PricePathNames)
            if strcmp(transpathoptions.GEnewprice3.howtoupdate{ii,2},PricePathNames{jj})
                transpathoptions.GEnewprice3.permute(ii)=jj;
            end
        end
    end
    if isfield(transpathoptions,'updateaccuracycutoff')==0
        transpathoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end


%%
PricePathDist=Inf;
pathcounter=1;

PricePathNew=zeros(size(PricePathOld),'gpuArray'); PricePathNew(T,:)=PricePathOld(T,:);

%%
while PricePathDist>transpathoptions.tolerance && pathcounter<=transpathoptions.maxiter
    
    %% For each agent type, first go back through the value & policy fns, then forwards through agent dist and agg vars.
    % After that is finished we can put the AggVars together, evaluate GE conditions, and update price path
    AggVarsFullPath=zeros(PTypeStructure.numFnsToEvaluate,T-1,PTypeStructure.N_i); % Does not include period T
    for ii=1:PTypeStructure.N_i
        iistr=PTypeStructure.Names_i{ii};
        
        % Grab everything relevant out of PTypeStructure
        n_d=PTypeStructure.(iistr).n_d; N_d=prod(n_d); l_d=length(n_d);
        n_a=PTypeStructure.(iistr).n_a; N_a=prod(n_a);
        n_z=PTypeStructure.(iistr).n_z; N_z=prod(n_z);
        n_e=PTypeStructure.(iistr).n_e; N_e=prod(n_e);
        N_j=PTypeStructure.(iistr).N_j;
        d_grid=PTypeStructure.(iistr).d_grid;
        a_grid=PTypeStructure.(iistr).a_grid;
        a_gridvals=PTypeStructure.(iistr).a_gridvals;
        daprime_gridvals=PTypeStructure.(iistr).daprime_gridvals;
        if N_z>0
            z_gridvals_J=PTypeStructure.(iistr).z_gridvals_J;
            pi_z_J=PTypeStructure.(iistr).pi_z_J;
        end
        if N_e>0
            e_gridvals_J=PTypeStructure.(iistr).e_gridvals_J;
            pi_e_J=PTypeStructure.(iistr).pi_e_J;
        end
        ReturnFn=PTypeStructure.(iistr).ReturnFn;
        Parameters=PTypeStructure.(iistr).Parameters;
        DiscountFactorParamNames=PTypeStructure.(iistr).DiscountFactorParamNames;
        ReturnFnParamNames=PTypeStructure.(iistr).ReturnFnParamNames;
        vfoptions=PTypeStructure.(iistr).vfoptions;
        simoptions=PTypeStructure.(iistr).simoptions;
        FnsToEvaluate=PTypeStructure.(iistr).FnsToEvaluate;
        FnsToEvaluateParamNames=PTypeStructure.(iistr).FnsToEvaluateParamNames;
        AggVarNames=PTypeStructure.(iistr).AggVarNames;


        % Following few lines I would normally do outside of the while loop, but have to set them for each ptype
        % AgeWeights=AgeWeights_initial.(iistr);
        % AgentDist=AgentDist_initial.(iistr);
        % AgentDist=AgentDist_initial;
        
        AggVarsPath=TransitionPath_FHorz_PType_singlepath(PricePathOld, ParamPath, PricePathNames,ParamPathNames,T,AgentDist_init.(iistr),V_final.(iistr),AgeWeights_T.(iistr),l_d,N_d,n_d,N_a,n_a,N_z,n_z,N_e,n_e,N_j,d_grid,a_grid,daprime_gridvals,a_gridvals,ReturnFn, FnsToEvaluate, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec, ParamPathSizeVec, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, vfoptions, simoptions);
        % AggVarsPath=zeros(length(FnsToEvaluate),T-1);

        AggVarsFullPath(PTypeStructure.(iistr).WhichFnsForCurrentPType,:,ii)=AggVarsPath;

    end % done loop over ii
    
    
    %% Note: Cannot do transition paths in which the mass of each agent type changes.
    AggVarsPooledPath=reshape(PTypeStructure.FnsAndPTypeIndicator,[PTypeStructure.numFnsToEvaluate,1,PTypeStructure.N_i]).*sum(AggVarsFullPath.*shiftdim(AgentDist_init.ptweights,-2),3); % Weighted sum over agent type dimension
    FullAggVarNames=fieldnames(FullFnsToEvaluate);
    
    for tt=1:T-1
        % Note that the parameters that are relevant to the GeneralEqmEqns
        % (those in GeneralEqmEqnParamNames) must be independent of agent
        % type. So arbitrarilty use the last agent (current content of iistr)
        Parameters=PTypeStructure.(iistr).Parameters;

        GEprices=PricePathOld(tt,:);

        %An easy way to get the new prices is just to call GeneralEqmConditions_Case1
        %and then adjust it for the current prices
            % When using negative powers matlab will often return complex
            % numbers, even if the solution is actually a real number. I
            % force converting these to real, albeit at the risk of missing problems
            % created by actual complex numbers.
        if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
            for ff=1:length(FullAggVarNames)
                Parameters.(FullAggVarNames{ff})=AggVarsPooledPath(ff,tt);
            end
            PricePathNew(tt,:)=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
        elseif transpathoptions.GEnewprice==0 % THIS NEEDS CORRECTING
            % Remark: following assumes that there is one'GeneralEqmEqnParameter' per 'GeneralEqmEqn'
            for j=1:length(GeneralEqmEqns)
                for ff=1:length(FullAggVarNames)
                    Parameters.(FullAggVarNames{ff})=AggVarsPooledPath(ff,tt);
                end
                GEeqn_temp=@(GEprices) sum(real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2)).^2);
                PricePathNew(tt,j)=fminsearch(GEeqn_temp,GEprices);
            end
        % Note there is no GEnewprice==2, it uses a completely different code
        elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
            for ff=1:length(FullAggVarNames)
                Parameters.(FullAggVarNames{ff})=AggVarsPooledPath(ff,tt);
            end
            p_i=real(GeneralEqmConditions_Case1_v2(GeneralEqmEqns,Parameters, 2));
            p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
            I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
            p_i=I_makescutoff.*p_i;
            PricePathNew(tt,:)=(PricePathOld(tt,:).*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
        end

    end % Done loop over tt, evaluating the GE conditions
    
    
    %% Now we just check for convergence, update prices, and give some feedback on progress
    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges
    
    if transpathoptions.verbose==1     
        pathcounter
        disp('Old, New')
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
        if length(FullAggVarNames)>12
            ncolumns=4;
        elseif length(FullAggVarNames)>6
            ncolumns=3;
        else
            ncolumns=2;
        end
        nrows=ceil(length(FullAggVarNames)/ncolumns);
        fig2=figure(2);
        for pp=1:length(FullAggVarNames)
            subplot(nrows,ncolumns,pp); plot(AggVarsPath(:,pp))
            title(FullAggVarNames{pp})
        end
    end
    
    % Set price path to be 9/10ths the old path and 1/10th the new path (but making sure to leave prices in periods 1 & T unchanged).
    if transpathoptions.weightscheme==0
        PricePathOld=PricePathNew; % The update weights are already in GEnewprice setup
    elseif transpathoptions.weightscheme==1 % Just a constant weighting
        PricePathOld(1:T-1,:)=transpathoptions.oldpathweight.*PricePathOld(1:T-1,:)+(1-transpathoptions.oldpathweight).*PricePathNew(1:T-1,:);
    elseif transpathoptions.weightscheme==2 % A exponentially decreasing weighting on new path from (1-oldpathweight) in first period, down to 0.1*(1-oldpathweight) in T-1 period.
        % I should precalculate these weighting vectors
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
