function [PricePathOld,GEcondnPath]=TransitionPath_MixHorz_PType_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_initial, FullFnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, PricePathSizeVec_ii, ParamPathSizeVec_ii, GEeqnNames,nGeneralEqmEqns,nGeneralEqmEqns_acrossptypes,GeneralEqmEqnsCell,GeneralEqmEqnParamNames, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, use_stockvars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, stockvarsNames, stockvarInPricePathNames, transpathoptions, PTypeStructure)
% This code will work for all transition paths except those that involve at
% change in the transition matrix pi_z (can handle a change in pi_z, but
% only if it is a 'surprise', not anticipated changes)

% PricePathOld is matrix of size T-by-'number of prices'
% ParamPath is matrix of size T-by-'number of parameters that change over path'

% Remark to self: No real need for T as input, as this is anyway the length of PricePathOld

N_i=PTypeStructure.N_i; % For convenience
FullAggVarNames=fieldnames(FullFnsToEvaluate);

l_p=size(PricePathOld,2);

if transpathoptions.verbose==1
    % Set up some things to be used later
    pathnametitles=strjoin(PricePathNames,' ');
    wpathnametitle=10*length(PricePathNames); % roughly the space that will use to print the prices themselves
end


%% Setup used for graphs
% For graph of the Prices
if length(PricePathNames)>12
    ncolumns_pricepath=4;
elseif length(PricePathNames)>6
    ncolumns_pricepath=3;
else
    ncolumns_pricepath=2;
end
nrows_pricepath=ceil(length(PricePathNames)/ncolumns_pricepath);

pp_indexinpricepath=zeros(1,length(PricePathNames));
pp_c=0;
for pp=1:length(PricePathNames)
    if PTypeStructure.PricePath_Idependsonptype(pp)==0
        pp_c=pp_c+1;
        pp_indexinpricepath(pp)=pp_c;
    else
        pp_c=pp_c+1;
        pp_indexinpricepath(pp)=pp_c;
        pp_c=pp_c+(N_i-1);
    end
end

% For graph of the AggVars
if length(FullAggVarNames)>12
    ncolumns_aggvars=4;
elseif length(FullAggVarNames)>6
    ncolumns_aggvars=3;
else
    ncolumns_aggvars=2;
end
nrows_aggvars=ceil(length(FullAggVarNames)/ncolumns_aggvars);

% which AggVars are used in a GE condition that gets evaluated based on ptype (so can plot them too)
aa_aggvarbyptype=zeros(1,length(FullAggVarNames));
for aa=1:length(FullAggVarNames)
    for gg=1:length(GEeqnNames)
        if transpathoptions.GEptype(gg)==1
            % if any(strcmp(GeneralEqmEqnParamNames(gg).Names,FullAggVarNames{aa}). But we have to allow for the inputs to GE to have endings like _tminus1
            temp=FullAggVarNames{aa};
            for aa2=1:length(GeneralEqmEqnParamNames(gg).Names)
                temp2=GeneralEqmEqnParamNames(gg).Names{aa2};
                if length(temp2)==length(temp)
                    if strcmp(temp2,temp)
                        aa_aggvarbyptype(aa)=1; % aa AggVar is in this GE condition that depends on ptype
                    end
                elseif length(temp2)>length(temp)
                    if strcmp(temp2(1:length(temp)+1),[temp,'_']) % all the 'versions' have an underscore (e.g. _tminus1 or _tplus1)
                        aa_aggvarbyptype(aa)=1; % aa AggVar is in this GE condition that depends on ptype
                    end
                end
            end
        end
    end
end

% For graph of the General Eqm Conditions
if length(GEeqnNames)>12
    ncolumns_GEcondns=4;
elseif length(GEeqnNames)>6
    ncolumns_GEcondns=3;
else
    ncolumns_GEcondns=2;
end
nrows_GEcondns=ceil(length(GEeqnNames)/ncolumns_GEcondns);

gg_indexinGEcondns=zeros(1,length(GEeqnNames));
gg_c=0;
for gg=1:length(GEeqnNames)
    if transpathoptions.GEptype(gg)==0
        gg_c=gg_c+1;
        gg_indexinGEcondns(gg)=gg_c;
    else
        gg_c=gg_c+1;
        gg_indexinGEcondns(gg)=gg_c;
        gg_c=gg_c+(N_i-1);
    end
end

%%
for ii=1:N_i
    iistr=PTypeStructure.Names_i{ii};
    if isfinite(PTypeStructure.(iistr).N_j)
        [PTypeStructure.(iistr).PolicyIndexesPath,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).II1,PTypeStructure.(iistr).II2,PTypeStructure.(iistr).exceptlastj,PTypeStructure.(iistr).exceptfirstj,PTypeStructure.(iistr).justfirstj]=TransitionPath_FHorz_substeps_Step0_setup(PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).N_j,T,transpathoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
    else
        [PTypeStructure.(iistr).PolicyIndexesPath,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).II1,PTypeStructure.(iistr).II2]=TransitionPath_InfHorz_substeps_Step0_setup(PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,T,transpathoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
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
    AggVarsFullPath=zeros(PTypeStructure.numFnsToEvaluate,T-1,N_i); % Does not include period T
    for ii=1:N_i
        iistr=PTypeStructure.Names_i{ii};

        % Following few lines I would normally do outside of the while loop, but have to set them for each ptype
        % AgentDist=AgentDist_initial.(iistr);
        % WARNING: The following would overwrite themselves next iteration
        % V_final=V_final.(iistr);
        % AgeWeights_T=AgeWeights_T.(iistr);
        % jequalOneDist_T=jequalOneDist_T.(iistr);

        % Some parts of PricePath and ParamPath may depend on ptype
        % Get just the values that correspond to the current ptype
        PricePathOld_ii=PricePathOld(:,PTypeStructure.(iistr).RelevantPricePath);
        ParamPath_ii=ParamPath(:,PTypeStructure.(iistr).RelevantParamPath);

        % Have not yet set up the following to allow dependence on ptype (should do this at some point)
        PricePathNames_ii=PricePathNames;
        ParamPathNames_ii=ParamPathNames;

        PolicyIndexesPath_ii=PTypeStructure.(iistr).PolicyIndexesPath;

        if isfinite(PTypeStructure.(iistr).N_j)
            %% Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            [~,PolicyIndexesPath_ii]=TransitionPath_FHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath_ii,V_final.(iistr),PTypeStructure.(iistr).Parameters,PricePathOld_ii,ParamPath_ii,PricePathSizeVec_ii,ParamPathSizeVec_ii,PricePathNames_ii,ParamPathNames_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).n_e,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).d_gridvals, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals_J,PTypeStructure.(iistr).e_gridvals_J,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).pi_e_J,PTypeStructure.(iistr).ReturnFn,PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, transpathoptions,PTypeStructure.(iistr).vfoptions);

            %% Modify PolicyIndexesPath into forms needed for forward iteration
            [PolicyPath_ForAgentDistIter_ii,PolicyProbsPath_ii,PolicyValuesPath_ii]=TransitionPath_FHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath_ii,T,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).n_e,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).d_gridvals,PTypeStructure.(iistr).aprime_gridvals,transpathoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
        else
            %% Go from T-1 to 1 calculating the Value function and Optimal policy function at each step.
            warning("TransitionPath_Case1_MixHorz_PType_shooting not yet hanlding e_gridvals and/or pi_e")
            [~,PolicyIndexesPath_ii]=TransitionPath_InfHorz_substeps_Step1_ValueFnIter(T,PolicyIndexesPath_ii,V_final.(iistr),PTypeStructure.(iistr).Parameters,PricePathOld_ii,ParamPath_ii,PricePathSizeVec_ii,ParamPathSizeVec_ii,PricePathNames_ii,ParamPathNames_ii,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).n_e,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).d_gridvals, PTypeStructure.(iistr).a_grid, PTypeStructure.(iistr).z_gridvals,[],PTypeStructure.(iistr).pi_z,[],PTypeStructure.(iistr).ReturnFn,PTypeStructure.(iistr).DiscountFactorParamNames, PTypeStructure.(iistr).ReturnFnParamNames, transpathoptions,PTypeStructure.(iistr).vfoptions);

            %% Modify PolicyIndexesPath into forms needed for forward iteration
            [PolicyPath_ForAgentDistIter_ii,PolicyProbsPath_ii,PolicyValuesPath_ii]=TransitionPath_InfHorz_substeps_Step2_AdjustPolicy(PolicyIndexesPath_ii,T,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).n_d,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).n_e,PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).d_gridvals,PTypeStructure.(iistr).aprime_gridvals,transpathoptions,PTypeStructure.(iistr).vfoptions,PTypeStructure.(iistr).simoptions);
        end

        %% Iterate forward over t: iterate agent dist, calculate aggvars, evaluate general eqm
        % Call AgentDist the current periods distn and AgentDistnext the next periods distn which we must calculate
        AgentDist_ii=AgentDist_initial.(iistr);
        AggVarNames_ii=PTypeStructure.(iistr).AggVarNames;
        AggVarsPath_ii=zeros(length(AggVarNames_ii),T-1);

        % Initialise _tminus1 entries in Parameters from initialvalues (used at tt=1)
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames.(iistr))
                Parameters.([tminus1AggVarsNames.(iistr){pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames.(iistr){pp});
            end
        end
        if use_stockvars==1
            for pp=1:length(stockvarsNames)
                Parameters.([stockvarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(stockvarsNames{pp});
            end
        end

        for tt=1:T-1

            % Get t-1 PricePath, ParamPath and AggVars before we update them
            if tt>1
                if use_tminus1price==1
                    for pp=1:length(tminus1priceNames)
                        Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                    end
                end
                if use_tminus1params==1
                    for pp=1:length(tminus1paramNames)
                        Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                    end
                end
                if use_tminus1AggVars==1
                    for pp=1:length(tminus1AggVarsNames.(iistr))
                        % The AggVars have not yet been updated, so they still contain previous period values
                        Parameters.([tminus1AggVarsNames.(iistr){pp},'_tminus1'])=Parameters.(tminus1AggVarsNames.(iistr){pp});
                    end
                end
                if use_stockvars==1 % Comes from PricePathNew, unlike the _tminus1price, which comes from the PricePathOld
                    for pp=1:length(stockvarsNames)
                        Parameters.([stockvarsNames{pp},'_tminus1'])=PricePathNew(tt-1,PricePathSizeVec(1,stockvarInPricePathNames(pp)):PricePathSizeVec(2,stockvarInPricePathNames(pp)));
                    end
                end
            end

            % Update current PricePath and ParamPath
            for pp=1:length(PricePathNames)
                Parameters.(PricePathNames{pp})=PricePathOld_ii(tt,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
            end
            for pp=1:length(ParamPathNames)
                Parameters.(ParamPathNames{pp})=ParamPath_ii(tt,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
            end

            % Get t+1 PricePath
            if use_tplus1price==1
                for pp=1:length(tplus1priceNames)
                    kk=tplus1pricePathkk(pp);
                    Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld_ii(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
                end
            end

            if isfinite(PTypeStructure.(iistr).N_j)
                %% Get the current optimal policy, and iterate the agent dist
                if transpathoptions.(iistr).trivialjequalonedist==0
                    PTypeStructure.(iistr).jequalOneDist=PTypeStructure.(iistr).jequalOneDist_T(:,tt+1);  % Note: t+1 as we are about to create the next period AgentDist
                end
                if ndims(PTypeStructure.(iistr).AgeWeights_T)==2
                    AgeWeights_ii=PTypeStructure.(iistr).AgeWeights_T(:,tt);
                elseif PTypeStructure.(iistr).simoptions.fastOLG==0 || PTypeStructure.(iistr).N_e>0
                    AgeWeights_ii=PTypeStructure.(iistr).AgeWeights_T(:,:,tt);
                else % simoptions.fastOLG==1
                    AgeWeights_ii=PTypeStructure.(iistr).AgeWeights_T(:,tt);
                end

                AgentDistnext_ii=TransitionPath_FHorz_substeps_Step3tt_IterAgentDist(AgentDist_ii,PolicyPath_ForAgentDistIter_ii,PolicyProbsPath_ii,tt,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).pi_z_J,PTypeStructure.(iistr).pi_z_J_sim,PTypeStructure.(iistr).pi_e_J,PTypeStructure.(iistr).pi_e_J_sim,PTypeStructure.(iistr).II1,PTypeStructure.(iistr).II2,PTypeStructure.(iistr).exceptlastj,PTypeStructure.(iistr).exceptfirstj,PTypeStructure.(iistr).justfirstj,PTypeStructure.(iistr).jequalOneDist,transpathoptions,PTypeStructure.(iistr).simoptions);

                %% AggVars
                if PTypeStructure.(iistr).N_z==0 && PTypeStructure.(iistr).N_e==0
                    PVP_ii_t=PolicyValuesPath_ii(:,:,:,tt);
                else
                    PVP_ii_t=PolicyValuesPath_ii(:,:,:,:,tt);
                end
                AggVars_ii=TransitionPath_FHorz_substeps_Step4tt_AggVars(AgentDist_ii,AgeWeights_ii,PVP_ii_t,tt,PTypeStructure.(iistr).FnsToEvaluateCell,PTypeStructure.(iistr).FnsToEvaluateParamNames,AggVarNames_ii,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).N_j,PTypeStructure.(iistr).l_d,PTypeStructure.(iistr).l_aprime,PTypeStructure.(iistr).l_a,PTypeStructure.(iistr).l_z,PTypeStructure.(iistr).l_e,PTypeStructure.(iistr).N_d,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).a_gridvals,PTypeStructure.(iistr).ze_gridvals_J_fastOLG,transpathoptions);
            else
                warning("TransitionPath_MixHorz_PType_shooting not hanlding pi_e yet")
                AgentDist_ii_InfHorz=reshape(AgentDist_ii,[PTypeStructure.(iistr).N_a*PTypeStructure.(iistr).N_z,1]);
                AgentDistnext_ii=TransitionPath_InfHorz_substeps_Step3tt_IterAgentDist(AgentDist_ii_InfHorz,PolicyPath_ForAgentDistIter_ii,PolicyProbsPath_ii,tt,PTypeStructure.(iistr).N_a,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).N_probs,PTypeStructure.(iistr).pi_z,[],PTypeStructure.(iistr).II1,PTypeStructure.(iistr).II2,transpathoptions,PTypeStructure.(iistr).simoptions);
                if PTypeStructure.(iistr).N_z==0 && PTypeStructure.(iistr).N_e==0
                    PVP_ii_t=PolicyValuesPath_ii(:,:,tt);
                else
                    PVP_ii_t=PolicyValuesPath_ii(:,:,:,tt);
                end
                AggVars_ii=TransitionPath_InfHorz_substeps_Step4tt_AggVars(AgentDist_ii,PVP_ii_t,tt,PTypeStructure.(iistr).FnsToEvaluateCell,PTypeStructure.(iistr).FnsToEvaluateParamNames,AggVarNames_ii,PTypeStructure.(iistr).Parameters,PTypeStructure.(iistr).n_a,PTypeStructure.(iistr).n_z,PTypeStructure.(iistr).n_e,PTypeStructure.(iistr).N_z,PTypeStructure.(iistr).N_e,PTypeStructure.(iistr).a_gridvals,PTypeStructure.(iistr).z_gridvals,transpathoptions);
            end

            for ff=1:length(AggVarNames_ii)
                Parameters.(AggVarNames_ii{ff})=AggVars_ii.(AggVarNames_ii{ff}).Mean;
            end

            % Keep AggVars in the AggVarsPath
            for ff=1:length(AggVarNames_ii)
                AggVarsPath_ii(ff,tt)=AggVars_ii.(AggVarNames_ii{ff}).Mean;
            end

            %% GE EQNS THAT DEPEND ON PTYPE SHOULD BE DONE HERE!!!

            AgentDist_ii=AgentDistnext_ii;
        end

        AggVarsFullPath(logical(PTypeStructure.(iistr).WhichFnsForCurrentPType),:,ii)=AggVarsPath_ii;

    end % done loop over ii


    %% Note: Cannot yet do transition paths in which the mass of each agent type changes.
    % AggVarsPooledPath=sum(reshape(PTypeStructure.FnsAndPTypeIndicator,[PTypeStructure.numFnsToEvaluate,1,PTypeStructure.N_i]).*AggVarsFullPath.*shiftdim(AgentDist_init.ptweights,-2),3); % Weighted sum over agent type dimension
    % Note: don't need the above line, as I already dealt with PTypeStructure.FnsAndPTypeIndicator when creating AggVarsFullPath
    AggVarsPooledPath=sum(AggVarsFullPath.*shiftdim(AgentDist_initial.ptweights,-2),3); % Weighted sum over agent type dimension


    %% Do the general eqm conditions and create PricePathNew based on these
    if all(transpathoptions.GEptype==0)
        GECondnPath=zeros(T,length(GEeqnNames));

        % NEEDED???  Restore all AggVarNames and tminus1AggVarsNames for GEeqns
        AggVarNames=cell(1,N_i);
        tminus1AggVarsNames=cell(1,N_i);
        use_tminus1AggVars=0;
        for ii=1:N_i
            iistr=PTypeStructure.Names_i{ii};
            AggVarNames{ii}=PTypeStructure.(iistr).AggVarNames;
            if isfield(PTypeStructure.(iistr), 'tminus1AggVarsNames')
                use_tminus1AggVars=1;
                tminus1AggVarsNames{ii}=PTypeStructure.(iistr).tminus1AggVarsNames;
            end
        end
        AggVarNames=vertcat(AggVarNames{:});
        tminus1AggVarsNames=vertcat(tminus1AggVarsNames{:});

        % Parameters that may be relevant to General Eqm
        Parameters=PTypeStructure.ParametersRaw;

        % Initialise _tminus1 entries in Parameters from initialvalues (used at tt=1)
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames)
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
        if use_stockvars==1
            for pp=1:length(stockvarsNames)
                Parameters.([stockvarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(stockvarsNames{pp});
            end
        end

        for tt=1:T-1

            % Get t-1 PricePath, ParamPath and AggVars before we update them
            if tt>1
                if use_tminus1price==1
                    for pp=1:length(tminus1priceNames)
                        Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                    end
                end
                if use_tminus1params==1
                    for pp=1:length(tminus1paramNames)
                        Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                    end
                end
                if use_tminus1AggVars==1
                    for pp=1:length(tminus1AggVarsNames)
                        % The AggVars have not yet been updated, so they still contain previous period values
                        Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                    end
                end
                if use_stockvars==1 % Comes from PricePathNew, unlike the _tminus1price, which comes from the PricePathOld
                    for pp=1:length(stockvarsNames)
                        Parameters.([stockvarsNames{pp},'_tminus1'])=PricePathNew(tt-1,PricePathSizeVec(1,stockvarInPricePathNames(pp)):PricePathSizeVec(2,stockvarInPricePathNames(pp)));
                    end
                end
            end

            % Update current PricePath and ParamPath
            for pp=1:length(PricePathNames)
                Parameters.(PricePathNames{pp})=PricePathOld(tt,PricePathSizeVec(1,pp):PricePathSizeVec(2,pp));
            end
            for pp=1:length(ParamPathNames)
                Parameters.(ParamPathNames{pp})=ParamPath(tt,ParamPathSizeVec(1,pp):ParamPathSizeVec(2,pp));
            end

            % Update current AggVars [we have to add this as GE conditions are in a separate tt loop to the AggVars]
            for ff=1:length(FullAggVarNames)
                Parameters.(FullAggVarNames{ff})=AggVarsPooledPath(ff,tt);
            end

            % Get t+1 PricePath
            if use_tplus1price==1
                for pp=1:length(tplus1priceNames)
                    kk=tplus1pricePathkk(pp);
                    Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
                end
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
            % Evaluate the general eqm conditions, and based on them create PricePathNew (interpretation depends on transpathoptions)
            [PricePathNew_tt,GEcondnPath_tt]=updatePricePathNew_TPath_tt(Parameters,GeneralEqmEqnsCell,GeneralEqmEqnParamNames,PricePathOld(tt,:),transpathoptions);
            PricePathNew(tt,:)=PricePathNew_tt;
            GEcondnPath(tt,:)=GEcondnPath_tt;

        end % Done loop over tt, evaluating the GE conditions
    else % Some GE conditions depend on PType
        GECondnPath=zeros(T,nGeneralEqmEqns_acrossptypes);
        error("need to fit these loops together")

        % NEEDED??? Restore AggVarNames and tminus1AggVarsNames by PType for GEeqns
        AggVarNames=cell(1,N_i);
        tminus1AggVarsNames=cell(1,N_i);
        use_tminus1AggVars=0;
        for ii=1:N_i
            iistr=PTypeStructure.Names_i{ii};
            AggVarNames{ii}=PTypeStructure.(iistr).AggVarNames;
            tminus1AggVarsNames{ii}=PTypeStructure.(iistr).tminus1AggVarsNames;
            if ~isempty(tminus1AggVarsNames{ii})
                use_tminus1AggVars=1;
            end
        end
        AggVarNames=vertcat(AggVarNames{:});
        tminus1AggVarsNames=vertcat(tminus1AggVarsNames{:});

        % Parameters that may be relevant to General Eqm
        Parameters=PTypeStructure.ParametersRaw;
        for ii=1:N_i
            iistr=PTypeStructure.Names_i{ii};
            Parameters_ii.(iistr)=PTypeStructure.(iistr).Parameters; % For use with General Eqm conditions that are evaluated conditional on ptype
        end

        % Some of the General eqm eqns depend on ptype

        % Initialise _tminus1 entries in Parameters (and Parameters_ii) from initialvalues (used at tt=1)
        if use_tminus1price==1
            for pp=1:length(tminus1priceNames)
                Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                if isstruct(Parameters.([tminus1priceNames{pp},'_tminus1']))
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=Parameters.([tminus1priceNames{pp},'_tminus1']).(iistr);
                    end
                elseif length(Parameters.([tminus1priceNames{pp},'_tminus1']))==N_i % Depends on ptype
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        tempii=Parameters.([tminus1priceNames{pp},'_tminus1']);
                        Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=tempii(ii);
                    end
                end
            end
        end
        if use_tminus1params==1
            for pp=1:length(tminus1paramNames)
                Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
                if isstruct(Parameters.([tminus1paramNames{pp},'_tminus1']))
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=Parameters.([tminus1paramNames{pp},'_tminus1']).(iistr);
                    end
                elseif length(Parameters.([tminus1paramNames{pp},'_tminus1']))==N_i % Depends on ptype
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        tempii=Parameters.([tminus1paramNames{pp},'_tminus1']);
                        Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=tempii(ii);
                    end
                end
            end
        end
        if use_tminus1AggVars==1
            for pp=1:length(tminus1AggVarsNames)
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
                if isstruct(transpathoptions.initialvalues.(tminus1AggVarsNames{pp}))
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp}).(iistr);
                    end
                elseif length(transpathoptions.initialvalues.(tminus1AggVarsNames{pp}))==N_i % Depends on ptype
                    temp=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=temp(ii);
                    end
                end
            end
        end
        if use_stockvars==1
            for pp=1:length(stockvarsNames)
                Parameters.([stockvarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(stockvarsNames{pp});
                if isstruct(transpathoptions.initialvalues.(stockvarsNames{pp}))
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).([stockvarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(stockvarsNames{pp}).(iistr);
                    end
                elseif length(transpathoptions.initialvalues.(stockvarsNames{pp}))==N_i % Depends on ptype
                    temp=transpathoptions.initialvalues.(stockvarsNames{pp});
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).([stockvarsNames{pp},'_tminus1'])=temp(ii);
                    end
                end
            end
        end

        for tt=1:T-1

            % Get t-1 PricePath, ParamPath and AggVars before we update them
            if tt>1
                if use_tminus1price==1
                    for pp=1:length(tminus1priceNames)
                        Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                        if isstruct(Parameters.([tminus1priceNames{pp},'_tminus1']))
                            for ii=1:N_i
                                iistr=PTypeStructure.Names_i{ii};
                                Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=Parameters.([tminus1priceNames{pp},'_tminus1']).(iistr);
                            end
                        elseif length(Parameters.([tminus1priceNames{pp},'_tminus1']))==N_i % Depends on ptype
                            for ii=1:N_i
                                iistr=PTypeStructure.Names_i{ii};
                                tempii=Parameters.([tminus1priceNames{pp},'_tminus1']);
                                Parameters_ii.(iistr).([tminus1priceNames{pp},'_tminus1'])=tempii(ii);
                            end
                        end
                    end
                end
                if use_tminus1params==1
                    for pp=1:length(tminus1paramNames)
                        Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                        if isstruct(Parameters.([tminus1paramNames{pp},'_tminus1']))
                            for ii=1:N_i
                                iistr=PTypeStructure.Names_i{ii};
                                Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=Parameters.([tminus1paramNames{pp},'_tminus1']).(iistr);
                            end
                        elseif length(Parameters.([tminus1paramNames{pp},'_tminus1']))==N_i % Depends on ptype
                            for ii=1:N_i
                                iistr=PTypeStructure.Names_i{ii};
                                tempii=Parameters.([tminus1paramNames{pp},'_tminus1']);
                                Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=tempii(ii);
                            end
                        end
                    end
                end
                if use_tminus1AggVars==1
                    for pp=1:length(tminus1AggVarsNames)
                        % The AggVars have not yet been updated, so they still contain previous period values
                        Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=Parameters.(tminus1AggVarsNames{pp});
                        if length(Parameters.(tminus1AggVarsNames{pp}))==N_i % Depends on ptype
                            for ii=1:N_i
                                iistr=PTypeStructure.Names_i{ii};
                                Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=Parameters_ii.(iistr).(tminus1AggVarsNames{pp});
                            end
                        end
                    end
                end
                if use_stockvars==1 % Comes from PricePathNew, unlike the _tminus1price, which comes from the PricePathOld
                    for pp=1:length(stockvarsNames)
                        Parameters.([stockvarsNames{pp},'_tminus1'])=PricePathNew(tt-1,PricePathSizeVec(1,stockvarInPricePathNames(pp)):PricePathSizeVec(2,stockvarInPricePathNames(pp)));
                        if (PricePathSizeVec(2,stockvarInPricePathNames(pp))-PricePathSizeVec(1,stockvarInPricePathNames(pp))+1)==N_i % Depends on ptype
                            for ii=1:N_i
                                iistr=PTypeStructure.Names_i{ii};
                                Parameters_ii.(iistr).([stockvarsNames{pp},'_tminus1'])=PricePathNew(tt-1,PricePathSizeVec(1,stockvarInPricePathNames(pp))+ii-1);
                            end
                        end
                    end
                end
            end


            % Update current PricePath and ParamPath
            for kk=1:length(PricePathNames)
                Parameters.(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
                if (PricePathSizeVec(2,kk)-PricePathSizeVec(1,kk)+1)==N_i
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).(PricePathNames{kk})=PricePathOld(tt,PricePathSizeVec(1,kk)+ii-1);
                    end
                end
            end
            for kk=1:length(ParamPathNames)
                Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
                if (ParamPathSizeVec(2,kk)-ParamPathSizeVec(1,kk)+1)==N_i
                    for ii=1:N_i
                        iistr=PTypeStructure.Names_i{ii};
                        Parameters_ii.(iistr).(ParamPathNames{kk})=PricePathOld(tt,ParamPathSizeVec(1,kk)+ii-1);
                    end
                end
            end

            % Update current AggVars [we have to add this when doing ptype as GE conditions are in a separate tt loop to the AggVars]
            for ff=1:length(FullAggVarNames)
                Parameters.(FullAggVarNames{ff})=AggVarsPooledPath(ff,tt);
                % Keep the AggVars conditional on ptype for all the AggVars; overkill but that is fine
                for ii=1:N_i
                    iistr=PTypeStructure.Names_i{ii};
                    Parameters_ii.(iistr).(FullAggVarNames{ff})=AggVarsFullPath(ff,tt,ii);
                end
            end


            % Get t+1 PricePath
            if use_tplus1price==1
                for pp=1:length(tplus1priceNames)
                    kk=tplus1pricePathkk(pp);
                    Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
                    if isstruct(Parameters.([tplus1priceNames{pp},'_tplus1']))
                        for ii=1:N_i
                            iistr=PTypeStructure.Names_i{ii};
                            Parameters_ii.(iistr).([tplus1priceNames{pp},'_tplus1'])=Parameters.([tplus1priceNames{pp},'_tplus1']).(iistr);
                        end
                    elseif length(Parameters.([tplus1priceNames{pp},'_tplus1']))==N_i % Depends on ptype
                        temp=Parameters.([tplus1priceNames{pp},'_tplus1']);
                        for ii=1:N_i
                            iistr=PTypeStructure.Names_i{ii};
                            Parameters_ii.(iistr).([tplus1priceNames{pp},'_tplus1'])=temp(ii);
                        end
                    end
                end
            end


            if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
                % Loop over the general eqm conditions, so we can deal seperately with those that depend on ptype and those that do not
                gg_c=0;
                for gg=1:nGeneralEqmEqns
                    if transpathoptions.GEptype(gg)==0
                        gg_c=gg_c+1;
                        PricePathNew(tt,gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg},GeneralEqmEqnParamNames(gg).Names, Parameters));
                    elseif transpathoptions.GEptype(gg)==1
                        gg_c=gg_c+1;
                        PricePathNew(tt,gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_ii.(iistr)));
                    end
                end
            % Note there is no GEnewprice==2, it uses a completely different code
            elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
                p_i=zeros(1,length(GeneralEqmEqnsCell)+sum(transpathoptions.GEptype),'gpuArray');
                gg_c=0;
                for gg=1:nGeneralEqmEqns
                    if transpathoptions.GEptype(gg)==0
                        gg_c=gg_c+1;
                        p_i(gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters));
                    elseif transpathoptions.GEptype(gg)==1
                        for ii=1:N_i
                            iistr=PTypeStructure.Names_i{ii};
                            gg_c=gg_c+1;
                            p_i(gg_c)=real(GeneralEqmConditions_Case1_v3g(GeneralEqmEqnsCell{gg}, GeneralEqmEqnParamNames(gg).Names, Parameters_ii.(iistr)));
                        end
                    end
                end

                p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
                I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
                p_i=I_makescutoff.*p_i;
                PricePathNew(tt,:)=(PricePathOld(tt,:).*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
                GEcondnPath(tt,:)=p_i;
            end

        end % Done loop over tt, evaluating the GE conditions

    end


    %% Now we just check for convergence, update prices, and give some feedback on progress
    % See how far apart the price paths are
    PricePathDist=max(abs(reshape(PricePathNew(1:T-1,:)-PricePathOld(1:T-1,:),[numel(PricePathOld(1:T-1,:)),1])));
    % Notice that the distance is always calculated ignoring the time t=T periods, as these needn't ever converges

    if transpathoptions.verbose==1
        fprintf(' \n')
        fprintf('%-*s || %-*s \n',wpathnametitle,'Old',wpathnametitle,'New')
        fprintf('%-*s || %-*s \n',wpathnametitle,pathnametitles,wpathnametitle,pathnametitles)

        % Would be nice to have a way to get the iteration count without having the whole printout of path values (I think that would be useful?)
        [PricePathOld,PricePathNew]
    end

    % Create plots of the transition path (before we update pricepath)
    createTPathFeedbackPlots(PricePathNames,FullAggVarNames,GEeqnNames,PricePathOld,AggVarsPooledPath,GEcondnPath,transpathoptions);

    % Update PricePathOld
    PricePathOld=updatePricePath(PricePathOld,PricePathNew,transpathoptions,T);

    TransPathConvergence=PricePathDist/transpathoptions.tolerance; %So when this gets to 1 we have convergence
    if transpathoptions.verbose==1
        fprintf('Number of iterations on transition path: %i \n',pathcounter)
        fprintf('Current distance between old and new price path (in L-Infinity norm): %8.6f \n', PricePathDist)
        fprintf('Ratio of current distance to the convergence tolerance: %.2f (convergence when reaches 1) \n',TransPathConvergence)
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
