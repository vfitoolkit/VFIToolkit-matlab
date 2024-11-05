function PricePathOld=TransitionPath_Case1_FHorz_PType_shooting(PricePathOld, PricePathNames, ParamPath, ParamPathNames, T, V_final, AgentDist_init, jequalOneDist_T, AgeWeights_T, FullFnsToEvaluate, GeneralEqmEqns, PricePathSizeVec, ParamPathSizeVec, PricePathSizeVec_ii, ParamPathSizeVec_ii, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, transpathoptions, PTypeStructure)
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


%% GE eqns, switch from structure to cell setup
GEeqnNames=fieldnames(GeneralEqmEqns);
nGeneralEqmEqns=length(GEeqnNames);
nGeneralEqmEqns_acrossptypes=sum(transpathoptions.GEptype==0)+N_i*sum(transpathoptions.GEptype==1);

GeneralEqmEqnsCell=cell(1,nGeneralEqmEqns);
for gg=1:nGeneralEqmEqns
    temp=getAnonymousFnInputNames(GeneralEqmEqns.(GEeqnNames{gg}));
    GeneralEqmEqnParamNames(gg).Names=temp;
    GeneralEqmEqnsCell{gg}=GeneralEqmEqns.(GEeqnNames{gg});
end
% Now: 
%  GeneralEqmEqns is still the structure
%  GeneralEqmEqnsCell is cell
%  GeneralEqmEqnParamNames(ff).Names contains the names


%% Set up GEnewprice==3 (if relevant) [More complex than normal as have to allow for transpathoptions.GEptype]
if transpathoptions.GEnewprice==3
    transpathoptions.weightscheme=0;
    
    % Need to make sure that order of rows in transpathoptions.GEnewprice3.howtoupdate
    % Is same as order of fields in GeneralEqmEqns
    % I do this by just reordering rows of transpathoptions.GEnewprice3.howtoupdate
    temp=transpathoptions.GEnewprice3.howtoupdate;
    % GEeqnNames=fieldnames(GeneralEqmEqns);
    gg_c=0;
    for gg=1:length(GEeqnNames)
        for jj=1:size(temp,1)
            if strcmp(temp{jj,1},GEeqnNames{gg}) % Names match
                for ii=1:(1+transpathoptions.GEptype(gg)*(N_i-1)) % Note: 1 or N_i, depending on transpathoptions.GEptype(gg)
                    gg_c=gg_c+1;
                    transpathoptions.GEnewprice3.howtoupdate{gg_c,1}=temp{jj,1};
                    transpathoptions.GEnewprice3.howtoupdate{gg_c,2}=temp{jj,2};
                    transpathoptions.GEnewprice3.howtoupdate{gg_c,3}=temp{jj,3};
                    transpathoptions.GEnewprice3.howtoupdate{gg_c,4}=temp{jj,4};
                end
            end
        end
    end
    % nGeneralEqmEqns=length(GEeqnNames);

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
    if size(transpathoptions.GEnewprice3.howtoupdate,1)==nGeneralEqmEqns_acrossptypes
        % do nothing, this is how things should be
    else
        error('transpathoptions.GEnewprice3.howtoupdate does not fit with GeneralEqmEqns (different number of conditions) \n')
    end
    transpathoptions.GEnewprice3.permute=zeros(size(transpathoptions.GEnewprice3.howtoupdate,1),1);
    for gg=1:size(transpathoptions.GEnewprice3.howtoupdate,1) % number of rows is the number of prices (and number of GE conditions)
        for pp=1:length(PricePathNames)
            if strcmp(transpathoptions.GEnewprice3.howtoupdate{gg,2},PricePathNames{pp})
                transpathoptions.GEnewprice3.permute(gg)=pp;
            end
        end
    end
    if isfield(transpathoptions,'updateaccuracycutoff')==0
        transpathoptions.updateaccuracycutoff=0; % No cut-off (only changes in the price larger in magnitude that this will be made (can be set to, e.g., 10^(-6) to help avoid changes at overly high precision))
    end
end
% Note: permute is a bit odd, but I think I get away with it because the repeated entries are always going to have the same order as the permanenty types so will work out okay


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
        
        % Grab everything relevant out of PTypeStructure
        n_d=PTypeStructure.(iistr).n_d; N_d=PTypeStructure.(iistr).N_d; l_d=PTypeStructure.(iistr).l_d;
        n_a=PTypeStructure.(iistr).n_a; N_a=PTypeStructure.(iistr).N_a;
        n_z=PTypeStructure.(iistr).n_z; N_z=PTypeStructure.(iistr).N_z;
        n_e=PTypeStructure.(iistr).n_e; N_e=PTypeStructure.(iistr).N_e;
        N_j=PTypeStructure.(iistr).N_j;
        d_grid=PTypeStructure.(iistr).d_grid;
        a_grid=PTypeStructure.(iistr).a_grid;
        a_gridvals=PTypeStructure.(iistr).a_gridvals;
        daprime_gridvals=PTypeStructure.(iistr).daprime_gridvals;
        if N_z>0
            z_gridvals_J=PTypeStructure.(iistr).z_gridvals_J;
            pi_z_J=PTypeStructure.(iistr).pi_z_J;
            pi_z_J_sim=PTypeStructure.(iistr).pi_z_J_sim;
            exceptlastj=PTypeStructure.(iistr).exceptlastj;
            exceptfirstj=PTypeStructure.(iistr).exceptfirstj;
            justfirstj=PTypeStructure.(iistr).justfirstj;
        else
            z_gridvals_J=[]; pi_z_J=[]; pi_z_J_sim=[];
            exceptlastj=[]; exceptfirstj=[]; justfirstj=[];
        end
        if N_e>0
            e_gridvals_J=PTypeStructure.(iistr).e_gridvals_J;
            pi_e_J=PTypeStructure.(iistr).pi_e_J;
            pi_e_J_sim=PTypeStructure.(iistr).pi_e_J_sim;
        else
            e_gridvals_J=[]; pi_e_J=[]; pi_e_J_sim=[];
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
        % AgentDist=AgentDist_initial.(iistr);
        % V_final=V_final.(iistr);
        % AgeWeights_T=AgeWeights_T.(iistr);
        % jequalOneDist_T=jequalOneDist_T.(iistr);

        % Some parts of PricePath and ParamPath may depend on ptype
        % Get just the values that correspond to the current ptype
        PricePathOld_ii=PricePathOld(:,PTypeStructure.(iistr).RelevantPricePath);
        ParamPath_ii=ParamPath(:,PTypeStructure.(iistr).RelevantParamPath);
        % PricePathSizeVec_ii, ParamPathSizeVec_ii
        
        % For current ptype, do the backward iteration of V and Policy, then forward iterate agent dist and get the AggVarsPath
        AggVarsPath=TransitionPath_FHorz_PType_singlepath(PricePathOld_ii, ParamPath_ii, PricePathNames,ParamPathNames,T,V_final.(iistr),AgentDist_init.(iistr),jequalOneDist_T.(iistr),AgeWeights_T.(iistr),l_d,N_d,n_d,N_a,n_a,N_z,n_z,N_e,n_e,N_j,d_grid,a_grid,daprime_gridvals,a_gridvals,z_gridvals_J, pi_z_J,pi_z_J_sim,e_gridvals_J,pi_e_J,pi_e_J_sim,ReturnFn, FnsToEvaluate, Parameters, DiscountFactorParamNames, ReturnFnParamNames, FnsToEvaluateParamNames, AggVarNames, PricePathSizeVec_ii, ParamPathSizeVec_ii, use_tminus1price, use_tminus1params, use_tplus1price, use_tminus1AggVars, tminus1priceNames, tminus1paramNames, tplus1priceNames, tminus1AggVarsNames, exceptlastj,exceptfirstj,justfirstj, transpathoptions, vfoptions, simoptions);
        % AggVarsPath=zeros(length(FnsToEvaluate),T-1);

        AggVarsFullPath(PTypeStructure.(iistr).WhichFnsForCurrentPType,:,ii)=AggVarsPath;

    end % done loop over ii
    
    
    %% Note: Cannot do transition paths in which the mass of each agent type changes.
    % AggVarsPooledPath=sum(reshape(PTypeStructure.FnsAndPTypeIndicator,[PTypeStructure.numFnsToEvaluate,1,PTypeStructure.N_i]).*AggVarsFullPath.*shiftdim(AgentDist_init.ptweights,-2),3); % Weighted sum over agent type dimension
    % Note: don't need the above line, as I already dealt with PTypeStructure.FnsAndPTypeIndicator when creating AggVarsFullPath
    AggVarsPooledPath=sum(AggVarsFullPath.*shiftdim(AgentDist_init.ptweights,-2),3); % Weighted sum over agent type dimension

    %% Do the general eqm conditions and create PricePathNew based on these
    if all(transpathoptions.GEptype==0)
        GECondnPath=zeros(T,length(GEeqnNames));
        for tt=1:T-1
            % Parameters that may be relevant to General Eqm
            Parameters=PTypeStructure.ParametersRaw;

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
            % Update current AggVars [we have to add this when doing ptype as GE conditions are in a seperate tt loop to the AggVars]
            for ff=1:length(FullAggVarNames)
                Parameters.(FullAggVarNames{ff})=AggVarsPooledPath(ff,tt);
                % No need for AggVars by ptype here, as the GE conditions do not depend on 
            end

            % Get t+1 PricePath
            if use_tplus1price==1
                for pp=1:length(tplus1priceNames)
                    kk=tplus1pricePathkk(pp);
                    Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePathOld(tt+1,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk)); % Make is so that the time t+1 variables can be used
                end
            end

            if transpathoptions.GEnewprice==1 % The GeneralEqmEqns are not really general eqm eqns, but instead have been given in the form of GEprice updating formulae
                PricePathNew(tt,:)=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters));
            % Note there is no GEnewprice==2, it uses a completely different code
            elseif transpathoptions.GEnewprice==3 % Version of shooting algorithm where the new value is the current value +- fraction*(GECondn)
                p_i=real(GeneralEqmConditions_Case1_v3(GeneralEqmEqnsCell, GeneralEqmEqnParamNames, Parameters));
                p_i=p_i(transpathoptions.GEnewprice3.permute); % Rearrange GeneralEqmEqns into the order of the relevant prices
                I_makescutoff=(abs(p_i)>transpathoptions.updateaccuracycutoff);
                p_i=I_makescutoff.*p_i;
                PricePathNew(tt,:)=(PricePathOld(tt,:).*transpathoptions.GEnewprice3.keepold)+transpathoptions.GEnewprice3.add.*transpathoptions.GEnewprice3.factor.*p_i-(1-transpathoptions.GEnewprice3.add).*transpathoptions.GEnewprice3.factor.*p_i;
                GECondnPath(tt,:)=p_i;
            end
            
        end % Done loop over tt, evaluating the GE conditions
    else % Some GE conditions depend on PType
        GECondnPath=zeros(T,nGeneralEqmEqns_acrossptypes);
        % Some of the General eqm eqns depend on ptype
        for tt=1:T-1
           % Parameters that may be relevant to General Eqm
            Parameters=PTypeStructure.ParametersRaw;
            for ii=1:N_i
                iistr=PTypeStructure.Names_i{ii};
                Parameters_ii.(iistr)=PTypeStructure.(iistr).Parameters; % For use with General Eqm conditions that are evaluated conditional on ptype
            end
            
            % Get t-1 PricePath and ParamPath before we update them
            if use_tminus1price==1
                for pp=1:length(tminus1priceNames)
                    if tt>1
                        Parameters.([tminus1priceNames{pp},'_tminus1'])=Parameters.(tminus1priceNames{pp});
                    else
                        Parameters.([tminus1priceNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1priceNames{pp});
                    end
                    if length(Parameters.([tminus1priceNames{pp},'_tminus1']))==N_i % Depends on ptype
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
                    if tt>1
                        Parameters.([tminus1paramNames{pp},'_tminus1'])=Parameters.(tminus1paramNames{pp});
                    else
                        Parameters.([tminus1paramNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1paramNames{pp});
                    end
                    if length(Parameters.([tminus1paramNames{pp},'_tminus1']))==N_i % Depends on ptype
                        for ii=1:N_i
                            iistr=PTypeStructure.Names_i{ii};
                            tempii=Parameters.([tminus1paramNames{pp},'_tminus1']);
                            Parameters_ii.(iistr).([tminus1paramNames{pp},'_tminus1'])=tempii(ii);
                        end
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
                    if length(Parameters.([tminus1AggVarsNames{pp},'_tminus1']))==N_i % Depends on ptype
                        for ii=1:N_i
                            iistr=PTypeStructure.Names_i{ii};
                            tempii=Parameters.([tminus1AggVarsNames{pp},'_tminus1']);
                            Parameters_ii.(iistr).([tminus1AggVarsNames{pp},'_tminus1'])=tempii(ii);
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
            % Update current AggVars [we have to add this when doing ptype as GE conditions are in a seperate tt loop to the AggVars]
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
                    if length(Parameters.([tplus1priceNames{pp},'_tplus1']))==N_i % Depends on ptype
                        for ii=1:N_i
                            iistr=PTypeStructure.Names_i{ii};
                            tempii=Parameters.([tplus1priceNames{pp},'_tplus1']);
                            Parameters_ii.(iistr).([tplus1priceNames{pp},'_tplus1'])=tempii(ii);
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
                GECondnPath(tt,:)=p_i;
            end

        end % Done loop over tt, evaluating the GE conditions
    end
    
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
        % Do a graph of the GE prices
        figure(1);
        for pp=1:length(PricePathNames)
            if PTypeStructure.PricePath_Idependsonptype(pp)==0
                subplot(nrows_pricepath,ncolumns_pricepath,pp); plot(1:1:T,PricePathOld(:,pp_indexinpricepath(pp)))
            else
                subplot(nrows_pricepath,ncolumns_pricepath,pp); plot(1:1:T,PricePathOld(:,pp_indexinpricepath(pp)))
                hold on
                for ii=2:N_i
                    subplot(nrows_pricepath,ncolumns_pricepath,pp); plot(1:1:T,PricePathOld(:,pp_indexinpricepath(pp)+ii-1))
                end
                hold off
            end
            title(PricePathNames{pp})
        end
    end
    if transpathoptions.graphaggvarspath==1
        % Do a graph of the AggVar paths
        figure(2);
        for aa=1:length(FullAggVarNames)
            subplot(nrows_aggvars,ncolumns_aggvars,aa); plot(1:1:T-1,AggVarsPooledPath(aa,:))
            title(FullAggVarNames{aa})
        end
    end
    if transpathoptions.graphGEconditions==1
        % Do a graph of the General eqm conditions
        figure(3);
        for gg=1:length(GEeqnNames)
            if transpathoptions.GEptype(gg)==0
                subplot(nrows_GEcondns,ncolumns_GEcondns,gg); plot(1:1:T,GECondnPath(:,gg_indexinGEcondns(gg)))
            else
                subplot(nrows_GEcondns,ncolumns_GEcondns,gg); plot(1:1:T,GECondnPath(:,gg_indexinGEcondns(gg)))
                hold on
                for ii=2:N_i
                    subplot(nrows_GEcondns,ncolumns_GEcondns,gg); plot(1:1:T,GECondnPath(:,gg_indexinGEcondns(gg)+ii-1))
                end
                hold off
            end
            title(GEeqnNames{gg})
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
