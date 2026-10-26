function AgeConditionalStatsPath=LifeCycleProfiles_TransPath_FHorz_Case1_PType(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, ParamPath, Parameters, T, n_d, n_a, n_z, N_j, Names_i, d_grid, a_grid, z_grid, transpathoptions, simoptions)
% Life-cycle profiles over an FHorz transition path with permanent types. The PType counterpart of
% LifeCycleProfiles_TransPath_FHorz_Case1.
%
% AgentDistPath and PolicyPath are as returned by AgentDistOnTransPath_Case1_FHorz_PType and
% ValueFnOnTransPath_Case1_FHorz_PType: AgentDistPath.(ptype) and PolicyPath.(ptype) have T as their
% last dimension, and each period's slice is laid out exactly as the stationary agent dist and Policy of
% that ptype. AgentDistPath.ptweights is N_i-by-1, or N_i-by-T when the ptype weights change over the path.
% Each period is LifeCycleProfiles_FHorz_Case1_PType on that period's slices, with StationaryDist.ptweights
% the period's weights, and the Parameters of each ptype set to that period's prices and parameters. The
% results are then reorganised by cohort.
%
% Output takes following form, for the stats pooled across ptypes, AgeConditionalStatsPath.(FnName), and
% for those of each ptype, AgeConditionalStatsPath.(FnName).(ptype):
% For 'initialages', for Mean it is a matrix in which first dimension indexes age in period 1 of transtion, second dimension is current age
%                   for LorenzCurve, QuantileCuttoffs, QuantileMeans, these are the second & third dimensions
% For 'bornduringtranstion', for Mean it is a matrix in which first dimension indexes period of transition in which born (age period 1), second dimension is current age
% (Mean, Median, Variance, LorenzCurve, Gini, QuantileCutoffs, QuantileMeans, as LifeCycleProfiles_TransPath_FHorz_Case1)
%
% Prices and parameters on the path can differ by ptype, given as a structure with a field for each
% ptype (or as N_i-by-T/T-by-N_i). A ParamPath structure may cover just some of the ptypes: the others
% do not have that parameter on the path.
%
% _tminus1 and _tplus1 inputs to the FnsToEvaluate are allowed, as on the transition path itself:
% X_tminus1 is X in period t-1 (transpathoptions.initialvalues.X in period 1, which may be by ptype, as a
% structure or a vector of length N_i), for X a price, a ParamPath parameter or one of the FnsToEvaluate
% (its mean over all ages, conditional on the ptype); p_tplus1 is the price in period t+1, and period T
% (the final stationary eqm) in period T.
%
% z and e grids may vary over the transition, differently for each ptype (a time-varying z_grid or
% simoptions.e_grid, see ExogShockSetup_FHorz_TPath); each period then uses that period's grids.

%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    transpathoptions.verbose=0;
else
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end

%% Check which simoptions have been used, set all others to defaults (the stationary command does the rest)
if exist('simoptions','var')==0
    simoptions=struct();
end

%%
if iscell(Names_i)
    N_i=length(Names_i);
else
    N_i=Names_i; % It is the number of PTypes (which have not been given names)
    Names_i={'ptype001'};
    for ii=2:N_i
        if ii<10
            Names_i{ii}=['ptype00',num2str(ii)];
        elseif ii<100
            Names_i{ii}=['ptype0',num2str(ii)];
        elseif ii<1000
            Names_i{ii}=['ptype',num2str(ii)];
        end
    end
end

%% Set up each ptype: its parameters, its price and parameter paths, and its (possibly time-varying) z and e grids
% Everything a period needs is kept by ptype, and each period hands the stationary PType command
% structures with a field for every ptype (so the stationary command reads each ptype's value straight off,
% whatever form the user gave them in).
Parameters_ii=struct(); % Parameters_ii.(iistr) is the Parameters of ptype iistr, updated period by period
PricePathMatrix_ii=struct();
PricePathNames_ii=struct();
PricePathSizeVec_ii=struct();
ParamPathMatrix_ii=struct();
ParamPathNames_ii=struct();
ParamPathSizeVec_ii=struct();
z_grid_ii=struct(); % the (fixed) z_grid of each ptype
e_grid_ii=struct(); % the (fixed) e_grid and pi_e of each ptype
pi_e_ii=struct();
N_e_ii=zeros(N_i,1);
transpathoptionsZ_ii=struct();
lagsinfo_ii=struct(); % the _tminus1 and _tplus1 names of each ptype
for ii=1:N_i
    iistr=Names_i{ii};

    simoptions_temp=PType_Options(simoptions,iistr);
    [n_d_temp,n_a_temp,~,~]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid);
    l_a_temp=length(n_a_temp);
    if n_d_temp(1)==0
        l_d_temp=0;
    else
        l_d_temp=length(n_d_temp);
    end
    if isstruct(N_j)
        N_j_temp=N_j.(iistr);
    else
        N_j_temp=N_j;
    end
    [n_z_temp,z_grid_temp,~,simoptions_temp]=PType_setup_ExogShocks(ii,iistr,N_i,n_z,z_grid,[],simoptions_temp,3);
    if n_z_temp(1)==0
        l_z_temp=0;
    else
        l_z_temp=length(n_z_temp);
    end
    z_grid_ii.(iistr)=z_grid_temp;
    N_e_ii(ii)=prod(simoptions_temp.n_e);
    if N_e_ii(ii)>0 && isfield(simoptions_temp,'e_grid')
        e_grid_ii.(iistr)=simoptions_temp.e_grid;
        pi_e_ii.(iistr)=simoptions_temp.pi_e;
    end

    Parameters_ii.(iistr)=PType_setup_Parameters(ii,iistr,N_i,Parameters,3);

    % PricePath can include prices that differ by ptype
    PricePath_temp=PricePath;
    PricePathNames=fieldnames(PricePath);
    for nn=1:length(PricePathNames)
        if isstruct(PricePath_temp.(PricePathNames{nn}))
            PricePath_temp.(PricePathNames{nn})=PricePath.(PricePathNames{nn}).(iistr);
        elseif any(size(PricePath_temp.(PricePathNames{nn}))==N_i)
            if size(PricePath_temp.(PricePathNames{nn}),1)==N_i
                temp=PricePath_temp.(PricePathNames{nn});
                PricePath_temp.(PricePathNames{nn})=temp(ii,:);
            elseif size(PricePath_temp.(PricePathNames{nn}),2)==N_i
                temp=PricePath_temp.(PricePathNames{nn});
                PricePath_temp.(PricePathNames{nn})=temp(:,ii);
            end
        end
    end
    % ParamPath can include parameters that differ by ptype; a structure that has no field for this
    % ptype means the parameter is not on the path for this ptype
    ParamPath_temp=ParamPath;
    ParamPathNames=fieldnames(ParamPath);
    for nn=1:length(ParamPathNames)
        if isstruct(ParamPath_temp.(ParamPathNames{nn}))
            if isfield(ParamPath.(ParamPathNames{nn}),iistr)
                ParamPath_temp.(ParamPathNames{nn})=ParamPath.(ParamPathNames{nn}).(iistr);
            else
                ParamPath_temp=rmfield(ParamPath_temp,ParamPathNames{nn});
            end
        elseif any(size(ParamPath_temp.(ParamPathNames{nn}))==N_i)
            if size(ParamPath_temp.(ParamPathNames{nn}),1)==N_i
                temp=ParamPath_temp.(ParamPathNames{nn});
                ParamPath_temp.(ParamPathNames{nn})=temp(ii,:);
            elseif size(ParamPath_temp.(ParamPathNames{nn}),2)==N_i
                temp=ParamPath_temp.(ParamPathNames{nn});
                ParamPath_temp.(ParamPathNames{nn})=temp(:,ii);
            end
        end
    end
    [PricePathMatrix_temp,ParamPathMatrix_temp,PricePathNames_temp,ParamPathNames_temp,PricePathSizeVec_temp,ParamPathSizeVec_temp]=PricePathParamPath_FHorz_StructToMatrix(PricePath_temp,ParamPath_temp,N_j_temp,T);
    PricePathMatrix_ii.(iistr)=PricePathMatrix_temp;
    PricePathNames_ii.(iistr)=PricePathNames_temp;
    PricePathSizeVec_ii.(iistr)=PricePathSizeVec_temp;
    ParamPathMatrix_ii.(iistr)=ParamPathMatrix_temp;
    ParamPathNames_ii.(iistr)=ParamPathNames_temp;
    ParamPathSizeVec_ii.(iistr)=ParamPathSizeVec_temp;

    % Whether z or e vary over the path for this ptype
    % gridpiboth=1: grid only. On a copy of the options, as the stationary command is given the user's own.
    % When z (or e) varies over the path, the setup gives z_gridvals_J_T [N_z,l_z,N_j,T] (and
    % e_gridvals_J_T [N_e,l_e,N_j,T], pi_e_J_T [N_e,N_j,T]), and each period hands the stationary command
    % that period's age-dependent joint grids. Age jj in period tt then uses the same grid as the value fn
    % and agent dist on the path did.
    transpathoptionsZ=transpathoptions;
    transpathoptionsZ.fastOLG=0;
    simoptionsZ=simoptions_temp;
    simoptionsZ.fastOLG=0;
    if ~isfield(simoptionsZ,'parallel')
        simoptionsZ.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptionsZ,'alreadygridvals')
        simoptionsZ.alreadygridvals=0;
    end
    [~,~,~,~,~,~,~,transpathoptionsZ]=ExogShockSetup_FHorz_TPath(n_z_temp,z_grid_temp,[],prod(n_a_temp),N_j_temp,T,Parameters_ii.(iistr),PricePathNames_temp,ParamPathMatrix_temp,ParamPathNames_temp,ParamPathSizeVec_temp,transpathoptionsZ,simoptionsZ,1);
    transpathoptionsZ_ii.(iistr)=transpathoptionsZ;

    % _tminus1 and _tplus1 of this ptype
    [FnsToEvaluate_temp,~,~,~]=PType_FnsToEvaluate(FnsToEvaluate,Names_i,ii,l_d_temp,l_a_temp,l_z_temp,0);
    [tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate_temp,struct(),PricePathNames_temp,ParamPathNames_temp,{},transpathoptions);
    lagsinfo_ii.(iistr).tplus1priceNames=tplus1priceNames;
    lagsinfo_ii.(iistr).tminus1priceNames=tminus1priceNames;
    lagsinfo_ii.(iistr).tminus1AggVarsNames=tminus1AggVarsNames;
    lagsinfo_ii.(iistr).tminus1paramNames=tminus1paramNames;
    lagsinfo_ii.(iistr).tplus1pricePathkk=tplus1pricePathkk;
    lagsinfo_ii.(iistr).use_tplus1price=use_tplus1price;
    lagsinfo_ii.(iistr).use_tminus1price=use_tminus1price;
    lagsinfo_ii.(iistr).use_tminus1params=use_tminus1params;
    lagsinfo_ii.(iistr).use_tminus1AggVars=use_tminus1AggVars;
end

%% Life-cycle profiles are conditional on each age, and the output is reorganised by cohort
if ~isfield(simoptions,'nquantiles')
    simoptions.nquantiles=20; % by default gives ventiles
end
if ~isfield(simoptions,'agegroupings')
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately
end
if ~isfield(simoptions,'npoints')
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
end
ngroups=length(simoptions.agegroupings);
if ngroups~=N_j
    error('LifeCycleProfiles_TransPath_FHorz_Case1_PType only permits using agegroupings of an individual model period (was the only obvious use as otherwise the grouping composition changes, if you envisage a use that is not this, let me know)')
end
AggVarNames=fieldnames(FnsToEvaluate);

OutputTypes={'Mean','Median','Variance','LorenzCurve','Gini','QuantileCutoffs','QuantileMeans'};

% The FnsToEvaluate used as _tminus1 by any ptype, and their means (over all ages) conditional on each
% ptype in the previous period. The life-cycle profiles are conditional on age, so do not give these, and
% they are computed each period from the same agent dist and policy.
tminus1AggVarsNamesAll={};
for ii=1:N_i
    tminus1AggVarsNamesAll=union(tminus1AggVarsNamesAll,lagsinfo_ii.(Names_i{ii}).tminus1AggVarsNames);
end
AggVarsLag_ii=struct();

AgeConditionalStatsPath=struct();

%% Period by period
for tt=1:T
    StationaryDist_tt=struct();
    Policy_tt=struct();
    z_grid_tt=struct();
    simoptions_tt=simoptions;
    anyevaries=0;
    for ii=1:N_i
        iistr=Names_i{ii};
        lagsinfo=lagsinfo_ii.(iistr);
        Parameters_temp=Parameters_ii.(iistr);

        % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
        % (setting them after period tt is written in would make every _tminus1 equal to period tt).
        % In period 1 they are the initialvalues, which may differ by ptype (a structure, or a vector of length N_i).
        if lagsinfo.use_tminus1price==1
            for pp=1:length(lagsinfo.tminus1priceNames)
                if tt>1
                    Parameters_temp.([lagsinfo.tminus1priceNames{pp},'_tminus1'])=Parameters_temp.(lagsinfo.tminus1priceNames{pp});
                else
                    temp=transpathoptions.initialvalues.(lagsinfo.tminus1priceNames{pp});
                    if isstruct(temp)
                        Parameters_temp.([lagsinfo.tminus1priceNames{pp},'_tminus1'])=temp.(iistr);
                    elseif length(temp)==N_i && N_i>1
                        Parameters_temp.([lagsinfo.tminus1priceNames{pp},'_tminus1'])=temp(ii);
                    else
                        Parameters_temp.([lagsinfo.tminus1priceNames{pp},'_tminus1'])=temp;
                    end
                end
            end
        end
        if lagsinfo.use_tminus1params==1
            for pp=1:length(lagsinfo.tminus1paramNames)
                if tt>1
                    Parameters_temp.([lagsinfo.tminus1paramNames{pp},'_tminus1'])=Parameters_temp.(lagsinfo.tminus1paramNames{pp});
                else
                    temp=transpathoptions.initialvalues.(lagsinfo.tminus1paramNames{pp});
                    if isstruct(temp)
                        Parameters_temp.([lagsinfo.tminus1paramNames{pp},'_tminus1'])=temp.(iistr);
                    elseif length(temp)==N_i && N_i>1
                        Parameters_temp.([lagsinfo.tminus1paramNames{pp},'_tminus1'])=temp(ii);
                    else
                        Parameters_temp.([lagsinfo.tminus1paramNames{pp},'_tminus1'])=temp;
                    end
                end
            end
        end
        if lagsinfo.use_tminus1AggVars==1
            for pp=1:length(lagsinfo.tminus1AggVarsNames)
                if tt>1
                    % The mean conditional on this ptype over all ages, last period (as EvalFnOnTransPath_AggVars_Case1_FHorz_PType)
                    Parameters_temp.([lagsinfo.tminus1AggVarsNames{pp},'_tminus1'])=AggVarsLag_ii.(iistr).(lagsinfo.tminus1AggVarsNames{pp});
                else
                    temp=transpathoptions.initialvalues.(lagsinfo.tminus1AggVarsNames{pp});
                    if isstruct(temp)
                        Parameters_temp.([lagsinfo.tminus1AggVarsNames{pp},'_tminus1'])=temp.(iistr);
                    elseif length(temp)==N_i && N_i>1
                        Parameters_temp.([lagsinfo.tminus1AggVarsNames{pp},'_tminus1'])=temp(ii);
                    else
                        Parameters_temp.([lagsinfo.tminus1AggVarsNames{pp},'_tminus1'])=temp;
                    end
                end
            end
        end

        PricePathMatrix_temp=PricePathMatrix_ii.(iistr);
        PricePathNames_temp=PricePathNames_ii.(iistr);
        PricePathSizeVec_temp=PricePathSizeVec_ii.(iistr);
        for kk=1:length(PricePathNames_temp)
            Parameters_temp.(PricePathNames_temp{kk})=PricePathMatrix_temp(tt,PricePathSizeVec_temp(1,kk):PricePathSizeVec_temp(2,kk));
        end
        ParamPathNames_temp=ParamPathNames_ii.(iistr);
        ParamPathSizeVec_temp=ParamPathSizeVec_ii.(iistr);
        for kk=1:length(ParamPathNames_temp)
            Parameters_temp.(ParamPathNames_temp{kk})=ParamPathMatrix_ii.(iistr)(tt,ParamPathSizeVec_temp(1,kk):ParamPathSizeVec_temp(2,kk));
        end

        if lagsinfo.use_tplus1price==1
            for pp=1:length(lagsinfo.tplus1priceNames)
                kk=lagsinfo.tplus1pricePathkk(pp);
                % Period T is the final stationary eqm, so the price after it is the same as at T
                Parameters_temp.([lagsinfo.tplus1priceNames{pp},'_tplus1'])=PricePathMatrix_temp(min(tt+1,T),PricePathSizeVec_temp(1,kk):PricePathSizeVec_temp(2,kk));
            end
        end
        Parameters_ii.(iistr)=Parameters_temp;

        % This period's agent dist and policy
        sliceAgentDist=repmat({':'},1,ndims(AgentDistPath.(iistr))-1);
        slicePolicy=repmat({':'},1,ndims(PolicyPath.(iistr))-1);
        StationaryDist_tt.(iistr)=AgentDistPath.(iistr)(sliceAgentDist{:},tt);
        Policy_tt.(iistr)=PolicyPath.(iistr)(slicePolicy{:},tt);

        % This period's z and e grids, when they vary over the path
        if transpathoptionsZ_ii.(iistr).zpathtrivial==0
            z_grid_tt.(iistr)=transpathoptionsZ_ii.(iistr).z_gridvals_J_T(:,:,:,tt); % [N_z,l_z,N_j], an age-dependent joint grid
        else
            z_grid_tt.(iistr)=z_grid_ii.(iistr);
        end
        if transpathoptionsZ_ii.(iistr).epathtrivial==0
            anyevaries=1;
        end
    end
    % The ptype weights of this period
    if size(AgentDistPath.ptweights,2)==1
        StationaryDist_tt.ptweights=AgentDistPath.ptweights;
    else
        StationaryDist_tt.ptweights=AgentDistPath.ptweights(:,tt);
    end
    % e_grid and pi_e by ptype, when they vary over the path for any ptype
    if anyevaries==1
        e_grid_tt=struct();
        pi_e_tt=struct();
        for ii=1:N_i
            iistr=Names_i{ii};
            if N_e_ii(ii)>0
                if transpathoptionsZ_ii.(iistr).epathtrivial==0
                    e_grid_tt.(iistr)=transpathoptionsZ_ii.(iistr).e_gridvals_J_T(:,:,:,tt); % [N_e,l_e,N_j], an age-dependent joint grid
                    pi_e_tt.(iistr)=transpathoptionsZ_ii.(iistr).pi_e_J_T(:,:,tt); % [N_e,N_j]
                else
                    e_grid_tt.(iistr)=e_grid_ii.(iistr);
                    pi_e_tt.(iistr)=pi_e_ii.(iistr);
                end
            end
        end
        simoptions_tt.e_grid=e_grid_tt;
        simoptions_tt.pi_e=pi_e_tt;
    end
    % Parameters with a field for every ptype
    Parameters_tt=struct();
    for ii=1:N_i
        iistr=Names_i{ii};
        ParamNames_temp=fieldnames(Parameters_ii.(iistr));
        for pp=1:length(ParamNames_temp)
            Parameters_tt.(ParamNames_temp{pp}).(iistr)=Parameters_ii.(iistr).(ParamNames_temp{pp});
        end
    end

    tempAgeConditionalStats=LifeCycleProfiles_FHorz_Case1_PType(StationaryDist_tt,Policy_tt,FnsToEvaluate,Parameters_tt,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid_tt,simoptions_tt);

    % This period's aggregates (over all ages, conditional on ptype) of the FnsToEvaluate that are used
    % as _tminus1, which become the _tminus1 values of next period
    if ~isempty(tminus1AggVarsNamesAll)
        FnsToEvaluateLag=struct();
        for pp=1:length(tminus1AggVarsNamesAll)
            FnsToEvaluateLag.(tminus1AggVarsNamesAll{pp})=FnsToEvaluate.(tminus1AggVarsNamesAll{pp});
        end
        simoptionsLag=simoptions_tt;
        if isfield(simoptionsLag,'outputasstructure')
            simoptionsLag=rmfield(simoptionsLag,'outputasstructure'); % FnsToEvaluateLag is a structure, so the output is too
        end
        AggVarsLag=EvalFnOnAgentDist_AggVars_FHorz_Case1_PType(StationaryDist_tt,Policy_tt,FnsToEvaluateLag,Parameters_tt,n_d,n_a,n_z,N_j,Names_i,d_grid,a_grid,z_grid_tt,simoptionsLag);
        for ii=1:N_i
            iistr=Names_i{ii};
            for pp=1:length(lagsinfo_ii.(iistr).tminus1AggVarsNames)
                AggVarsLag_ii.(iistr).(lagsinfo_ii.(iistr).tminus1AggVarsNames{pp})=gather(AggVarsLag.(lagsinfo_ii.(iistr).tminus1AggVarsNames{pp}).(iistr).Mean);
            end
        end
    end

    % The stats pooled across ptypes (groupname empty), and those of each ptype that has them
    for ff=1:length(AggVarNames)
        groupnames={''};
        for ii=1:N_i
            if isfield(tempAgeConditionalStats.(AggVarNames{ff}),Names_i{ii})
                groupnames{end+1}=Names_i{ii};
            end
        end
        for gg=1:length(groupnames)
            if isempty(groupnames{gg})
                tempStats=tempAgeConditionalStats.(AggVarNames{ff});
            else
                tempStats=tempAgeConditionalStats.(AggVarNames{ff}).(groupnames{gg});
            end
            if tt==1 % Preallocate (as NaN, so cohorts that are not alive stay NaN)
                for oo=1:length(OutputTypes)
                    if any(oo==[1,2,3,5]) % Mean, Median, Variance, Gini
                        initialages_oo=nan(ngroups,ngroups); % Note: will be triangular (rest remains nan): first dimension is initial age (during period 1 of transtion, so age at reform), second dimension is their age
                        bornduringtranstion_oo=nan(T,ngroups); % First dimensions is the transition period during which they were born, second dimension is their age
                    elseif oo==4 % Lorenz curve
                        initialages_oo=nan(simoptions.npoints,ngroups,ngroups);
                        bornduringtranstion_oo=nan(simoptions.npoints,T,ngroups);
                    elseif oo==6 % Quantile cutoffs
                        initialages_oo=nan(simoptions.nquantiles+1,ngroups,ngroups);
                        bornduringtranstion_oo=nan(simoptions.nquantiles+1,T,ngroups);
                    elseif oo==7 % Quantile Means
                        initialages_oo=nan(simoptions.nquantiles,ngroups,ngroups);
                        bornduringtranstion_oo=nan(simoptions.nquantiles,T,ngroups);
                    end
                    if isempty(groupnames{gg})
                        AgeConditionalStatsPath.(AggVarNames{ff}).initialages.(OutputTypes{oo})=initialages_oo;
                        AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion.(OutputTypes{oo})=bornduringtranstion_oo;
                    else
                        AgeConditionalStatsPath.(AggVarNames{ff}).(groupnames{gg}).initialages.(OutputTypes{oo})=initialages_oo;
                        AgeConditionalStatsPath.(AggVarNames{ff}).(groupnames{gg}).bornduringtranstion.(OutputTypes{oo})=bornduringtranstion_oo;
                    end
                end
            end
            if isempty(groupnames{gg})
                StatsPath=AgeConditionalStatsPath.(AggVarNames{ff});
            else
                StatsPath=AgeConditionalStatsPath.(AggVarNames{ff}).(groupnames{gg});
            end
            for oo=1:length(OutputTypes)
                temp=gather(tempStats.(OutputTypes{oo}));
                % First, sort out the initial ages
                if tt<=ngroups
                    temp2=StatsPath.initialages.(OutputTypes{oo}); % Note: will be triangular (rest remains nan)
                    % Age at time of reform (age in period when the ParamPath is revealed). Those aged jj in the first
                    % period are aged jj+tt-1 now, so are still alive for jj up to ngroups-tt+1
                    for jj=1:ngroups-tt+1
                        if any(oo==[1,2,3,5])
                            % Age jj in first period, are now aged jj+tt-1
                            temp2(jj,jj+tt-1)=temp(jj+tt-1); % Mean, Median, Variance, Gini
                        else
                            temp2(:,jj,jj+tt-1)=temp(:,jj+tt-1); % LorenzCurve, QuantileCutoffs, QuantileMean
                        end
                    end
                    StatsPath.initialages.(OutputTypes{oo})=temp2;
                end
                % Now, those born in the transition
                temp3=StatsPath.bornduringtranstion.(OutputTypes{oo});
                for bb=max(1,tt-N_j+1):tt
                    if any(oo==[1,2,3,5])
                        % Born in bb, are now aged tt-bb+1
                        temp3(bb,tt-bb+1)=temp(tt-bb+1); % Mean, Median, Variance, Gini
                    else
                        temp3(:,bb,tt-bb+1)=temp(:,tt-bb+1); % LorenzCurve, QuantileCutoffs, QuantileMean
                    end
                end
                StatsPath.bornduringtranstion.(OutputTypes{oo})=temp3;
            end
            if isempty(groupnames{gg})
                % (keeps the per-ptype substructures already stored under this FnsToEvaluate)
                AgeConditionalStatsPath.(AggVarNames{ff}).initialages=StatsPath.initialages;
                AgeConditionalStatsPath.(AggVarNames{ff}).bornduringtranstion=StatsPath.bornduringtranstion;
            else
                AgeConditionalStatsPath.(AggVarNames{ff}).(groupnames{gg})=StatsPath;
            end
        end
    end
end

end
