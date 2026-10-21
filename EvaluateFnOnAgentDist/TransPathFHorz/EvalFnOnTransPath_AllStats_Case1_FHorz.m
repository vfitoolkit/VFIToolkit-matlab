function AllStatsPath=EvalFnOnTransPath_AllStats_Case1_FHorz(FnsToEvaluate, AgentDistPath, PolicyPath, PricePath, ParamPath, Parameters, T, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, transpathoptions, simoptions)
% AllStats (mean, median, variance, standard deviation, Lorenz curve, Gini, quantiles) of the
% FnsToEvaluate in each period of an FHorz transition path. The FHorz counterpart of
% EvalFnOnTransPath_AllStats_InfHorz.
%
% AgentDistPath and PolicyPath are as returned by AgentDistOnTransPath_Case1_FHorz and
% ValueFnOnTransPath_Case1_FHorz: each period's slice (the last dimension is T) is laid out exactly as
% the stationary agent dist and Policy of the same model. So each period is just
% EvalFnOnAgentDist_AllStats_FHorz_Case1 on that period's slice, with the Parameters set to that
% period's prices and parameters, and all the options it takes (simoptions.conditionalrestrictions,
% whichstats, nquantiles, npoints, ...) work as they do there.
%
% Output: AllStatsPath.(FnName).(Stat) is 1-by-T for a scalar stat (Mean, Median, Variance, ...) and
% m-by-T for a vector one (LorenzCurve, QuantileCutoffs, QuantileMeans). With conditional restrictions
% also AllStatsPath.(RestrictionName).RestrictedSampleMass and AllStatsPath.(RestrictionName).(FnName).(Stat).
% Anything a period does not compute (e.g. a restriction with zero mass) is NaN in that column. Text
% fields (LorenzCurveComment, GiniComment, MoreInequality.SharesComment) are 1-by-T cells.
%
% _tminus1 and _tplus1 inputs to the FnsToEvaluate are allowed, as on the transition path itself:
% X_tminus1 is X in period t-1 (transpathoptions.initialvalues.X in period 1), for X a price, a
% ParamPath parameter or one of the FnsToEvaluate (its mean); p_tplus1 is the price in period t+1, and
% period T (the final stationary eqm) in period T.
%
% z and e grids may vary over the transition (a time-varying z_grid or simoptions.e_grid, see
% ExogShockSetup_FHorz_TPath); each period then uses that period's grids.

%% Check which transpathoptions have been used, set all others to defaults
if exist('transpathoptions','var')==0
    transpathoptions.verbose=0;
else
    if ~isfield(transpathoptions,'verbose')
        transpathoptions.verbose=0;
    end
end
transpathoptions.fastOLG=0; % only used for the shock setup below, which is just to learn whether z/e vary over the path

%% Check which simoptions have been used, set all others to defaults (the stationary command does the rest)
if exist('simoptions','var')==0
    simoptions=struct();
end
if ~isfield(simoptions,'n_e')
    simoptions.n_e=0;
end
if ~isfield(simoptions,'n_semiz')
    simoptions.n_semiz=0;
end

%% Internally PricePath is matrix of size T-by-'number of prices'.
% ParamPath is matrix of size T-by-'number of parameters that change over the transition path'.
[PricePath,ParamPath,PricePathNames,ParamPathNames,PricePathSizeVec,ParamPathSizeVec]=PricePathParamPath_FHorz_StructToMatrix(PricePath,ParamPath,N_j,T);

%% Whether z or e vary over the path
% gridpiboth=1: grid only. On a copy of the options, as the stationary command is given the user's own.
% When z (or e) varies over the path, the setup gives transpathoptionsZ.z_gridvals_J_T [N_z,l_z,N_j,T]
% (and e_gridvals_J_T [N_e,l_e,N_j,T], pi_e_J_T [N_e,N_j,T]), and each period hands the stationary
% command that period's age-dependent joint grids. Age jj in period tt then uses the same grid as the
% value fn and agent dist on the path did.
simoptionsZ=simoptions;
simoptionsZ.fastOLG=0;
if ~isfield(simoptionsZ,'parallel')
    simoptionsZ.parallel=1+(gpuDeviceCount>0);
end
if ~isfield(simoptionsZ,'alreadygridvals')
    simoptionsZ.alreadygridvals=0;
end
[~,~,~,~,~,~,~,transpathoptionsZ]=ExogShockSetup_FHorz_TPath(n_z,z_grid,[],prod(n_a),N_j,T,Parameters,PricePathNames,ParamPathNames,transpathoptions,simoptionsZ,1);
z_grid_tt=z_grid;
simoptions_tt=simoptions;

%% Check if using _tminus1 and/or _tplus1 variables.
[tplus1priceNames,tminus1priceNames,tminus1AggVarsNames,tminus1paramNames,tplus1pricePathkk,use_tplus1price,use_tminus1price,use_tminus1params,use_tminus1AggVars]=inputsFindtplus1tminus1(FnsToEvaluate,struct(),PricePathNames,ParamPathNames,{},transpathoptions);

%% Period by period
% Each period's slice keeps the layout of the stationary objects, whatever the shocks are
sliceAgentDist=repmat({':'},1,ndims(AgentDistPath)-1);
slicePolicy=repmat({':'},1,ndims(PolicyPath)-1);

AllStatsPath=struct();
for tt=1:T
    % The _tminus1 values first, while Parameters still holds the previous period's prices and parameters
    % (setting them after period tt is written in would make every _tminus1 equal to period tt)
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
    if use_tminus1AggVars==1
        for pp=1:length(tminus1AggVarsNames)
            if tt>1
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=AllStatsPath.(tminus1AggVarsNames{pp}).Mean(tt-1);
            else
                Parameters.([tminus1AggVarsNames{pp},'_tminus1'])=transpathoptions.initialvalues.(tminus1AggVarsNames{pp});
            end
        end
    end

    for kk=1:length(PricePathNames)
        Parameters.(PricePathNames{kk})=PricePath(tt,PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
    end
    for kk=1:length(ParamPathNames)
        Parameters.(ParamPathNames{kk})=ParamPath(tt,ParamPathSizeVec(1,kk):ParamPathSizeVec(2,kk));
    end

    if use_tplus1price==1
        for pp=1:length(tplus1priceNames)
            kk=tplus1pricePathkk(pp);
            % Period T is the final stationary eqm, so the price after it is the same as at T
            Parameters.([tplus1priceNames{pp},'_tplus1'])=PricePath(min(tt+1,T),PricePathSizeVec(1,kk):PricePathSizeVec(2,kk));
        end
    end

    % This period's z and e grids, when they vary over the path
    if transpathoptionsZ.zpathtrivial==0
        z_grid_tt=transpathoptionsZ.z_gridvals_J_T(:,:,:,tt); % [N_z,l_z,N_j], an age-dependent joint grid
    end
    if transpathoptionsZ.epathtrivial==0
        simoptions_tt.e_grid=transpathoptionsZ.e_gridvals_J_T(:,:,:,tt); % [N_e,l_e,N_j], an age-dependent joint grid
        simoptions_tt.pi_e=transpathoptionsZ.pi_e_J_T(:,:,tt); % [N_e,N_j]
    end

    AllStats=EvalFnOnAgentDist_AllStats_FHorz_Case1(AgentDistPath(sliceAgentDist{:},tt),PolicyPath(slicePolicy{:},tt),FnsToEvaluate,Parameters,[],n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid_tt,simoptions_tt);

    % Store every stat of this period as column tt. AllStats nests to varying depth (a FnsToEvaluate, or a
    % conditional restriction and then a FnsToEvaluate, each of which can hold MoreInequality), so it is
    % walked with a queue of field paths. A numeric field is stored as numel-by-T, created as NaN for every
    % period the first time it is met, so that a period which does not compute it is NaN rather than a
    % padded zero. A text field (e.g. GiniComment, which says why a Gini is not defined) is a 1-by-T cell,
    % holding in each period whatever that period returned.
    queue={{}};
    while ~isempty(queue)
        fieldpath=queue{1};
        queue(1)=[];
        if isempty(fieldpath)
            v=AllStats;
        else
            v=getfield(AllStats,fieldpath{:});
        end
        if isstruct(v)
            vnames=fieldnames(v);
            for kk=1:length(vnames)
                queue{end+1}=[fieldpath,vnames(kk)];
            end
        else
            % Is this field already in AllStatsPath?
            alreadythere=1;
            Stemp=AllStatsPath;
            for kk=1:length(fieldpath)
                if isstruct(Stemp) && isfield(Stemp,fieldpath{kk})
                    Stemp=Stemp.(fieldpath{kk});
                else
                    alreadythere=0;
                    break
                end
            end
            if ischar(v) || isstring(v) || iscell(v) % text: the comments are 1-by-1 cells holding a char
                if alreadythere==0
                    AllStatsPath=setfield(AllStatsPath,fieldpath{:},cell(1,T));
                end
                textpath=getfield(AllStatsPath,fieldpath{:});
                textpath{tt}=v;
                AllStatsPath=setfield(AllStatsPath,fieldpath{:},textpath);
            else
                v=gather(v);
                if alreadythere==0
                    AllStatsPath=setfield(AllStatsPath,fieldpath{:},nan(numel(v),T));
                end
                valuepath=getfield(AllStatsPath,fieldpath{:});
                valuepath(:,tt)=v(:);
                AllStatsPath=setfield(AllStatsPath,fieldpath{:},valuepath);
            end
        end
    end
end

end
