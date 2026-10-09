function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% Similar to SimLifeCycleProfiles but works from StationaryDist rather than
% simulating panel data. Where applicable it is faster and more accurate.
% options.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., options.agegroupings=1:10:N_j will divide into 10 year age bins and calculate stats for each of them
% simoptions.whichcombos ([numFnsToEvaluate, number of age groups, 1+number of conditional restrictions]) selects which combinations are
% computed, and simoptions.whichstats may be given per combination ([.., .., .., 7]); see the whichcombos section below.
% options.npoints can be used to determine how many points are used for the lorenz curve
% options.nquantiles can be used to change from reporting (age conditional) ventiles, to quartiles/deciles/percentiles/etc.
%
% Note that the quantile are what are typically reported as life-cycle profiles (or more precisely, the quantile cutoffs).
%
% Output takes following form
% ngroups=length(options.agegroupings);
% AgeConditionalStats(length(FnsToEvaluate)).Mean=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Median=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Variance=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).LorenzCurve=nan(options.npoints,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).Gini=nan(1,ngroups);
% AgeConditionalStats(length(FnsToEvaluate)).QuantileCutoffs=nan(options.nquantiles+1,ngroups); % Includes the min and max values
% AgeConditionalStats(length(FnsToEvaluate)).QuantileMeans=nan(options.nquantiles,ngroups);

%% Temporary warning due to changing function input order
if isempty(Parameters)
    warning('LifeCycleProfiles_FHorz_Case1 has changed the order of the fourth and fifth inputs (should now be something like Params,[] when previously it would have been [],Params)')
    warning('Annoying, but it makes inputs to LifeCycleProfiles_FHorz_Case1 have same order as those of similar functions')
end

%% Check which simoptions have been declared, set all others to defaults
if ~exist('simoptions','var')
    %If options is not given, just use all the defaults
    if isgpuarray(StationaryDist)
        simoptions.parallel=2;
    else
        simoptions.parallel=1;
    end
    simoptions.verbose=0;
    simoptions.nquantiles=20; % by default gives ventiles
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, 7]
    % Model Setup
    simoptions.gridinterplayer=0;
    simoptions.n_semiz=0;
    simoptions.n_e=0;
    simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    % When calling as a subcommand, the following is used internally
    simoptions.alreadygridvals=0;
    simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
else
    %Check options for missing fields, if there are some fill them with the defaults
    if isgpuarray(StationaryDist) % simoptions.parallel is overwritten based on StationaryDist
        simoptions.parallel=2;
    else
        simoptions.parallel=1;
    end
    if ~isfield(simoptions,'verbose')
        simoptions.verbose=0;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if ~isfield(simoptions,'agegroupings')
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if ~isfield(simoptions,'tolerance')
        simoptions.tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.
    end
    if isfield(simoptions,'SampleRestrictionFn') % If using SampleRestrictionFn then need to set some things
        if ~isfield(simoptions,'SampleRestrictionFn_include')
            simoptions.SampleRestrictionFn_include=1; % By default, include observations that meet the sample restriction (if zero, then exclude observations meeting this criterion)
        end
        simoptions.SampleRestrictionFnParamNames=getAnonymousFnInputNames(simoptions.SampleRestrictionFn); % Note: we remove those relating to the state space later
    end
    if ~isfield(simoptions,'whichstats')
        if any(simoptions.agegroupings(2:end)-simoptions.agegroupings(1:end-1)>4)
            % if some agegroupings are 'large', use the slower but lower memory versions
            simoptions.whichstats=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, 7]
        else
            simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes % whichstats here is [1,7]; it can be up to [nFns, nAgeGroups, 1+nRestr, 7]
        end
    end
    % Model Setup
    if ~isfield(simoptions,'gridinterplayer')
        simoptions.gridinterplayer=0;
    end
    if ~isfield(simoptions,'n_semiz')
        simoptions.n_semiz=0;
    end
    if ~isfield(simoptions,'n_e')
        simoptions.n_e=0;
    end
    if ~isfield(simoptions,'warnzerorestrictedmass')
        simoptions.warnzerorestrictedmass=2; % =2 (default) warns when a conditional restriction has zero mass; =1 silences the per-type warnings but the PType commands still warn once at the population level; =0 silences both (the calibration/estimation commands default to 0)
    end
    % When calling as a subcommand, the following is used internally
    if ~isfield(simoptions,'alreadygridvals')
        simoptions.alreadygridvals=0;
    end
    if ~isfield(simoptions,'alreadygridvals_semiexo')
        simoptions.alreadygridvals_semiexo=0; % =1 when calling as a subcommand
    end
end

% N_d=prod(n_d);
N_a=prod(n_a);
% N_z=prod(n_z);

l_a=length(n_a);

ngroups=length(simoptions.agegroupings);

if isstruct(FnsToEvaluate)
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    numFnsToEvaluate=length(FnsToEvalNames);
else
    numFnsToEvaluate=length(FnsToEvaluate);
end


%%
if simoptions.parallel==1
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_cpu(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
    return
end
% just make sure things are on gpu as they should be
StationaryDist=gpuArray(StationaryDist);
Policy=gpuArray(Policy);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);


%% I want to do some things now, so that they can be used in setting up conditional restrictions

a_gridvals=CreateGridvals(n_a,a_grid,1);
if N_z==0
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,1,3]); % (N_a,l_daprime,N_j)
else
    StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(Policy,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_z,l_daprime,N_j)
end

% Figure out l_daprime from PolicyValues
l_daprime=size(PolicyValues,1);


%% Implement new way of handling FnsToEvaluate

% Note: l_z includes e and semiz (when appropriate)
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    % FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_daprime+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end
if isfield(simoptions,'keepoutputasmatrix')
    if simoptions.keepoutputasmatrix==1
        FnsToEvaluateStruct=0;
    end
end

%% simoptions.whichcombos and per-combination simoptions.whichstats
% whichcombos: zeros/ones of size [numFnsToEvaluate, number of age groups] without conditional restrictions, and
% [numFnsToEvaluate, number of age groups, 1+number of restrictions] with them: page 1 is the unrestricted stats, pages 2:end the
% restrictions in the fieldnames order of simoptions.conditionalrestrictions. Ones are computed, zeros skipped; the output has the
% same fields either way and the skipped entries stay NaN; RestrictedSampleMass is always filled. Default all ones. A 2D input with
% restrictions is applied to every page. Intended for calibration/estimation, which only needs the targeted combinations.
% whichstats: the usual 1x7 vector, or [numFnsToEvaluate, number of age groups, 1+number of restrictions, 7] giving a whichstats
% vector for every combination ([numFnsToEvaluate, number of age groups, 7] is applied to every page). A field exists for a
% (fn, page) if any of its age groups asks for that statistic. A combination asking for no statistic is skipped like a whichcombos zero.
if isfield(simoptions,'conditionalrestrictions')
    nwhichpages=1+length(fieldnames(simoptions.conditionalrestrictions));
else
    nwhichpages=1;
end
if ~isfield(simoptions,'whichcombos')
    whichcombos=ones(numFnsToEvaluate,ngroups,nwhichpages); % whichcombos here is [nFns, nAgeGroups, 1+nRestr]
else
    whichcombos=simoptions.whichcombos;
    if ~(isnumeric(whichcombos) || islogical(whichcombos)) || any(whichcombos(:)~=0 & whichcombos(:)~=1)
        error('simoptions.whichcombos must contain only zeros and ones')
    end
    if ismatrix(whichcombos) && nwhichpages>1 && isequal(size(whichcombos),[numFnsToEvaluate,ngroups])
        whichcombos=repmat(whichcombos,[1,1,nwhichpages]); % 2D input with restrictions: apply to every page
    end
    if ~isequal(size(whichcombos,1:3),[numFnsToEvaluate,ngroups,nwhichpages])
        error(['simoptions.whichcombos must be of size [',num2str(numFnsToEvaluate),',',num2str(ngroups),',',num2str(nwhichpages),'] (number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions; the third dimension is dropped when there are no conditional restrictions)'])
    end
    whichcombos=double(whichcombos);
end
wsG=simoptions.whichstats;
if isvector(wsG) && numel(wsG)==7
    whichstatsArr=repmat(reshape(wsG,[1,1,1,7]),[numFnsToEvaluate,ngroups,nwhichpages,1]);
elseif ndims(wsG)==3 && isequal(size(wsG),[numFnsToEvaluate,ngroups,7])
    whichstatsArr=repmat(reshape(wsG,[numFnsToEvaluate,ngroups,1,7]),[1,1,nwhichpages,1]);
elseif isequal(size(wsG,1:4),[numFnsToEvaluate,ngroups,nwhichpages,7])
    whichstatsArr=wsG;
else
    error(['simoptions.whichstats must be a 1x7 vector, or of size [',num2str(numFnsToEvaluate),',',num2str(ngroups),',',num2str(nwhichpages),',7] (number of FnsToEvaluate, number of age groups, 1+number of conditional restrictions, 7), or [',num2str(numFnsToEvaluate),',',num2str(ngroups),',7]'])
end
whichcombos=whichcombos.*any(whichstatsArr,4); % a combination with no statistic requested is skipped altogether


% Preallocate various things for the stats (as many will have jj as a dimension)
% Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
for ff=1:numFnsToEvaluate
    % whichstats for this fn across its age groups: a statistic is preallocated if any age group asks for it (the Lorenz curve if any age group has 1<=whichstats(4)<3)
    wsP=max(reshape(whichstatsArr(ff,:,1,:),[ngroups,7]),[],1);
    ws4=reshape(whichstatsArr(ff,:,1,4),[1,ngroups]);
    if any(ws4>=1 & ws4<3)
        wsP(4)=1;
    end
    if wsP(1)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(2)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(1)==1 && wsP(2)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(3)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(4)>=1
        AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
        if wsP(4)<3
            AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
        end
    end
    if wsP(5)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(6)>=1
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
    if wsP(7)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
end

%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it).
if isfield(simoptions,'conditionalrestrictions')
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),N_j);
    RestrictionMask=cell(length(CondlRestnFnNames),1); % each restriction kept as a logical mask over the (a,z) x j grid (1 byte per point); the restricted weights are formed from it where used

    % For each conditional restriction, create a 'restricted stationary distribution'
    for rr=1:length(CondlRestnFnNames)
        % The current conditional restriction function
        CondlRestnFn=simoptions.conditionalrestrictions.(CondlRestnFnNames{rr});
        % Get parameter names for Conditional Restriction functions
        temp=getAnonymousFnInputNames(CondlRestnFn);
        if length(temp)>(l_daprime+l_a+l_z)
            CondlRestnFnParamNames={temp{l_daprime+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames={};
        end

        if N_z==0
            RestrictionValues=zeros(N_a,N_j);
        else
            RestrictionValues=zeros(N_a,N_z,N_j);
        end
        for jj=1:N_j % Given the actual stats have to loop over j, I just do it here even though it could be done with EvalFnOnAgentDist_Grid_J instead
            % Get parameter values for Conditional Restriction functions
            CondlRestnFnParamsCell=CreateCellFromParams(Parameters,CondlRestnFnParamNames,jj);

            % Compute the restrictions
            if N_z==0
                RestrictionValues(:,jj)=logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,n_z,a_gridvals,[]));
            else
                RestrictionValues(:,:,jj)=logical(EvalFnOnAgentDist_Grid(CondlRestnFn,CondlRestnFnParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj)));
            end
        end
        if N_z>0
            RestrictionValues=reshape(RestrictionValues,[N_a*N_z,N_j]);
        end


        % Keep the restricted mass at each age, and the mask
        restrictedsamplemass(rr,:)=sum(StationaryDist.*RestrictionValues,1); % mass at each age that satisfies the restriction (includes the age weights)
        RestrictionMask{rr}=logical(RestrictionValues);

        if all(restrictedsamplemass(rr,:)==0) && simoptions.warnzerorestrictedmass==2
            warning('One of the conditional restrictions evaluates to a zero mass (at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:); % Just return this and hopefully it is clear to the user
        else
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:); % Seems likely this would be something user might want
        end


        % Preallocate various things for the stats (as many will have jj as a dimension)
        % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
        for ff=1:numFnsToEvaluate
            wsP=max(reshape(whichstatsArr(ff,:,1+rr,:),[ngroups,7]),[],1); % as above, for this restriction
            ws4=reshape(whichstatsArr(ff,:,1+rr,4),[1,ngroups]);
            if any(ws4>=1 & ws4<3)
                wsP(4)=1;
            end
            if wsP(1)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(1)==1 && wsP(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(3)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(4)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
                if wsP(4)<3
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
                end
            end
            if wsP(5)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(6)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
            end
            if wsP(7)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
        end
    end


end



%% Create a different 'Values' for each of the variable to be evaluated
if N_z==0
    % StationaryDistVec=reshape(StationaryDist,[N_a,N_j]);
    %
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    % a_gridvals=CreateGridvals(n_a,a_grid,1);

    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'


        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            if ~any(whichcombos(ff,kk,:)) % no combination of this (function, age group) is wanted
                continue
            end
            Values=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end

            Values=reshape(Values,[N_a*(jend-j1+1),1]);

            if useCondlRest==1
                % The conditional restrictions only change the weights, so sort Values once and reuse the sort for the unrestricted stats and for every restriction
                % (only the points with positive mass are sorted; every restricted distribution is zero outside these)
                positivemass_kk=(StationaryDistVec_kk>0);
                [SortedValues_kk,SortedValues_index_kk]=sort(Values(positivemass_kk));
                SortedStationaryDistVec_kk=StationaryDistVec_kk(positivemass_kk);
                SortedStationaryDistVec_kk=SortedStationaryDistVec_kk(SortedValues_index_kk);
            end
            if whichcombos(ff,kk,1)==1 % the unrestricted stats of this (function, age group) are wanted
            ws=reshape(whichstatsArr(ff,kk,1,:),[1,7]);
            if useCondlRest==0
                tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws);
            else
                tempStats=StatsFromWeightedGrid(SortedValues_kk,SortedStationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws);
            end

            % Store them in AgeConditionalStats
            if ws(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if ws(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                if ws(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                end
            end
            if ws(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if ws(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                if ws(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                end
            end
            if ws(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if ws(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
            end
            if ws(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            end
            end % whichcombos(ff,kk,1)

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditional restrictions:
                % Only change is to use the restricted distribution (the mask times the agent distribution, normalised) as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if whichcombos(ff,kk,1+rr)==1 && sum(restrictedsamplemass(rr,j1:jend))>0 % this restriction is wanted for this (function, age group) and has mass
                        wsr=reshape(whichstatsArr(ff,kk,1+rr,:),[1,7]);
                        RestrictedStationaryDistVec_kk=StationaryDist(:,j1:jend).*RestrictionMask{rr}(:,j1:jend)/sum(restrictedsamplemass(rr,j1:jend)); % the restricted mass at each point of the age group, normalised to mass one over the group (so each age of the group is weighted by its restricted mass)
                        RestrictedStationaryDistVec_kk=RestrictedStationaryDistVec_kk(positivemass_kk);
                        tempStats=StatsFromWeightedGrid(SortedValues_kk,RestrictedStationaryDistVec_kk(SortedValues_index_kk),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,wsr);

                        % Store them in AgeConditionalStats
                        if wsr(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
                        end
                        if wsr(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                            if wsr(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if wsr(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
                        end
                        if wsr(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                            if wsr(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                            end
                        end
                        if wsr(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
                        end
                        if wsr(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
                        end
                        if wsr(7)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
                        end
                    end
                end
            end

        end
    end

else
    %% N_z
    % StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);
    %
    % PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    % a_gridvals=CreateGridvals(n_a,a_grid,1);

    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDist(:,j1:jend),[N_a*N_z*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'

        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            if ~any(whichcombos(ff,kk,:)) % no combination of this (function, age group) is wanted
                continue
            end
            Values=nan(N_a,N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            end

            Values=reshape(Values,[N_a*N_z*(jend-j1+1),1]);
            if useCondlRest==1
                % The conditional restrictions only change the weights, so sort Values once and reuse the sort for the unrestricted stats and for every restriction
                % (only the points with positive mass are sorted; every restricted distribution is zero outside these)
                positivemass_kk=(StationaryDistVec_kk>0);
                [SortedValues_kk,SortedValues_index_kk]=sort(Values(positivemass_kk));
                SortedStationaryDistVec_kk=StationaryDistVec_kk(positivemass_kk);
                SortedStationaryDistVec_kk=SortedStationaryDistVec_kk(SortedValues_index_kk);
            end
            if whichcombos(ff,kk,1)==1 % the unrestricted stats of this (function, age group) are wanted
            ws=reshape(whichstatsArr(ff,kk,1,:),[1,7]);
            if useCondlRest==0
                tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,ws);
            else
                tempStats=StatsFromWeightedGrid(SortedValues_kk,SortedStationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,1,ws);
            end

            % Store them in AgeConditionalStats
            if ws(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if ws(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
            end
            if ws(1)==1 && ws(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
            end
            if ws(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if ws(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                if ws(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                end
            end
            if ws(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if ws(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
            end
            if ws(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            end
            end % whichcombos(ff,kk,1)


            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditional restrictions:
                % Only change is to use the restricted distribution (the mask times the agent distribution, normalised) as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if whichcombos(ff,kk,1+rr)==1 && sum(restrictedsamplemass(rr,j1:jend))>0 % this restriction is wanted for this (function, age group) and has mass
                        wsr=reshape(whichstatsArr(ff,kk,1+rr,:),[1,7]);
                        RestrictedStationaryDistVec_kk=StationaryDist(:,j1:jend).*RestrictionMask{rr}(:,j1:jend)/sum(restrictedsamplemass(rr,j1:jend)); % the restricted mass at each point of the age group, normalised to mass one over the group (so each age of the group is weighted by its restricted mass)
                        RestrictedStationaryDistVec_kk=RestrictedStationaryDistVec_kk(positivemass_kk);
                        tempStats=StatsFromWeightedGrid(SortedValues_kk,RestrictedStationaryDistVec_kk(SortedValues_index_kk),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,2,wsr);

                        % Store them in AgeConditionalStats
                        if wsr(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
                        end
                        if wsr(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                            if wsr(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if wsr(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
                        end
                        if wsr(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                            if wsr(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                            end
                        end
                        if wsr(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
                        end
                        if wsr(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
                        end
                        if wsr(7)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
                        end
                    end
                end
            end
        end
    end
end





end


