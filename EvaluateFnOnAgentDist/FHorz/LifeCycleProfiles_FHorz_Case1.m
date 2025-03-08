function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,PolicyIndexes,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
% Similar to SimLifeCycleProfiles but works from StationaryDist rather than
% simulating panel data. Where applicable it is faster and more accurate.
% options.agegroupings can be used to do conditional on 'age bins' rather than age
% e.g., options.agegroupings=1:10:N_j will divide into 10 year age bins and calculate stats for each of them
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
    simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
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
    if isfield(simoptions,'ExogShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.ExogShockFnParamNames=getAnonymousFnInputNames(simoptions.ExogShockFn);
    end
    if isfield(simoptions,'EiidShockFn') % If using ExogShockFn then figure out the parameter names
        simoptions.EiidShockFnParamNames=getAnonymousFnInputNames(simoptions.EiidShockFn);
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
            simoptions.whichstats=[1,1,1,1,1,1,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
        else
            simoptions.whichstats=[1,1,1,2,1,2,1]; % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes
        end
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
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_cpu(StationaryDist,PolicyIndexes,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
    return
end
% just make sure things are on gpu as they should be
StationaryDist=gpuArray(StationaryDist);
PolicyIndexes=gpuArray(PolicyIndexes);
d_grid=gpuArray(d_grid);
a_grid=gpuArray(a_grid);

%% Exogenous shock grids
% Create the combination of (semiz,z,e) as all three are the same for FnsToEvaluate 
[n_z,z_gridvals_J,N_z,l_z,simoptions]=CreateGridvals_FnsToEvaluate_FHorz(n_z,z_grid,N_j,simoptions,Parameters);

%% Implement new way of handling FnsToEvaluate
% Figure out l_daprime from Policy
l_daprime=size(PolicyIndexes,1);

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


% Preallocate various things for the stats (as many will have jj as a dimension)
% Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
for ff=1:numFnsToEvaluate
    if simoptions.whichstats(1)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(2)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(1)==1 && simoptions.whichstats(2)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(3)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(4)>=1
        AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
        if simoptions.whichstats(4)<3
            AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
        end
    end
    if simoptions.whichstats(5)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(6)>=1
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
        AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(7)==1
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



%% I want to do some things now, so that they can be used in setting up conditional restrictions

l_daprime=size(PolicyIndexes,1);
a_gridvals=CreateGridvals(n_a,a_grid,1);
if N_z==0
    StationaryDist=reshape(StationaryDist,[N_a,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,1,3]); % (N_a,l_daprime,N_j)
else
    StationaryDist=reshape(StationaryDist,[N_a*N_z,N_j]);
    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    PolicyValuesPermuteJ=permute(PolicyValues,[2,3,1,4]); % (N_a,N_z,l_daprime,N_j)
end


%% If there are any conditional restrictions, set up for these
% Evaluate AllStats, but conditional on the restriction being non-zero.

useCondlRest=0;
% Code works by evaluating the the restriction and imposing this on the distribution (and renormalizing it). 
if isfield(simoptions,'conditionalrestrictions')    
    useCondlRest=1;
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);

    restrictedsamplemass=nan(length(CondlRestnFnNames),N_j);
    RestrictionStruct=struct();
    
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
        
        RestrictedStationaryDistVec=StationaryDist;
        RestrictedStationaryDistVec(~RestrictionValues)=0; % zero mass on all points that do not meet the restriction
        
        % Need to keep two things, the restrictedsamplemass and the RestrictedStationaryDistVec (normalized to have mass of 1)
        restrictedsamplemass(rr,:)=sum(RestrictedStationaryDistVec,1);
        noonethatagej=(restrictedsamplemass(rr,:)==0); % 1 when mass of that age is 0
        RestrictionStruct(rr).RestrictedStationaryDistVec=(1-noonethatagej).*(RestrictedStationaryDistVec./(restrictedsamplemass(rr,:)+noonethatagej)); % Just RestrictedStationaryDistVec./restrictedsamplemass(rr,:), but get 0 instead of NaN if that agej is mass zero in restrictedsamplemass(rr,:) 

        if all(restrictedsamplemass(rr,:)==0)
            warning('One of the conditional restrictions evaluates to a zero mass (at all j)')
            fprintf(['Specifically, the restriction called ',CondlRestnFnNames{rr},' has a restricted sample that is of zero mass \n'])
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:); % Just return this and hopefully it is clear to the user
        else
            AgeConditionalStats.(CondlRestnFnNames{rr}).RestrictedSampleMass=restrictedsamplemass(rr,:); % Seems likely this would be something user might want
        end


        % Preallocate various things for the stats (as many will have jj as a dimension)
        % Stats to calculate and store in AgeConditionalStats.(FnsToEvalNames{ff})
        for ff=1:numFnsToEvaluate
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(1)==1 && simoptions.whichstats(2)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(3)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(4)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
                if simoptions.whichstats(4)<3
                    AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
                end
            end
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(6)>=1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
            end
            if simoptions.whichstats(7)==1
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top1share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top5share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Top10share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Bottom50share=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile50th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile90th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile95th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).MoreInequality.Percentile99th=nan(1,length(simoptions.agegroupings),'gpuArray');
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs=nan(simoptions.nquantiles+1,length(simoptions.agegroupings),'gpuArray'); % Includes the min and max values
                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans=nan(simoptions.nquantiles,length(simoptions.agegroupings),'gpuArray');
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
            Values=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end
            
            Values=reshape(Values,[N_a*(jend-j1+1),1]);
            
            tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
            
            % Store them in AgeConditionalStats
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if simoptions.whichstats(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                if simoptions.whichstats(1)==1
                    AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                end
            end
            if simoptions.whichstats(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if simoptions.whichstats(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                if simoptions.whichstats(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                end
            end
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if simoptions.whichstats(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
            end
            if simoptions.whichstats(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            end

            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditinal restrictions:
                % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if sum(restrictedsamplemass(rr,j1:jend))>0
                        tempStats=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec(:,j1:jend),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                        
                        % Store them in AgeConditionalStats
                        if simoptions.whichstats(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
                        end
                        if simoptions.whichstats(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                            if simoptions.whichstats(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if simoptions.whichstats(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
                        end
                        if simoptions.whichstats(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                            if simoptions.whichstats(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                            end
                        end
                        if simoptions.whichstats(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
                        end
                        if simoptions.whichstats(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
                        end
                        if simoptions.whichstats(7)==1
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
            Values=nan(N_a,N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                FnToEvaluateParamsCell=CreateCellFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj);
                Values(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnToEvaluateParamsCell,PolicyValuesPermuteJ(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));                
            end
            
            Values=reshape(Values,[N_a*N_z*(jend-j1+1),1]);
            tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

            % Store them in AgeConditionalStats
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if simoptions.whichstats(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
            end
            if simoptions.whichstats(1)==1 && simoptions.whichstats(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
            end
            if simoptions.whichstats(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if simoptions.whichstats(4)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                if simoptions.whichstats(4)<3
                    AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                end
            end
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if simoptions.whichstats(6)>=1
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
            end
            if simoptions.whichstats(7)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
                AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            end


            %% If there are any conditional restrictions then deal with these
            % Evaluate AllStats, but conditional on the restriction being one.
            if useCondlRest==1
                % Evaluate the conditinal restrictions:
                % Only change is to use RestrictionStruct(rr).RestrictedStationaryDistVec as the agent distribution
                for rr=1:length(CondlRestnFnNames)
                    if sum(restrictedsamplemass(rr,j1:jend))>0
                        tempStats=StatsFromWeightedGrid(Values,RestrictionStruct(rr).RestrictedStationaryDistVec(:,j1:jend),simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
                        
                        % Store them in AgeConditionalStats
                        if simoptions.whichstats(1)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
                        end
                        if simoptions.whichstats(2)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
                            if simoptions.whichstats(1)==1
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).RatioMeanToMedian(kk)=tempStats.RatioMeanToMedian;
                            end
                        end
                        if simoptions.whichstats(3)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
                        end
                        if simoptions.whichstats(4)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
                            if simoptions.whichstats(4)<3
                                AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                            end
                        end
                        if simoptions.whichstats(5)==1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
                        end
                        if simoptions.whichstats(6)>=1
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
                            AgeConditionalStats.(CondlRestnFnNames{rr}).(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;
                        end
                        if simoptions.whichstats(7)==1
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


