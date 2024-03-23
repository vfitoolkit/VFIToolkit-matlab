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
    simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
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
        simoptions.whichstats=ones(7,1); % See StatsFromWeightedGrid(), zeros skip some stats and can be used to reduce runtimes 
    end
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

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

%%
% Internally, only ever use age-dependent joint-grids (makes all the code much easier to write)
% Gradually rolling these out so that all the commands build off of these
z_gridvals_J=zeros(prod(n_z),length(n_z),'gpuArray');
if prod(n_z)==0 % no z
    z_gridvals_J=[];
elseif ndims(z_grid)==3 % already an age-dependent joint-grid
    if all(size(z_grid)==[prod(n_z),length(n_z),N_j])
        z_gridvals_J=z_grid;
    end
elseif all(size(z_grid)==[sum(n_z),N_j]) % age-dependent grid
    for jj=1:N_j
        z_gridvals_J(:,:,jj)=CreateGridvals(n_z,z_grid(:,jj),1);
    end
elseif all(size(z_grid)==[prod(n_z),length(n_z)]) % joint grid
    z_gridvals_J=z_grid.*ones(1,1,N_j,'gpuArray');
elseif all(size(z_grid)==[sum(n_z),1]) % basic grid
    z_gridvals_J=CreateGridvals(n_z,z_grid,1).*ones(1,1,N_j,'gpuArray');
end
if isfield(simoptions,'ExogShockFn')
    if isfield(simoptions,'ExogShockFnParamNames')
        for jj=1:N_j
            ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
            ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
            for ff=1:length(ExogShockFnParamsVec)
                ExogShockFnParamsCell(ff,1)={ExogShockFnParamsVec(ff)};
            end
            [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    else
        for jj=1:N_j
            [z_grid,~]=simoptions.ExogShockFn(N_j);
            if all(size(z_grid)==[sum(n_z),1])
                z_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(n_z,z_grid,1));
            else % already joint-grid
                z_gridvals_J(:,:,jj)=gpuArray(z_grid,1);
            end
        end
    end
end

% If using e variable, do same for this
if isfield(simoptions,'n_e')
    n_e=simoptions.n_e;
    N_e=prod(n_e);
    if N_e==0
        % Do nothing
    else
        if isfield(simoptions,'e_grid_J')
            error('No longer use simoptions.e_grid_J, instead just put the age-dependent grid in simoptions.e_grid (functionality of VFI Toolkit has changed to make it easier to use)')
        end
        if ~isfield(simoptions,'e_grid') % && ~isfield(simoptions,'e_grid_J')
            error('You are using an e (iid) variable, and so need to declare simoptions.e_grid')
        elseif ~isfield(simoptions,'pi_e')
            error('You are using an e (iid) variable, and so need to declare simoptions.pi_e')
        end

        e_gridvals_J=zeros(prod(simoptions.n_e),length(simoptions.n_e),'gpuArray');
        if ndims(simoptions.e_grid)==3 % already an age-dependent joint-grid
            if all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e),N_j])
                e_gridvals_J=simoptions.e_grid;
            end
        elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),N_j]) % age-dependent grid
            for jj=1:N_j
                e_gridvals_J(:,:,jj)=CreateGridvals(simoptions.n_e,simoptions.e_grid(:,jj),1);
            end
        elseif all(size(simoptions.e_grid)==[prod(simoptions.n_e),length(simoptions.n_e)]) % joint grid
            e_gridvals_J=simoptions.e_grid.*ones(1,1,N_j,'gpuArray');
        elseif all(size(simoptions.e_grid)==[sum(simoptions.n_e),1]) % basic grid
            e_gridvals_J=CreateGridvals(simoptions.n_e,simoptions.e_grid,1).*ones(1,1,N_j,'gpuArray');
        end
        if isfield(simoptions,'ExogShockFn')
            if isfield(simoptions,'ExogShockFnParamNames')
                for jj=1:N_j
                    ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                    ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                    for ff=1:length(ExogShockFnParamsVec)
                        ExogShockFnParamsCell(ff,1)={ExogShockFnParamsVec(ff)};
                    end
                    [simoptions.e_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
                    if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                    end
                end
            else
                for jj=1:N_j
                    [simoptions.e_grid,simoptions.pi_e]=simoptions.ExogShockFn(N_j);
                    if all(size(simoptions.e_grid)==[sum(simoptions.n_e),1])
                        e_gridvals_J(:,:,jj)=gpuArray(CreateGridvals(simoptions.n_e,simoptions.e_grid,1));
                    else % already joint-grid
                        e_gridvals_J(:,:,jj)=gpuArray(simoptions.e_grid,1);
                    end
                end
            end
        end

        % Now put e into z as that is easiest way to handle it from now on
        if N_z==0
            z_gridvals_J=e_gridvals_J;
            n_z=n_e;
            N_z=prod(n_z);
        else
            z_gridvals_J=[repmat(z_gridvals_J,N_e,1),repelem(e_gridvals_J,N_z,1)];
            n_z=[n_z,n_e];
            N_z=prod(n_z);
        end
    end
    % Because we added e into z, we need to strip them out of simoptions
    simoptions=rmfield(simoptions,'n_e');
end

% Also semiz if that is used
if isfield(simoptions,'n_semiz') % If using semi-exogenous shocks
    if N_z==0
        n_z=simoptions.n_semiz;
        z_gridvals_J=CreateGridvals(simoptions.n_semiz,simoptions.semiz_grid,1).*ones(1,1,N_j,'gpuArray');
    else
        % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
        n_z=[simoptions.n_semiz,n_z];
        z_gridvals_J=[repmat(CreateGridvals(simoptions.n_semiz,simoptions.semiz_grid,1).*ones(1,1,N_j,'gpuArray'),N_z,1),repelem(z_gridvals_J,prod(simoptions.n_semiz),1)];
    end
    % Because we added semiz into z, we need to strip them out of simoptions
    simoptions=rmfield(simoptions,'n_semiz');
end
N_z=prod(n_z);
if N_z==0
    l_z=0;
else
    l_z=length(n_z);
end



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

if isfield(simoptions,'SampleRestrictionFn') % If using SampleRestrictionFn then need to set some things
    if length(simoptions.SampleRestrictionFnParamNames)==(l_daprime+l_a+l_z)
        simoptions.SampleRestrictionFnParamNames={};
    else
        simoptions.SampleRestrictionFnParamNames=simoptions.SampleRestrictionFnParamNames{l_daprime+l_a+l_z+1:end};
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
    if simoptions.whichstats(3)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Variance=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(4)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Gini=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=nan(simoptions.npoints,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(5)==1
        AgeConditionalStats.(FnsToEvalNames{ff}).Minimum=nan(1,length(simoptions.agegroupings),'gpuArray');
        AgeConditionalStats.(FnsToEvalNames{ff}).Maximum=nan(1,length(simoptions.agegroupings),'gpuArray');
    end
    if simoptions.whichstats(6)==1
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




%% Create a different 'Values' for each of the variable to be evaluated
if N_z==0
    StationaryDistVec=reshape(StationaryDist,[N_a,N_j]);

    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,0,N_j,d_grid,a_grid,simoptions,1);
    a_gridvals=CreateGridvals(n_a,a_grid,1);

    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'

        %%
        if isfield(simoptions,'SampleRestrictionFn')
            IncludeObs=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                if isempty(simoptions.SampleRestrictionFnParamNames)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,simoptions.SampleRestrictionFnParamNames,jj));
                end
                IncludeObs(:,jj-j1+1)=EvalFnOnAgentDist_Grid(simoptions.SampleRestrictionFn, FnsToEvaluateParamsVec,PolicyValues(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end
            IncludeObs=reshape(logical(IncludeObs),[N_a*(jend-j1+1),1]);

            if simoptions.SampleRestrictionFn_include==0
                IncludeObs=(~IncludeObs);
            end
            % Can just do the sample restriction once now for the
            % stationary dist, and then later each time for the Values
            StationaryDistVec_kk=StationaryDistVec_kk(IncludeObs);
            StationaryDistVec_kk=StationaryDistVec_kk/sum(StationaryDistVec_kk); % Normalize
        end
        
        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            Values=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ff).Names)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
                end
                Values(:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnsToEvaluateParamsVec,PolicyValues(:,:,jj),l_daprime,n_a,0,a_gridvals,[]);
            end

            Values=reshape(Values,[N_a*(jend-j1+1),1]);

            if isfield(simoptions,'SampleRestrictionFn')
                Values=Values(IncludeObs);
                % Note: stationary dist has already been restricted (if relevant)
            end
            
            tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);
            
            
            % Store them in AgeConditionalStats
            AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
            AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
            AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
            AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
            AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top1share(kk)=tempStats.MoreInequality.Top1share;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top5share(kk)=tempStats.MoreInequality.Top5share;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Top10share(kk)=tempStats.MoreInequality.Top10share;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Bottom50share(kk)=tempStats.MoreInequality.Bottom50share;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile50th(kk)=tempStats.MoreInequality.Percentile50th;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile90th(kk)=tempStats.MoreInequality.Percentile90th;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile95th(kk)=tempStats.MoreInequality.Percentile95th;
            AgeConditionalStats.(FnsToEvalNames{ff}).MoreInequality.Percentile99th(kk)=tempStats.MoreInequality.Percentile99th;
            AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
            AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs(:,kk)=tempStats.QuantileCutoffs;
            AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans(:,kk)=tempStats.QuantileMeans;

        end
    end

else % N_z
    %%
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,N_j]);

    PolicyValues=PolicyInd2Val_FHorz(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,a_grid,simoptions,1);
    a_gridvals=CreateGridvals(n_a,a_grid,1);

    for kk=1:ngroups
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*N_z*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'

        %%
        if isfield(simoptions,'SampleRestrictionFn')
            IncludeObs=nan(N_a,N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                if isempty(simoptions.SampleRestrictionFnParamNames)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,simoptions.SampleRestrictionFnParamNames,jj));
                end
                IncludeObs(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(simoptions.SampleRestrictionFn, FnsToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));
            end
            IncludeObs=reshape(logical(IncludeObs),[N_a*N_z*(jend-j1+1),1]);

            if simoptions.SampleRestrictionFn_include==0
                IncludeObs=(~IncludeObs);
            end
            % Can just do the sample restriction once now for the
            % stationary dist, and then later each time for the Values
            StationaryDistVec_kk=StationaryDistVec_kk(IncludeObs);
            StationaryDistVec_kk=StationaryDistVec_kk/sum(StationaryDistVec_kk); % Normalize
        end

        %%
        for ff=1:numFnsToEvaluate % Each of the functions to be evaluated on the grid
            Values=nan(N_a,N_z,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ff).Names)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ff).Names,jj));
                end

                Values(:,:,jj-j1+1)=EvalFnOnAgentDist_Grid(FnsToEvaluate{ff}, FnsToEvaluateParamsVec,PolicyValues(:,:,:,jj),l_daprime,n_a,n_z,a_gridvals,z_gridvals_J(:,:,jj));                
            end
            
            Values=reshape(Values,[N_a*N_z*(jend-j1+1),1]);

            if isfield(simoptions,'SampleRestrictionFn')
                Values=Values(IncludeObs);
                % Note: stationary dist has already been restricted (if relevant)
            end

            tempStats=StatsFromWeightedGrid(Values,StationaryDistVec_kk,simoptions.npoints,simoptions.nquantiles,simoptions.tolerance,0,simoptions.whichstats);

            % Store them in AgeConditionalStats
            if simoptions.whichstats(1)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Mean(kk)=tempStats.Mean;
            end
            if simoptions.whichstats(2)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Median(kk)=tempStats.Median;
            end
            if simoptions.whichstats(3)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Variance(kk)=tempStats.Variance;
                AgeConditionalStats.(FnsToEvalNames{ff}).StdDeviation(kk)=tempStats.StdDeviation;
            end
            if simoptions.whichstats(4)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve(:,kk)=tempStats.LorenzCurve;
                AgeConditionalStats.(FnsToEvalNames{ff}).Gini(kk)=tempStats.Gini;
            end
            if simoptions.whichstats(5)==1
                AgeConditionalStats.(FnsToEvalNames{ff}).Minimum(kk)=tempStats.Minimum;
                AgeConditionalStats.(FnsToEvalNames{ff}).Maximum(kk)=tempStats.Maximum;
            end
            if simoptions.whichstats(6)==1
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
        end
    end
end


% 
% if FnsToEvaluateStruct==1
%     % Change the output into a structure
%     AgeConditionalStats2=AgeConditionalStats;
%     clear AgeConditionalStats
%     AgeConditionalStats=struct();
% %     AggVarNames=fieldnames(FnsToEvaluate);
%     for ff=1:length(FnsToEvalNames)
%         AgeConditionalStats.(FnsToEvalNames{ff}).Mean=AgeConditionalStats2(ff).Mean;
%         AgeConditionalStats.(FnsToEvalNames{ff}).Median=AgeConditionalStats2(ff).Median;
%         AgeConditionalStats.(FnsToEvalNames{ff}).Variance=AgeConditionalStats2(ff).Variance;
%         AgeConditionalStats.(FnsToEvalNames{ff}).LorenzCurve=AgeConditionalStats2(ff).LorenzCurve;
%         AgeConditionalStats.(FnsToEvalNames{ff}).Gini=AgeConditionalStats2(ff).Gini;
%         AgeConditionalStats.(FnsToEvalNames{ff}).QuantileCutoffs=AgeConditionalStats2(ff).QuantileCutoffs;
%         AgeConditionalStats.(FnsToEvalNames{ff}).QuantileMeans=AgeConditionalStats2(ff).QuantileMeans;
%     end
% end


end


