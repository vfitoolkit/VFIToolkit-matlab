function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions)
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


%% Check which simoptions have been declared, set all others to defaults 
if exist('simoptions','var')==1
    %Check options for missing fields, if there are some fill them with the defaults
    if isgpuarray(StationaryDist) % simoptions.parallel is overwritten based on StationaryDist
        simoptions.parallel=2;
    else
        simoptions.parallel=1;
    end
    if isfield(simoptions,'verbose')==0
        simoptions.verbose=0;
    end
    if isfield(simoptions,'nquantiles')==0
        simoptions.nquantiles=20; % by default gives ventiles
    end
    if isfield(simoptions,'agegroupings')==0
        simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
    end
    if isfield(simoptions,'npoints')==0
        simoptions.npoints=100; % number of points for lorenz curve (note this lorenz curve is also used to calculate the gini coefficient
    end
    if isfield(simoptions,'tolerance')==0    
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
        simoptions.SampleRestrictionFnParamNames=getAnonymousFnInputNames(simoptions.SampleRestrictionFn);
    end
    if isfield(simoptions,'experienceasset')==0    
        simoptions.experienceasset=0;
    end
else
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
    simoptions.experienceasset=0;
end

if simoptions.parallel==2 % just make sure things are on gpu as they should be
    StationaryDist=gpuArray(StationaryDist);
    Policy=gpuArray(Policy);
end

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if isempty(n_d)
    l_d=0;
    n_d=0;
elseif n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

if N_z==0
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_noz(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,N_j,d_grid,a_grid,simoptions);
    return
end

if n_z(1)==0
    l_z=0;
else
    l_z=length(n_z);
end

%%
if simoptions.experienceasset==1
    % Just rejig the decision variables and send off as a Case2
    n_d=[n_d,n_a(1:end-1)]; % Note: the decisions are all standard decisions, plus all the next period endogenous states except for the experience asset
    d_grid=[d_grid;a_grid(1:sum(n_a(1:end-1)))];
    AgeConditionalStats=LifeCycleProfiles_FHorz_Case2(StationaryDist,Policy,FnsToEvaluate,FnsToEvaluateParamNames,Parameters,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,simoptions);
    return
end

if simoptions.parallel==2
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
end

%% z_grid (and e_grid where appropriate)
eval('fieldexists_ExogShockFn=1;simoptions.ExogShockFn;','fieldexists_ExogShockFn=0;')
eval('fieldexists_ExogShockFnParamNames=1;simoptions.ExogShockFnParamNames;','fieldexists_ExogShockFnParamNames=0;')
eval('fieldexists_pi_z_J=1;simoptions.pi_z_J;','fieldexists_pi_z_J=0;')

if fieldexists_pi_z_J==1
    z_grid_J=simoptions.z_grid_J;
elseif fieldexists_ExogShockFn==1
    if size(z_grid,2)==1 % kronecker-grid
        z_grid_J=zeros(sum(n_z),N_j);
        for jj=1:N_j
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,~]=simoptions.ExogShockFn(jj);
            end
            z_grid_J(:,jj)=gather(z_grid);
        end
    elseif size(z_grid,2)==l_z % joint-grids
        z_grid_J=zeros(N_z,l_z,N_j);
        for jj=1:N_j
            if fieldexists_ExogShockFnParamNames==1
                ExogShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.ExogShockFnParamNames,jj);
                ExogShockFnParamsCell=cell(length(ExogShockFnParamsVec),1);
                for ii=1:length(ExogShockFnParamsVec)
                    ExogShockFnParamsCell(ii,1)={ExogShockFnParamsVec(ii)};
                end
                [z_grid,~]=simoptions.ExogShockFn(ExogShockFnParamsCell{:});
            else
                [z_grid,~]=simoptions.ExogShockFn(jj);
            end
            z_grid_J(:,:,jj)=gather(z_grid);
        end
    end
else
    if size(z_grid,2)==1 % kronecker-grid
        z_grid_J=repmat(z_grid,1,N_j);
    elseif size(z_grid,2)==l_z % joint-grids
        z_grid_J=zeros(N_z,l_z,N_j);
        for jj=1:N_j
            z_grid_J(:,:,jj)=z_grid;
        end
    end
end
if ndims(z_grid_J)==2
    jointzgrid=0;
elseif ndims(z_grid_J)==3
    jointzgrid=1;
end

%
if isfield(simoptions,'SemiExoStateFn') % If using semi-exogenous shocks
    % For purposes of function evaluation we can just treat the semi-exogenous states as exogenous states
    n_z=[n_z,simoptions.n_semiz];
    z_grid_J=[z_grid_J;simoptions.semiz_grid.*ones(1,N_j)];
    l_z=length(n_z);
end

%
if isfield(simoptions,'n_e')
    % Because of how FnsToEvaluate works I can just get the e variables and
    % then 'combine' them with z
    eval('fieldexists_EiidShockFn=1;simoptions.EiidShockFn;','fieldexists_EiidShockFn=0;')
    eval('fieldexists_EiidShockFnParamNames=1;simoptions.EiidShockFnParamNames;','fieldexists_EiidShockFnParamNames=0;')
    eval('fieldexists_pi_e_J=1;simoptions.pi_e_J;','fieldexists_pi_e_J=0;')
    
    n_e=simoptions.n_e;
    N_e=prod(n_e);
    l_e=length(n_e);
    
    if fieldexists_pi_e_J==1
        e_grid_J=simoptions.e_grid_J;
    elseif fieldexists_EiidShockFn==1
        if size(simoptions.e_grid,2)==1 % kronecker-grid
            e_grid_J=zeros(sum(simoptions.n_e),N_j);
            for jj=1:N_j
                if fieldexists_EiidShockFnParamNames==1
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
                else
                    [e_grid,~]=simoptions.EiidShockFn(jj);
                end
                e_grid_J(:,jj)=gather(e_grid);
            end
        elseif size(simoptions.e_grid,2)==l_e % joint-grids
            e_grid_J=zeros(N_e,l_e,N_j);
            for jj=1:N_j
                if fieldexists_EiidShockFnParamNames==1
                    EiidShockFnParamsVec=CreateVectorFromParams(Parameters, simoptions.EiidShockFnParamNames,jj);
                    EiidShockFnParamsCell=cell(length(EiidShockFnParamsVec),1);
                    for ii=1:length(EiidShockFnParamsVec)
                        EiidShockFnParamsCell(ii,1)={EiidShockFnParamsVec(ii)};
                    end
                    [e_grid,~]=simoptions.EiidShockFn(EiidShockFnParamsCell{:});
                else
                    [e_grid,~]=simoptions.EiidShockFn(jj);
                end
                e_grid_J(:,:,jj)=gather(e_grid);
            end
        end
    else
        if size(simoptions.e_grid,2)==1 % kronecker-grid
            e_grid_J=repmat(simoptions.e_grid,1,N_j);
        elseif size(simoptions.e_grid,2)==l_z % joint-grids
            e_grid_J=zeros(N_e,l_e,N_j);
            for jj=1:N_j
                e_grid_J(:,:,jj)=simoptions.e_grid;
            end
        end
    end
    
    if ndims(e_grid_J)==2
        jointegrid=0;
    elseif ndims(e_grid_J)==3
        jointegrid=1;
    end

    
    % Now combine into z
    if n_z(1)==0
        l_ze=l_e;
        n_ze=simoptions.n_e;
    else
        l_ze=l_z+l_e;
        n_ze=[n_z,n_e];
    end
    N_ze=prod(n_ze);
else
    N_e=0;
    n_ze=n_z;
    N_ze=N_z;
    l_ze=l_z;
end


%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_ze)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_ze+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
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

%% Create a different 'Values' for each of the variable to be evaluated

StationaryDistVec=reshape(StationaryDist,[N_a*N_ze,N_j]);

ngroups=length(simoptions.agegroupings);
if simoptions.parallel==2
    PolicyValues=PolicyInd2Val_FHorz_Case1(Policy,n_d,n_a,n_ze,N_j,d_grid,a_grid);

    permuteindexes=[1+(1:1:(l_a+l_ze)),1,1+l_a+l_ze+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d+l_a,N_j]

    % Do some preallocation of the output structure
    AgeConditionalStats=struct();
    for ii=length(FnsToEvaluate):-1:1 % Backwards as preallocating
        % length(FnsToEvaluate)
        AgeConditionalStats(ii).Mean=nan(1,ngroups,'gpuArray');
        AgeConditionalStats(ii).Median=nan(1,ngroups,'gpuArray');
        AgeConditionalStats(ii).Variance=nan(1,ngroups,'gpuArray');
        AgeConditionalStats(ii).LorenzCurve=nan(simoptions.npoints,ngroups,'gpuArray');
        AgeConditionalStats(ii).Gini=nan(1,ngroups,'gpuArray');
        AgeConditionalStats(ii).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups,'gpuArray'); % Includes the min and max values
        AgeConditionalStats(ii).QuantileMeans=nan(simoptions.nquantiles,ngroups,'gpuArray');
    end
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_ze*(l_d+l_a),N_j]);  % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
    
    for kk=1:length(simoptions.agegroupings)
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*N_ze*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'
        
        %%
        if isfield(simoptions,'SampleRestrictionFn')
            IncludeObs=nan(N_a*N_ze,jend-j1+1,'gpuArray'); % Preallocate
            if N_z>0 && N_e==0
                for jj=j1:jend
                    if jointzgrid==0
                        z_grid=z_grid_J(:,jj);
                    else
                        z_grid=z_grid_J(:,:,jj);
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(simoptions.SampleRestrictionFnParamNames)% check for 'FnsToEvaluateParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,simoptions.SampleRestrictionFnParamNames,jj));
                    end
                    IncludeObs(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1(simoptions.SampleRestrictionFn, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel),[N_a*N_z,1]);
                end
            elseif N_z==0 && N_e>0
                for jj=j1:jend
                    if jointegrid==0
                        e_grid=e_grid_J(:,jj);
                    else
                        e_grid=e_grid_J(:,:,jj);
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(simoptions.SampleRestrictionFnParamNames) % check for 'FnsToEvaluateParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,simoptions.SampleRestrictionFnParamNames,jj));
                    end
                    IncludeObs(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1(simoptions.SampleRestrictionFn, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_e,l_d+l_a]),n_d,n_a,n_e,a_grid,e_grid,simoptions.parallel),[N_a*N_e,1]);
                end
            elseif N_z>0 && N_e>0
                for jj=j1:jend
                    if jointzgrid==0
                        z_grid=z_grid_J(:,jj);
                    else
                        z_grid=z_grid_J(:,:,jj);
                    end
                    if jointegrid==0
                        e_grid=e_grid_J(:,jj);
                    else
                        e_grid=e_grid_J(:,:,jj);
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(simoptions.SampleRestrictionFnParamNames)% check for 'FnsToEvaluateParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,simoptions.SampleRestrictionFnParamNames,jj));
                    end
                    IncludeObs(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1e(simoptions.SampleRestrictionFn, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_ze,l_d+l_a]),n_d,n_a,n_z,n_e,a_grid,z_grid,e_grid,simoptions.parallel),[N_a*N_ze,1]);
                end
            end
            IncludeObs=reshape(logical(IncludeObs),[N_a*N_ze*(jend-j1+1),1]);
            
            if simoptions.SampleRestrictionFn_include==0
                IncludeObs=(~IncludeObs);
            end
            % Can just do the sample restriction once now for the
            % stationary dist, and then later each time for the Values
            StationaryDistVec_kk=StationaryDistVec_kk(IncludeObs);
        end
        
        %%
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a*N_ze,jend-j1+1,'gpuArray'); % Preallocate
            if N_z>0 && N_e==0
                for jj=j1:jend
                    if jointzgrid==0
                        z_grid=z_grid_J(:,jj);
                    else
                        z_grid=z_grid_J(:,:,jj);
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)% check for 'FnsToEvaluateParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    end
                    Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d+l_a]),n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel),[N_a*N_z,1]);
                end
            elseif N_z==0 && N_e>0
                for jj=j1:jend
                    if jointegrid==0
                        e_grid=e_grid_J(:,jj);
                    else
                        e_grid=e_grid_J(:,:,jj);
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)% check for 'FnsToEvaluateParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    end
                    Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_e,l_d+l_a]),n_d,n_a,n_e,a_grid,e_grid,simoptions.parallel),[N_a*N_e,1]);
                end
            elseif N_z>0 && N_e>0
                for jj=j1:jend
                    if jointzgrid==0
                        z_grid=z_grid_J(:,jj);
                    else
                        z_grid=z_grid_J(:,:,jj);
                    end                  
                    if jointegrid==0
                        e_grid=e_grid_J(:,jj);
                    else
                        e_grid=e_grid_J(:,:,jj);
                    end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names)% check for 'FnsToEvaluateParamNames={}'
                        FnsToEvaluateParamsVec=[];
                    else
                        FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                    end
                    Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1e(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_ze,l_d+l_a]),n_d,n_a,n_z,n_e,a_grid,z_grid,e_grid,simoptions.parallel),[N_a*N_ze,1]);
                end
            end
            
            Values=reshape(Values,[N_a*N_ze*(jend-j1+1),1]);
            
            if isfield(simoptions,'SampleRestrictionFn')
                Values=Values(IncludeObs);
                % Note: stationary dist has already been restricted (if relevant)
            end            
            
            [SortedValues,SortedValues_index] = sort(Values);
            
            SortedWeights = StationaryDistVec_kk(SortedValues_index);
            CumSumSortedWeights=cumsum(SortedWeights);
            
            WeightedValues=Values.*StationaryDistVec_kk;
            SortedWeightedValues=WeightedValues(SortedValues_index);
            
            % Calculate the 'age conditional' mean
            AgeConditionalStats(ii).Mean(kk)=sum(WeightedValues);
            % Calculate the 'age conditional' median
            [~,medianindex]=min(abs(SortedWeights-0.5));
            AgeConditionalStats(ii).Median(kk)=SortedValues(medianindex); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
            
            
            if SortedValues(1)==SortedValues(end)
                % The current FnsToEvaluate takes only one value, so nothing but the mean and median make sense
                AgeConditionalStats(ii).Variance(kk)=0;
                AgeConditionalStats(ii).LorenzCurve(:,kk)=linspace(1/simoptions.npoints,1/simoptions.npoints,1);
                AgeConditionalStats(ii).Gini(kk)=0;
                AgeConditionalStats(ii).QuantileCutoffs(:,kk)=nan(simoptions.nquantiles+1,1,'gpuArray');
                AgeConditionalStats(ii).QuantileMeans(:,kk)=SortedValues(1)*ones(simoptions.nquantiles,1);
            else
                % Calculate the 'age conditional' variance
                AgeConditionalStats(ii).Variance(kk)=sum((Values.^2).*StationaryDistVec_kk)-(AgeConditionalStats(ii).Mean(kk))^2; % Weighted square of values - mean^2
                
                if simoptions.npoints>0
                    if SortedWeightedValues(1)<0
                        AgeConditionalStats(ii).LorenzCurve(:,kk)=nan(simoptions.npoints,1);
                        AgeConditionalStats(ii).LorenzCurveComment(kk)={'Lorenz curve cannot be calculated as some values are negative'};
                        AgeConditionalStats(ii).Gini(kk)=nan;
                        AgeConditionalStats(ii).GiniComment(kk)={'Gini cannot be calculated as some values are negative'};
                    else
                        % Calculate the 'age conditional' lorenz curve
                        AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,2);
                        % Calculate the 'age conditional' gini
                        AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
                    end
                end
                
                % Calculate the 'age conditional' quantile means (ventiles by default)
                % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
                QuantileIndexes=zeros(simoptions.nquantiles-1,1,'gpuArray');
                QuantileCutoffs=zeros(simoptions.nquantiles-1,1,'gpuArray');
                QuantileMeans=zeros(simoptions.nquantiles,1,'gpuArray');
                
                for ll=1:simoptions.nquantiles-1
                    tempindex=find(CumSumSortedWeights>=ll/simoptions.nquantiles,1,'first');
                    QuantileIndexes(ll)=tempindex;
                    QuantileCutoffs(ll)=SortedValues(tempindex);
                    if ll==1
                        QuantileMeans(ll)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
                    elseif ll<(simoptions.nquantiles-1) % (1<ll) &&
                        QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                    else %if ll==(options.nquantiles-1)
                        QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                        QuantileMeans(ll+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
                    end
                end
                % Min value
                tempindex=find(CumSumSortedWeights>=simoptions.tolerance,1,'first');
                minvalue=SortedValues(tempindex);
                % Max value
                tempindex=find(CumSumSortedWeights>=(1-simoptions.tolerance),1,'first');
                maxvalue=SortedValues(tempindex);
                AgeConditionalStats(ii).QuantileCutoffs(:,kk)=[minvalue; QuantileCutoffs; maxvalue];
                AgeConditionalStats(ii).QuantileMeans(:,kk)=QuantileMeans;
            end
        end
    end
else % options.parallel~=2
    % Do some preallocation of the output structure
    AgeConditionalStats=struct();
    for ii=length(FnsToEvaluate):-1:1 % Backwards as preallocating
        % length(FnsToEvaluate)
        AgeConditionalStats(ii).Mean=nan(1,ngroups);
        AgeConditionalStats(ii).Median=nan(1,ngroups);
        AgeConditionalStats(ii).Variance=nan(1,ngroups);
        AgeConditionalStats(ii).LorenzCurve=nan(simoptions.npoints,ngroups);
        AgeConditionalStats(ii).Gini=nan(1,ngroups);
        AgeConditionalStats(ii).QuantileCutoffs=nan(simoptions.nquantiles+1,ngroups); % Includes the min and max values
        AgeConditionalStats(ii).QuantileMeans=nan(simoptions.nquantiles,ngroups);
    end
    
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
%     z_grid=gather(z_grid);
    
    a_gridvals=CreateGridvals(n_a,a_grid,2);

    PolicyIndexes=reshape(Policy,[size(Policy,1),N_a,N_ze,N_j]);
    for kk=1:length(simoptions.agegroupings)
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*N_ze*(jend-j1+1),1]);
        StationaryDistVec_kk=StationaryDistVec_kk./sum(StationaryDistVec_kk); % Normalize to sum to one for this 'agegrouping'
        
        clear gridvalsFull
        for jj=j1:jend
            [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,:,jj),n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
            gridvalsFull(jj-j1+1).d_gridvals=d_gridvals;
            gridvalsFull(jj-j1+1).aprime_gridvals=aprime_gridvals;
            
            if N_z>0
                if jointzgrid==0
                    z_gridvals=CreateGridvals(n_z,z_grid_J(:,jj),2);
                else
                    z_gridvals=z_grid_J(:,:,jj);
                end
            end
            if N_e>0
                if jointegrid==0
                    e_gridvals=CreateGridvals(simoptions.n_e,e_grid_J(:,jj),2);
                else
                    e_gridvals=e_grid_J(:,:,jj);
                end
            end
            if N_z>0 && N_e>0
                z_gridvals=[kron(z_gridvals,ones(N_e,1)),kron(ones(N_z,1),e_gridvals)];
            elseif N_e>0 && N_z==0
                z_gridvals=e_gridvals;
                % Note that n_z>0 and n_e=0 we just leave z_gridvals as is
            end
            
            gridvalsFull(jj-j1+1).z_gridvals=z_gridvals;
        end
        
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a*N_ze,jend-j1+1); % Preallocate
            if l_d>0
                for jj=j1:jend
                    d_gridvals=gridvalsFull(jj-j1+1).d_gridvals;
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
                    z_gridvals=gridvalsFull(jj-j1+1).z_gridvals;
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % check for 'FnsToEvaluateParamNames={}'
                        for ll=1:N_a*N_ze
                            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(d_gridvals{l1+(l2-1)*N_a,:},aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:});
                        end
                    else
                        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                        for ll=1:N_a*N_ze
                            %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(d_gridvals{l1+(l2-1)*N_a,:},aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            else % l_d==0
                for jj=j1:jend
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
                    z_gridvals=gridvalsFull(jj-j1+1).z_gridvals;
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % check for 'FnsToEvaluateParamNames={}'
                        for ll=1:N_a*N_ze
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:});
                        end
                    else
                        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                        for ll=1:N_a*N_ze
                            l1=rem(ll-1,N_a)+1;
                            l2=ceil(ll/N_a);
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{l1+(l2-1)*N_a,:},a_gridvals{l1,:},z_gridvals{l2,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end                   
            end
            
            % From here on is essentially identical to the 'with gpu' case.
            Values=reshape(Values,[N_a*N_ze*(jend-j1+1),1]);
            
            [SortedValues,SortedValues_index] = sort(Values);
            
            SortedWeights = StationaryDistVec_kk(SortedValues_index);
            CumSumSortedWeights=cumsum(SortedWeights);
            
            WeightedValues=Values.*StationaryDistVec_kk;
            SortedWeightedValues=WeightedValues(SortedValues_index);
            
            % Calculate the 'age conditional' mean
            AgeConditionalStats(ii).Mean(kk)=sum(WeightedValues);
            % Calculate the 'age conditional' median
            [~,medianindex]=min(abs(SortedWeights-0.5));
            AgeConditionalStats(ii).Median(kk)=SortedValues(medianindex); % The max is just to deal with 'corner' case where there is only one element in SortedWeightedValues
            % Calculate the 'age conditional' variance
            AgeConditionalStats(ii).Variance(kk)=sum((Values.^2).*StationaryDistVec_kk)-(AgeConditionalStats(ii).Mean(kk))^2; % Weighted square of values - mean^2
            
            
            if simoptions.npoints>0
                % Calculate the 'age conditional' lorenz curve
                AgeConditionalStats(ii).LorenzCurve(:,kk)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,1);
                % Calculate the 'age conditional' gini
                AgeConditionalStats(ii).Gini(kk)=Gini_from_LorenzCurve(AgeConditionalStats(ii).LorenzCurve(:,kk));
            end
            
            % Calculate the 'age conditional' quantile means (ventiles by default)
            % Calculate the 'age conditional' quantile cutoffs (ventiles by default)
            QuantileIndexes=zeros(1,simoptions.nquantiles-1);
            QuantileCutoffs=zeros(1,simoptions.nquantiles-1);
            QuantileMeans=zeros(1,simoptions.nquantiles);
            
            for ll=1:simoptions.nquantiles-1
                tempindex=find(CumSumSortedWeights>=ll/simoptions.nquantiles,1,'first');
                QuantileIndexes(ll)=tempindex;
                QuantileCutoffs(ll)=SortedValues(tempindex);
                if ll==1
                    QuantileMeans(ll)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
                elseif ll<(simoptions.nquantiles-1) % (1<ll) &&
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                else %if ll==(options.nquantiles-1)
                    QuantileMeans(ll)=sum(SortedWeightedValues(QuantileIndexes(ll-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes(ll-1)));
                    QuantileMeans(ll+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
                end
            end
            % Min value
            tempindex=find(CumSumSortedWeights>=simoptions.tolerance,1,'first');
            minvalue=SortedValues(tempindex);
            % Max value
            tempindex=find(CumSumSortedWeights>=(1-simoptions.tolerance),1,'first');
            maxvalue=SortedValues(tempindex);
            AgeConditionalStats(ii).QuantileCutoffs(:,kk)=[minvalue, QuantileCutoffs, maxvalue]';
            AgeConditionalStats(ii).QuantileMeans(:,kk)=QuantileMeans';
            
        end
    end
end

if FnsToEvaluateStruct==1
    % Change the output into a structure
    AgeConditionalStats2=AgeConditionalStats;
    clear AgeConditionalStats
    AgeConditionalStats=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        AgeConditionalStats.(AggVarNames{ff}).Mean=AgeConditionalStats2(ff).Mean;
        AgeConditionalStats.(AggVarNames{ff}).Median=AgeConditionalStats2(ff).Median;
        AgeConditionalStats.(AggVarNames{ff}).Variance=AgeConditionalStats2(ff).Variance;
        AgeConditionalStats.(AggVarNames{ff}).LorenzCurve=AgeConditionalStats2(ff).LorenzCurve;
        AgeConditionalStats.(AggVarNames{ff}).Gini=AgeConditionalStats2(ff).Gini;
        AgeConditionalStats.(AggVarNames{ff}).QuantileCutoffs=AgeConditionalStats2(ff).QuantileCutoffs;
        AgeConditionalStats.(AggVarNames{ff}).QuantileMeans=AgeConditionalStats2(ff).QuantileMeans;
    end
end


end


