function AgeConditionalStats=LifeCycleProfiles_FHorz_Case1_noz(StationaryDist,Policy,FnsToEvaluate,Parameters,FnsToEvaluateParamNames,n_d,n_a,N_j,d_grid,a_grid,simoptions)
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

if ~isfield(simoptions,'agegroupings')
    simoptions.agegroupings=1:1:N_j; % by default does each period seperately, can be used to say, calculate gini for age bins
end

% N_d=prod(n_d);
N_a=prod(n_a);

if isempty(n_d)
    l_d=0;
    n_d=0;
elseif n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+1:end}}; % the first inputs will always be (d,aprime,a,z)
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

StationaryDistVec=reshape(StationaryDist,[N_a,N_j]);
ngroups=length(simoptions.agegroupings);
if simoptions.parallel==2
    PolicyValues=PolicyInd2Val_FHorz_Case1_noz(Policy,n_d,n_a,N_j,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a)),1,1+l_a+1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,l_d+l_a,N_j]

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
    
    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*(l_d+l_a),N_j]);  % I reshape here, and THEN JUST RESHAPE AGAIN WHEN USING. THIS IS STUPID AND SLOW.
    for kk=1:length(simoptions.agegroupings)
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*(jend-j1+1),1]);
        tempmass=sum(StationaryDistVec_kk); % Just needed to handle the case that it equals zero because of possiblity there is noone of this age
        if tempmass>0 % Need to allow for possiblity there is noone of this age
            StationaryDistVec_kk=StationaryDistVec_kk./tempmass; % Normalize to sum to one for this 'agegrouping'
        end
        
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
                IncludeObs(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1_noz(simoptions.SampleRestrictionFn, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d+l_a]),n_d,n_a,a_grid,simoptions.parallel),[N_a,1]);
            end
           
            IncludeObs=reshape(logical(IncludeObs),[N_a*(jend-j1+1),1]);
            
            if simoptions.SampleRestrictionFn_include==0
                IncludeObs=(~IncludeObs);
            end
            % Can just do the sample restriction once now for the
            % stationary dist, and then later each time for the Values
            StationaryDistVec_kk=StationaryDistVec_kk(IncludeObs);
        end
        
        %%
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a,jend-j1+1,'gpuArray'); % Preallocate
            for jj=j1:jend
                % Includes check for cases in which no parameters are actually required
                if isempty(FnsToEvaluateParamNames(ii).Names)% check for 'FnsToEvaluateParamNames={}'
                    FnsToEvaluateParamsVec=[];
                else
                    FnsToEvaluateParamsVec=gpuArray(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                end
                Values(:,jj-j1+1)=reshape(EvalFnOnAgentDist_Grid_Case1_noz(FnsToEvaluate{ii}, FnsToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,l_d+l_a]),n_d,n_a,a_grid,simoptions.parallel),[N_a,1]);
            end
            
            Values=reshape(Values,[N_a*(jend-j1+1),1]);
            
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
                
                if tempmass>0
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
    
    a_gridvals=CreateGridvals(n_a,a_grid,2);

    PolicyIndexes=reshape(Policy,[size(Policy,1),N_a,N_j]);
    for kk=1:length(simoptions.agegroupings)
        j1=simoptions.agegroupings(kk);
        if kk<length(simoptions.agegroupings)
            jend=simoptions.agegroupings(kk+1)-1;
        else
            jend=N_j;
        end
        StationaryDistVec_kk=reshape(StationaryDistVec(:,j1:jend),[N_a*(jend-j1+1),1]);
        tempmass=sum(StationaryDistVec_kk); % Just needed to handle the case that it equals zero because of possiblity there is noone of this age
        if tempmass>0 % Need to allow for possiblity there is noone of this age
            StationaryDistVec_kk=StationaryDistVec_kk./tempmass; % Normalize to sum to one for this 'agegrouping'
        end
        
        clear gridvalsFull
        for jj=j1:jend
            [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,n_a,n_a,0,d_grid,a_grid,simoptions,1, 2);
            gridvalsFull(jj-j1+1).d_gridvals=d_gridvals;
            gridvalsFull(jj-j1+1).aprime_gridvals=aprime_gridvals;
        end
        
        for ii=1:length(FnsToEvaluate) % Each of the functions to be evaluated on the grid
            Values=nan(N_a,jend-j1+1); % Preallocate
            if l_d>0
                for jj=j1:jend
                    d_gridvals=gridvalsFull(jj-j1+1).d_gridvals;
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % check for 'FnsToEvaluateParamNames={}'
                        for ll=1:N_a
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(d_gridvals{ll,:},aprime_gridvals{ll,:},a_gridvals{ll,:});
                        end
                    else
                        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                        for ll=1:N_a
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(d_gridvals{ll,:},aprime_gridvals{ll,:},a_gridvals{ll,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end
            else % l_d==0
                for jj=j1:jend
                    aprime_gridvals=gridvalsFull(jj-j1+1).aprime_gridvals;
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(ii).Names) % check for 'FnsToEvaluateParamNames={}'
                        for ll=1:N_a
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{ll,:},a_gridvals{ll,:});
                        end
                    else
                        FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(ii).Names,jj));
                        for ll=1:N_a
                            Values(ll,jj-j1+1)=FnsToEvaluate{ii}(aprime_gridvals{ll,:},a_gridvals{ll,:},FnToEvaluateParamsCell{:});
                        end
                    end
                end                   
            end
            
            % From here on is essentially identical to the 'with gpu' case.
            Values=reshape(Values,[N_a*(jend-j1+1),1]);
            
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
            
            if tempmass>0
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


