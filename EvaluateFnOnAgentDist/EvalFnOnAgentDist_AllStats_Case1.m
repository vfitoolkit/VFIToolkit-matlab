function AllStats=EvalFnOnAgentDist_AllStats_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions)
% Returns a wide variety of statistics
%
% simoptions optional inputs

%%
if ~exist('simoptions','var')
    simoptions.parallel=1+(gpuDeviceCount>0);
    simoptions.npoints=100;
    simoptions.nquantiles=20;
else
    if ~isfield(simoptions,'parallel')
        simoptions.parallel=1+(gpuDeviceCount>0);
    end
    if ~isfield(simoptions,'npoints')
        simoptions.npoints=100;
    end
    if ~isfield(simoptions,'nquantiles')
        simoptions.nquantiles=20;
    end
end

Tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

AllStats=struct();

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluate_copy=FnsToEvaluate; % keep a copy in case needed for conditional restrictions
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    FnsToEvalNames=fieldnames(FnsToEvaluate);
    for ff=1:length(FnsToEvalNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(FnsToEvalNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(FnsToEvalNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if simoptions.parallel==2
    StationaryDistVec=gpuArray(StationaryDistVec);
    PolicyIndexes=gpuArray(PolicyIndexes);
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
        
    for kk=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(kk).Names) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names);
        end
        
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{kk}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
        WeightedValues=Values.*StationaryDistVec;
        WeightedValues(isnan(WeightedValues))=0; % Values of -Inf times weight of zero give nan, we want them to be zeros.        
        
        [SortedValues,SortedValues_index] = sort(Values);
        
        SortedWeights=StationaryDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        % Now calculate all the statistics
        
        % Mean
        Mean=sum(WeightedValues); % want to reuse in std dev
        AllStats.(FnsToEvalNames{kk}).Mean=Mean;
        % Standard deviation
        AllStats.(FnsToEvalNames{kk}).StdDev=sqrt(sum(StationaryDistVec.*((Values-Mean.*ones(N_a*N_z,1)).^2)));
        % Variance
        AllStats.(FnsToEvalNames{kk}).Variance=AllStats.(FnsToEvalNames{kk}).StdDev^2;
        
        % Top X share indexes (simoptions.npoints will be number of points in Lorenz Curve)
        Top1cutpoint=round(0.99*simoptions.npoints);
        Top5cutpoint=round(0.95*simoptions.npoints);
        Top10cutpoint=round(0.90*simoptions.npoints);
        Top50cutpoint=round(0.50*simoptions.npoints);
        LorenzCurve=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,2)';
        AllStats.(FnsToEvalNames{kk}).LorenzCurve=LorenzCurve;
        AllStats.(FnsToEvalNames{kk}).Top1share=1-LorenzCurve(Top1cutpoint);
        AllStats.(FnsToEvalNames{kk}).Top5share=1-LorenzCurve(Top5cutpoint);
        AllStats.(FnsToEvalNames{kk}).Top10share=1-LorenzCurve(Top10cutpoint);
        AllStats.(FnsToEvalNames{kk}).Bottom50share=LorenzCurve(Top50cutpoint);
        
        % Now some cutoffs (note: qlimitvec is effectively already the cumulative sum)
        index_median=find(CumSumSortedWeights>=0.5,1,'first');
        AllStats.(FnsToEvalNames{kk}).Median=SortedValues(index_median);
        AllStats.(FnsToEvalNames{kk}).Percentile50th=SortedValues(index_median);
        index_p90=find(CumSumSortedWeights>=0.90,1,'first');
        AllStats.(FnsToEvalNames{kk}).Percentile90th=SortedValues(index_p90);
        index_p95=find(CumSumSortedWeights>=0.95,1,'first');
        AllStats.(FnsToEvalNames{kk}).Percentile95th=SortedValues(index_p95);
        index_p99=find(CumSumSortedWeights>=0.99,1,'first');
        AllStats.(FnsToEvalNames{kk}).Percentile99th=SortedValues(index_p99);
        
        % Quantiles
        QuantileIndexes_kk=zeros(1,simoptions.nquantiles-1,'gpuArray');
        QuantileCutoffs_kk=zeros(1,simoptions.nquantiles-1,'gpuArray');
        QuantileMeans_kk=zeros(1,simoptions.nquantiles,'gpuArray');
        for ii=1:simoptions.nquantiles-1
            [~,tempindex]=find(CumSumSortedWeights>=ii/simoptions.nquantiles,1,'first');
            QuantileIndexes_kk(ii)=tempindex;
            QuantileCutoffs_kk(ii)=SortedValues(tempindex);
            if ii==1
                QuantileMeans_kk(ii)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<ii) && (ii<(simoptions.nquantiles-1))
                QuantileMeans_kk(ii)=sum(SortedWeightedValues(QuantileIndexes_kk(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_kk(ii-1)));
            elseif ii==(simoptions.nquantiles-1)
                QuantileMeans_kk(ii)=sum(SortedWeightedValues(QuantileIndexes_kk(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_kk(ii-1)));
                QuantileMeans_kk(ii+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        
        % Min value
        [~,tempindex]=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        [~,tempindex]=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        AllStats.(FnsToEvalNames{kk}).QuantileCutOffs=[minvalue, QuantileCutoffs_kk, maxvalue];
        AllStats.(FnsToEvalNames{kk}).QuantileMeans=QuantileMeans_kk;
        
    end
    
else
    StationaryDistVec=gather(StationaryDistVec);
    PolicyIndexes=gather(PolicyIndexes);

    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for kk=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(kk).Names) % check for 'FnsToEvaluateParamNames={}'
            Values=zeros(N_a*N_z,1);
            if l_d==0
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            else % l_d>0
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            end
        else
            Values=zeros(N_a*N_z,1);
            if l_d==0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            else % l_d>0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                for ii=1:N_a*N_z
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            end
        end
                
        WeightedValues=Values.*StationaryDistVec;
        
        
        [~,SortedValues_index] = sort(Values);
        
        SortedWeights=StationaryDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);

        % Now calculate all the statistics
        
        % Mean
        Mean=sum(WeightedValues); % want to reuse in std dev
        AllStats.(FnsToEvalNames{kk}).Mean=Mean;
        % Standard deviation
        AllStats.(FnsToEvalNames{kk}).StdDev=sqrt(sum(StationaryDistVec.*((Values-Mean.*ones(N_a*N_z,1)).^2)));
        % Variance
        AllStats.(FnsToEvalNames{kk}).Variance=AllStats.(FnsToEvalNames{kk}).StdDev^2;
        
        % Top X share indexes (simoptions.npoints will be number of points in Lorenz Curve)
        Top1cutpoint=round(0.99*simoptions.npoints);
        Top5cutpoint=round(0.95*simoptions.npoints);
        Top10cutpoint=round(0.90*simoptions.npoints);
        Top50cutpoint=round(0.50*simoptions.npoints);
        LorenzCurve=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedWeights,simoptions.npoints,1)';
        AllStats.(FnsToEvalNames{kk}).LorenzCurve=LorenzCurve;
        AllStats.(FnsToEvalNames{kk}).Top1share=1-LorenzCurve(Top1cutpoint);
        AllStats.(FnsToEvalNames{kk}).Top5share=1-LorenzCurve(Top5cutpoint);
        AllStats.(FnsToEvalNames{kk}).Top10share=1-LorenzCurve(Top10cutpoint);
        AllStats.(FnsToEvalNames{kk}).Bottom50share=LorenzCurve(Top50cutpoint);
        
        % Now some cutoffs (note: qlimitvec is effectively already the cumulative sum)
        index_median=find(CumSumSortedWeights>=0.5,1,'first');
        AllStats.(FnsToEvalNames{kk}).Median=SortedValues(index_median);
        AllStats.(FnsToEvalNames{kk}).Percentile50th=SortedValues(index_median);
        index_p90=find(CumSumSortedWeights>=0.90,1,'first');
        AllStats.(FnsToEvalNames{kk}).Percentile90th=SortedValues(index_p90);
        index_p95=find(CumSumSortedWeights>=0.95,1,'first');
        AllStats.(FnsToEvalNames{kk}).Percentile95th=SortedValues(index_p95);
        index_p99=find(CumSumSortedWeights>=0.99,1,'first');
        AllStats.(FnsToEvalNames{kk}).Percentile99th=SortedValues(index_p99);
        
        % Quantiles
        QuantileIndexes_kk=zeros(1,simoptions.nquantiles-1);
        QuantileCutoffs_kk=zeros(1,simoptions.nquantiles-1);
        QuantileMeans_kk=zeros(1,simoptions.nquantiles);
        for ii=1:simoptions.nquantiles-1
            [~,tempindex]=find(CumSumSortedWeights>=ii/simoptions.nquantiles,1,'first');
            QuantileIndexes_kk(ii)=tempindex;
            QuantileCutoffs_kk(ii)=SortedValues(tempindex);
            if ii==1
                QuantileMeans_kk(ii)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<ii) && (ii<(simoptions.nquantiles-1))
                QuantileMeans_kk(ii)=sum(SortedWeightedValues(QuantileIndexes_kk(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_kk(ii-1)));
            elseif ii==(simoptions.nquantiles-1)
                QuantileMeans_kk(ii)=sum(SortedWeightedValues(QuantileIndexes_kk(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_kk(ii-1)));
                QuantileMeans_kk(ii+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        [~,tempindex]=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        [~,tempindex]=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        AllStats.(FnsToEvalNames{kk}).QuantileCutOffs=[minvalue, QuantileCutoffs_kk, maxvalue];
        AllStats.(FnsToEvalNames{kk}).QuantileMeans=QuantileMeans_kk;
    end
end

%% If there are any conditional restrictions then send these off to be done
% Evaluate AllStats, but conditional on the restriction being non-zero.
%
% Code works by evaluating the the restriction and imposing this on the
% distribution (and renormalizing it) and then just sending this off to
% EvalFnOnAgendDist_AllStats_Case1() again. Some of the results are then
% modified so that there is both, e.g., 'mean' and 'total'.
if isfield(simoptions,'conditionalrestrictions')
    % First couple of lines get the conditional restrictions and convert
    % them to a names and cell
    CondlRestnFnNames=fieldnames(simoptions.conditionalrestrictions);
    for ff=1:length(CondlRestnFnNames)
        temp=getAnonymousFnInputNames(simoptions.conditionalrestrictions.(CondlRestnFnNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            CondlRestnFnParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            CondlRestnFnParamNames(ff).Names={};
        end
        CondlRestnFns{ff}=simoptions.conditionalrestrictions.(CondlRestnFnNames{ff});
    end
    simoptions=rmfield(simoptions,'conditionalrestrictions'); % Have to delete this before resend it to EvalFnOnAgentDist_AllStats_Case1()
    
    % Note that some things have already been created above, so we don't need
    % to recreate them to evaluated the restrictions.
    
    if simoptions.parallel==2
        % Evaluate the conditinal restrictions
        for kk=1:length(CondlRestnFnNames)
            % Includes check for cases in which no parameters are actually required
            if isempty(CondlRestnFnParamNames(kk).Names) % check for '={}'
                CondlRestnFnParamsVec=[];
            else
                CondlRestnFnParamsVec=CreateVectorFromParams(Parameters,CondlRestnFnParamNames(kk).Names);
            end
            
            Values=EvalFnOnAgentDist_Grid_Case1(CondlRestnFns{kk}, CondlRestnFnParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,simoptions.parallel);
            Values=reshape(Values,[N_a*N_z,1]);

            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{kk},' has a restricted sample that is of zero mass \n'])
                AllStats.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AllStats.(CondlRestnFnNames{kk})=EvalFnOnAgentDist_AllStats_Case1(RestrictedStationaryDistVec, PolicyIndexes, FnsToEvaluate_copy, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AllStats.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AllStats.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Mean;
                end
                AllStats.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    else % simoptions.parallel~=2
        for kk=1:length(FnsToEvaluate)
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames(kk).Names) % check for 'FnsToEvaluateParamNames={}'
                Values=zeros(N_a*N_z,1);
                if l_d==0
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    end
                else % l_d>0
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                    end
                end
            else
                Values=zeros(N_a*N_z,1);
                if l_d==0
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                    Values=zeros(N_a*N_z,1);
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    end
                else % l_d>0
                    FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names));
                    for ii=1:N_a*N_z
                        j1=rem(ii-1,N_a)+1;
                        j2=ceil(ii/N_a);
                        Values(ii)=FnsToEvaluate{kk}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                    end
                end
            end
            
            RestrictedStationaryDistVec=StationaryDistVec;
            RestrictedStationaryDistVec(Values==0)=0; % Drop all those that don't meet the restriction
            restrictedsamplemass=sum(RestrictedStationaryDistVec);
            RestrictedStationaryDistVec=RestrictedStationaryDistVec/restrictedsamplemass; % Normalize to mass one

            if restrictedsamplemass==0
                warning('One of the conditional restrictions evaluates to a zero mass')
                fprintf(['Specifically, the restriction called ',CondlRestnFnNames{kk},' has a restricted sample that is of zero mass \n'])
                AllStats.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Just return this and hopefully it is clear to the user
            else
                AllStats.(CondlRestnFnNames{kk})=EvalFnOnAgentDist_AllStats_Case1(RestrictedStationaryDistVec, PolicyIndexes, FnsToEvaluate_copy, Parameters, [], n_d, n_a, n_z, d_grid, a_grid, z_grid, simoptions);
                
                % Create some renormalizations where relevant (just the mean)
                for ii=1:length(FnsToEvaluate) %Note FnsToEvaluate alread created above
                    AllStats.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Total=restrictedsamplemass*AllStats.(CondlRestnFnNames{kk}).(FnsToEvalNames{ii}).Mean;
                end
                AllStats.(CondlRestnFnNames{kk}).RestrictedSampleMass=restrictedsamplemass; % Seems likely this would be something user might want
            end
        end
    end
end




end