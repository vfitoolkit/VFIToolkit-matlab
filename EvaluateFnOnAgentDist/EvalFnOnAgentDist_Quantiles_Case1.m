function varargout=EvalFnOnAgentDist_Quantiles_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, NumQuantiles, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel)
% Returns the cut-off values and the within percentile means from dividing
% the StationaryDist into NumPercentiles percentiles.
%
% Parallel is an optional input

%%
if exist('Parallel','var')==0
    Parallel=1+(gpuDeviceCount>0);
elseif isempty(Parallel)
    Parallel=1+(gpuDeviceCount>0);
end

Tolerance=10^(-12); % Numerical tolerance used when calculating min and max values.

% Note that to unnormalize the Lorenz Curve you can just multiply it be the AggVars for the same variable. This will give you the inverse cdf.

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

%% Implement new way of handling FnsToEvaluate
if isstruct(FnsToEvaluate)
    FnsToEvaluateStruct=1;
    clear FnsToEvaluateParamNames
    AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        temp=getAnonymousFnInputNames(FnsToEvaluate.(AggVarNames{ff}));
        if length(temp)>(l_d+l_a+l_a+l_z)
            FnsToEvaluateParamNames(ff).Names={temp{l_d+l_a+l_a+l_z+1:end}}; % the first inputs will always be (d,aprime,a,z)
        else
            FnsToEvaluateParamNames(ff).Names={};
        end
        FnsToEvaluate2{ff}=FnsToEvaluate.(AggVarNames{ff});
    end    
    FnsToEvaluate=FnsToEvaluate2;
else
    FnsToEvaluateStruct=0;
end

%%
StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if Parallel==2
    d_grid=gpuArray(d_grid);
    a_grid=gpuArray(a_grid);
    z_grid=gpuArray(z_grid);
    QuantileCutOffs=zeros(length(FnsToEvaluate),NumQuantiles+1,'gpuArray'); %Includes min and max
    QuantileMeans=zeros(length(FnsToEvaluate),NumQuantiles,'gpuArray');
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for kk=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(kk).Names);
        end
        
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{kk}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = StationaryDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes_i=zeros(1,NumQuantiles-1,'gpuArray');
        QuantileCutoffs_i=zeros(1,NumQuantiles-1,'gpuArray');
        QuantileMeans_i=zeros(1,NumQuantiles,'gpuArray');
        for ii=1:NumQuantiles-1
            [tempindex,~]=find(CumSumSortedWeights>=ii/NumQuantiles,1,'first');
            QuantileIndexes_i(ii)=tempindex;
            QuantileCutoffs_i(ii)=SortedValues(tempindex);
            if ii==1
                QuantileMeans_i(ii)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<ii) && (ii<(NumQuantiles-1))
                QuantileMeans_i(ii)=sum(SortedWeightedValues(QuantileIndexes_i(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_i(ii-1)));
            elseif ii==(NumQuantiles-1)
                QuantileMeans_i(ii)=sum(SortedWeightedValues(QuantileIndexes_i(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_i(ii-1)));
                QuantileMeans_i(ii+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        [tempindex,~]=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        [tempindex,~]=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        QuantileCutOffs(kk,:)=[minvalue, QuantileCutoffs_i, maxvalue];
        QuantileMeans(kk,:)=QuantileMeans_i;
    end
    
else
    QuantileCutOffs=zeros(length(FnsToEvaluate),NumQuantiles+1); %Includes min and max
    QuantileMeans=zeros(length(FnsToEvaluate),NumQuantiles);
    
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for kk=1:length(FnsToEvaluate)

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
        
        [SortedValues,SortedValues_index] = sort(Values);
        SortedWeights = StationaryDistVec(SortedValues_index);
        
        CumSumSortedWeights=cumsum(SortedWeights);
        
        WeightedValues=Values.*StationaryDistVec;
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        QuantileIndexes_i=zeros(1,NumQuantiles-1);
        QuantileCutoffs_i=zeros(1,NumQuantiles-1);
        QuantileMeans_i=zeros(1,NumQuantiles);
        for ii=1:NumQuantiles-1
            [tempindex,~]=find(CumSumSortedWeights>=ii/NumQuantiles,1,'first');
            QuantileIndexes_i(ii)=tempindex;
            QuantileCutoffs_i(ii)=SortedValues(tempindex);
            if ii==1
                QuantileMeans_i(ii)=sum(SortedWeightedValues(1:tempindex))./CumSumSortedWeights(tempindex); %Could equally use sum(SortedWeights(1:tempindex)) in denominator
            elseif (1<ii) && (ii<(NumQuantiles-1))
                QuantileMeans_i(ii)=sum(SortedWeightedValues(QuantileIndexes_i(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_i(ii-1)));
            elseif ii==(NumQuantiles-1)
                QuantileMeans_i(ii)=sum(SortedWeightedValues(QuantileIndexes_i(ii-1)+1:tempindex))./(CumSumSortedWeights(tempindex)-CumSumSortedWeights(QuantileIndexes_i(ii-1)));
                QuantileMeans_i(ii+1)=sum(SortedWeightedValues(tempindex+1:end))./(CumSumSortedWeights(end)-CumSumSortedWeights(tempindex));
            end
        end
        
        % Min value
        [tempindex,~]=find(CumSumSortedWeights>=Tolerance,1,'first');
        minvalue=SortedValues(tempindex);
        % Max value
        [tempindex,~]=find(CumSumSortedWeights>=(1-Tolerance),1,'first');
        maxvalue=SortedValues(tempindex);
        
        QuantileCutOffs(kk,:)=[minvalue, QuantileCutoffs_i, maxvalue];
        QuantileMeans(kk,:)=QuantileMeans_i;
    end
    
end

%%
if FnsToEvaluateStruct==1
    % Change the output into a structure
    Quantiles=struct();
%     AggVarNames=fieldnames(FnsToEvaluate);
    for ff=1:length(AggVarNames)
        Quantiles.(AggVarNames{ff}).QuantileMeans=QuantileMeans(ff,:);
        Quantiles.(AggVarNames{ff}).QuantileCutOffs=QuantileCutOffs(ff,:);
    end
    
    varargout={Quantiles};
else
    varargout={QuantileCutOffs, QuantileMeans};
end



end

