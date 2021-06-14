function LorenzCurve=EvalFnOnAgentDist_LorenzCurve_Case2(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, evalfnoptions)
% Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1 to 100
%
% Note that to unnormalize the Lorenz Curve you can just multiply it be the
% AggVars for the same variable. This will give you the inverse cdf.
%
% evalfnoptions.parallel and evalfnoptions.npoints are optional inputs

if exist('evalfnoptions','var')==0
    evalfnoptions.parallel=1+(gpuDeviceCount>0);
    evalfnoptions.npoints=100;
else
    %Check evalfnoptions for missing fields, if there are some fill them with the defaults
    if isfield(evalfnoptions,'parallel')==0
        evalfnoptions.parallel=1+(gpuDeviceCount>0);
    end
    if isfield(evalfnoptions,'npoints')==0
        evalfnoptions.npoints=100;
    end
end

if isfield(evalfnoptions,'exclude')==1
    exclude=1;
else
    exclude=0;
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if evalfnoptions.parallel==2
    StationaryDistVec=gpuArray(StationaryDistVec);
    PolicyIndexes=gpuArray(PolicyIndexes);

    
    PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid,evalfnoptions.parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d]
    
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    LorenzCurve=zeros(length(FnsToEvaluate),evalfnoptions.npoints,'gpuArray');
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        Values=EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,evalfnoptions.parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
        if exclude==1 % Exclude by setting the mass of agents to zero
            if strcmp(evalfnoptions.exclude.condn,'equal')
                StationaryDistVec(Values==evalfnoptions.exclude.values)=0;
            elseif strcmp(evalfnoptions.exclude.condn,'greaterthan')
                StationaryDistVec(Values>evalfnoptions.exclude.values)=0;
            elseif strcmp(evalfnoptions.exclude.condn,'greaterorequal')
                StationaryDistVec(Values>=evalfnoptions.exclude.values)=0;
            elseif strcmp(evalfnoptions.exclude.condn,'lessthan')
                StationaryDistVec(Values<evalfnoptions.exclude.values)=0;
            elseif strcmp(evalfnoptions.exclude.condn,'lessorequal')
                StationaryDistVec(Values<=evalfnoptions.exclude.values)=0;
            end
            StationaryDistVec=StationaryDistVec/sum(StationaryDistVec); % Renormalize the part of the StationaryDist that remains after satisfying the exclusion condition.
        end
        
        WeightedValues=Values.*StationaryDistVec;
        WeightedValues(isnan(WeightedValues))=0; % Values of -Inf times weight of zero give nan, we want them to be zeros.
        AggVars(i)=sum(WeightedValues);
        
        [~,SortedValues_index] = sort(Values);
        
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedStationaryDistVec=cumsum(SortedStationaryDistVec);
%         
%         %We now want to use interpolation, but this won't work unless all
%         %values in are CumSumSortedSteadyStateDist distinct. So we now remove
%         %any duplicates (ie. points of zero probability mass/density). We then
%         %have to remove the corresponding points of SortedValues
% %         [~,UniqueIndex] = unique(CumSumSortedStationaryDistVec,'first');
%         [~,UniqueIndex] = uniquetol(CumSumSortedStationaryDistVec); % uses a default tolerance of 1e-6 for single-precision inputs and 1e-12 for double-precision inputs
%         CumSumSortedStationaryDistVec_NoDuplicates=CumSumSortedStationaryDistVec(sort(UniqueIndex));
%         SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
%         CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
%         
% 
%         InverseCDF_xgrid=gpuArray(1/evalfnoptions.npoints:1/evalfnoptions.npoints:1);
%         
%         InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%         % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%         % have already sorted and removed duplicates this will just be the last
%         % point so we can just grab it directly.
%         InverseCDF_SSvalues(evalfnoptions.npoints)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
%         % interp1 may have similar problems at the bottom of the cdf
%         j=1; %use j to figure how many points with this problem
%         while InverseCDF_xgrid(j)<CumSumSortedStationaryDistVec_NoDuplicates(1)
%             j=j+1;
%         end
%         for jj=1:j-1 %divide evenly through these states (they are all identical)
%             InverseCDF_SSvalues(jj)=(jj/j)*InverseCDF_SSvalues(j);
%         end
%         
%         SSvalues_LorenzCurve(i,:)=InverseCDF_SSvalues./SSvalues_AggVars(i);
        LorenzCurve(i,:)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedStationaryDistVec,evalfnoptions.npoints,evalfnoptions.parallel)';

    end
else
    StationaryDistVec=gather(StationaryDistVec);
    PolicyIndexes=gather(PolicyIndexes);

    AggVars=zeros(length(FnsToEvaluate),1);
    LorenzCurve=zeros(length(FnsToEvaluate),evalfnoptions.npoints);
%     d_val=zeros(l_d,1);
%     a_val=zeros(l_a,1);
%     z_val=zeros(l_z,1);

    [d_gridvals, ~]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'FnsToEvaluateParamNames={}'
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
            end
        else
            Values=zeros(N_a*N_z,1);
            FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
            for ii=1:N_a*N_z
%                 j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
            end
        end
        
        WeightedValues=Values.*StationaryDistVec;
        AggVars(i)=sum(WeightedValues);
        
        
        [~,SortedValues_index] = sort(Values);
        
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedStationaryDistVec=cumsum(SortedStationaryDistVec);
%         
%         %We now want to use interpolation, but this won't work unless all
%         %values in are CumSumSortedSteadyStateDist distinct. So we now remove
%         %any duplicates (ie. points of zero probability mass/density). We then
%         %have to remove the corresponding points of SortedValues
%         [~,UniqueIndex] = unique(CumSumSortedStationaryDistVec,'first');
%         CumSumSortedStationaryDistVec_NoDuplicates=CumSumSortedStationaryDistVec(sort(UniqueIndex));
%         SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
%         
%         CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
%         
%         InverseCDF_xgrid=1/evalfnoptions.npoints:1/evalfnoptions.npoints:1;
%         
%         InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%         % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%         % have already sorted and removed duplicates this will just be the last
%         % point so we can just grab it directly.
%         InverseCDF_SSvalues(evalfnoptions.npoints)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
%         % interp1 may have similar problems at the bottom of the cdf
%         j=1; %use j to figure how many points with this problem
%         while InverseCDF_xgrid(j)<CumSumSortedStationaryDistVec_NoDuplicates(1)
%             j=j+1;
%         end
%         for jj=1:j-1 %divide evenly through these states (they are all identical)
%             InverseCDF_SSvalues(jj)=(jj/j)*InverseCDF_SSvalues(j);
%         end
%         
%         SSvalues_LorenzCurve(i,:)=InverseCDF_SSvalues./SSvalues_AggVars(i);
        LorenzCurve(i,:)=gather(LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedStationaryDistVec,evalfnoptions.npoints,evalfnoptions.parallel)');
    end
end




end

