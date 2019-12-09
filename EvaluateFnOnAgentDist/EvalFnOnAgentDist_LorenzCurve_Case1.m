function LorenzCurve=EvalFnOnAgentDist_LorenzCurve_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel,npoints)
% Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1
% to 100. Unless the optional npoints input is used in which case it will be
% npoints-by-1.
%
% Note that to unnormalize the Lorenz Curve you can just multiply it be the
% AggVars for the same variable. This will give you the inverse cdf.

if exist('npoints','var')==0
    npoints=100;
end

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if Parallel==2
    StationaryDistVec=gpuArray(StationaryDistVec);
    PolicyIndexes=gpuArray(PolicyIndexes);
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    LorenzCurve=zeros(length(FnsToEvaluate),npoints,'gpuArray');
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
        WeightedValues=Values.*StationaryDistVec;
        WeightedValues(isnan(WeightedValues))=0; % Values of -Inf times weight of zero give nan, we want them to be zeros.
        AggVars(i)=sum(WeightedValues);
        
        
        [~,SortedValues_index] = sort(Values);
        
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedStationaryDistVec=cumsum(SortedStationaryDistVec);
        
%         %We now want to use interpolation, but this won't work unless all
%         %values in are CumSumSortedSteadyStateDist distinct. So we now remove
%         %any duplicates (ie. points of zero probability mass/density). We then
%         %have to remove the corresponding points of SortedValues. Since we
%         %are just looking for 100 points to make up our cdf I round all
%         %variables to 5 decimal points before checking for uniqueness (Do
%         %this because otherwise rounding in the ~12th decimal place was causing
%         % problems with vector not being sorted as strictly increasing.
%         [~,UniqueIndex] = unique(floor(CumSumSortedStationaryDistVec*10^5),'first');
%         CumSumSortedStationaryDistVec_NoDuplicates=CumSumSortedStationaryDistVec(sort(UniqueIndex));
%         SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
%         
%         CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
%                  
%         InverseCDF_xgrid=gpuArray(1/npoints:1/npoints:1);
%         
%         
%         InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%         % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%         % have already sorted and removed duplicates this will just be the last
%         % point so we can just grab it directly.
%         %         InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(end);
%         InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(end);
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
        LorenzCurve(i,:)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedStationaryDistVec,npoints)';
    end
    
else
    StationaryDistVec=gather(StationaryDistVec);
    PolicyIndexes=gather(PolicyIndexes);

    AggVars=zeros(length(FnsToEvaluate),1);
    LorenzCurve=zeros(length(FnsToEvaluate),npoints);
    
%     [d_gridvals, aprime_gridvals, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,1,2);
    [d_gridvals, aprime_gridvals]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,1, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'FnsToEvaluateParamNames={}'
            Values=zeros(N_a*N_z,1);
            if l_d==0
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            else % l_d>0
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
                    %                     a_val=a_gridvals{j1,:};
                    %                     z_val=z_gridvals{j2,:};
                    %                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
                    %                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
                    %                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
                end
            end
        else
            Values=zeros(N_a*N_z,1);
            if l_d==0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                Values=zeros(N_a*N_z,1);
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=FnsToEvaluate{i}(aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
            else % l_d>0
                FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
                for ii=1:N_a*N_z
                    %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                    j1=rem(ii-1,N_a)+1;
                    j2=ceil(ii/N_a);
%                     a_val=a_gridvals{j1,:};
%                     z_val=z_gridvals{j2,:};
%                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,SSvalueParamsVec);
                    Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},aprime_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
                end
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
%         %have to remove the corresponding points of SortedValues. 
%         [~,UniqueIndex] = uniquetol(CumSumSortedStationaryDistVec); % uses a default tolerance of 1e-6 for single-precision inputs and 1e-12 for double-precision inputs
% 
%         CumSumSortedStationaryDistVec_NoDuplicates=CumSumSortedStationaryDistVec(sort(UniqueIndex));
%         SortedWeightedValues_NoDuplicates=SortedWeightedValues(sort(UniqueIndex));
%         
%         CumSumSortedWeightedValues_NoDuplicates=cumsum(SortedWeightedValues_NoDuplicates);
%         
%         
%         InverseCDF_xgrid=1/npoints:1/npoints:1;
%         
%         InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%         % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%         % have already sorted and removed duplicates this will just be the last
%         % point so we can just grab it directly.
%         %         InverseCDF_SSvalues(100)=CumSumSortedWeightedValues_NoDuplicates(end);
%         InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(end);
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
        LorenzCurve(i,:)=gather(LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedStationaryDistVec,npoints)');
    end
end


end

