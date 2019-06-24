function LorenzCurve=EvalFnOnAgentDist_LorenzCurve_Case2(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel, npoints)
% Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1 to 100
%
% Note that to unnormalize the Lorenz Curve you can just multiply it be the
% AggVars for the same variable. This will give you the inverse cdf.

if nargin<13
    npoints=100;
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);

if Parallel==2
    PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid,Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    AggVars=zeros(length(FnsToEvaluate),1,'gpuArray');
    LorenzCurve=zeros(length(FnsToEvaluate),npoints,'gpuArray');
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names)  % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        Values=EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
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
%         InverseCDF_xgrid=gpuArray(1/npoints:1/npoints:1);
%         
%         InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%         % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%         % have already sorted and removed duplicates this will just be the last
%         % point so we can just grab it directly.
%         InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
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
    AggVars=zeros(length(FnsToEvaluate),1);
    LorenzCurve=zeros(length(FnsToEvaluate),npoints);
    d_val=zeros(l_d,1);
    a_val=zeros(l_a,1);
    z_val=zeros(l_z,1);
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
            Values=zeros(N_a,N_z);
            for j1=1:N_a
                a_ind=ind2sub_homemade_gpu([n_a],j1);
                for jj1=1:l_a
                    if jj1==1
                        a_val(jj1)=a_grid(a_ind(jj1));
                    else
                        a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                    end
                end
                for j2=1:N_z
                    s_ind=ind2sub_homemade_gpu([n_z],j2);
                    for jj2=1:l_z
                        if jj2==1
                            z_val(jj2)=z_grid(s_ind(jj2));
                        else
                            z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                        end
                    end
                    d_ind=PolicyIndexes(1:l_d,j1,j2);
                    for kk1=1:l_d
                        if kk1==1
                            d_val(kk1)=d_grid(d_ind(kk1));
                        else
                            d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                        end
                    end
                    Values(j1,j2)=FnsToEvaluate{i}(d_val,a_val,z_val);
                end
            end
            Values=reshape(Values,[N_a*N_z,1]);
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
            Values=zeros(N_a,N_z);
            for j1=1:N_a
                a_ind=ind2sub_homemade_gpu([n_a],j1);
                for jj1=1:l_a
                    if jj1==1
                        a_val(jj1)=a_grid(a_ind(jj1));
                    else
                        a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                    end
                end
                for j2=1:N_z
                    s_ind=ind2sub_homemade_gpu([n_z],j2);
                    for jj2=1:l_z
                        if jj2==1
                            z_val(jj2)=z_grid(s_ind(jj2));
                        else
                            z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                        end
                    end
                    d_ind=PolicyIndexes(1:l_d,j1,j2);
                    for kk1=1:l_d
                        if kk1==1
                            d_val(kk1)=d_grid(d_ind(kk1));
                        else
                            d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                        end
                    end
                    Values(j1,j2)=FnsToEvaluate{i}(d_val,a_val,z_val,FnToEvaluateParamsVec);
                end
            end
            Values=reshape(Values,[N_a*N_z,1]);
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
%         InverseCDF_xgrid=1/npoints:1/npoints:1;
%         
%         InverseCDF_SSvalues=interp1(CumSumSortedStationaryDistVec_NoDuplicates,CumSumSortedWeightedValues_NoDuplicates, InverseCDF_xgrid);
%         % interp1 cannot work for the point of InverseCDF_xgrid=1 (gives NaN). Since we
%         % have already sorted and removed duplicates this will just be the last
%         % point so we can just grab it directly.
%         InverseCDF_SSvalues(npoints)=CumSumSortedWeightedValues_NoDuplicates(length(CumSumSortedWeightedValues_NoDuplicates));
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
end




end

