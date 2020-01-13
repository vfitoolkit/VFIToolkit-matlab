function LorenzCurve=EvalFnOnAgentDist_LorenzCurve_FHorz_Case2(StationaryDist,PolicyIndexes, FnsToEvaluate, Parameters, FnsToEvaluateParamNames,n_d,n_a,n_z,N_j,d_grid,a_grid,z_grid,Parallel,npoints)
% Returns a Lorenz Curve 100-by-1 that contains all of the quantiles from 1
% to 100. Unless the optional npoints input is used in which case it will be
% npoints-by-1.
% 
% Note that to unnormalize the Lorenz Curve you can just multiply it be the
% AggVars for the same variable. This will give you the inverse cdf.

if isa(StationaryDist,'struct')
    % Using Age Dependent Grids so send there
    fprintf('ERROR: EvalFnOnAgentDist_LorenzCurve_FHorz_Case2() does no yet allow for Age Dependent Grids \n')
    fprintf('IF YOU HAVE A need/use for this functionality, please contact me: robertdkirkby@gmail.com and I will implement it \n')
    dbstack
    return
end

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

if Parallel==2
%     AggVars=zeros(length(FnsToEvaluateFn),1,'gpuArray');
    LorenzCurve=zeros(npoints,length(FnsToEvaluate),'gpuArray');
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
        
    PolicyValues=PolicyInd2Val_FHorz_Case2(PolicyIndexes,n_d,n_a,n_z,N_j,d_grid,2);
    permuteindexes=[1+(1:1:(l_a+l_z)),1,1+l_a+l_z+1];    
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_z,l_d,N_d]

    PolicyValuesPermuteVec=reshape(PolicyValuesPermute,[N_a*N_z*l_d,N_j]);
    for i=1:length(FnsToEvaluate)
        Values=nan(N_a*N_z,N_j,'gpuArray');
        for jj=1:N_j
            % Includes check for cases in which no parameters are actually required
            if isempty(FnsToEvaluateParamNames) %|| strcmp(SSvalueParamNames(i).Names(1),'')) % check for 'SSvalueParamNames={} or SSvalueParamNames={''}'
                FnToEvaluateParamsVec=[];
            else
                FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
            end
            Values(:,jj)=reshape(ValuesOnSSGrid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsVec,reshape(PolicyValuesPermuteVec(:,jj),[n_a,n_z,l_d]),n_d,n_a,n_z,a_grid,z_grid,2),[N_a*N_z,1]);
        end

        Values=reshape(Values,[N_a*N_z*N_j,1]);
        
        WeightedValues=Values.*StationaryDistVec;
        WeightedValues(isnan(WeightedValues))=0; % Values of -Inf times weight of zero give nan, we want them to be zeros.
        %     AggVars(i)=sum(WeightedValues);
        
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
%         %         % I now also get rid of all of those points after the cdf reaches
%         %         % 1-10^(-9). This is just because otherwise rounding in the ~12th
%         %         % decimal place was causing problems with vector not being
%         %         % 'sorted'.
%         %         firstIndex = find((CumSumSortedSteadyStateDistVec_NoDuplicates-1+10^(-9))>0,1,'first');
%         %         CumSumSortedSteadyStateDistVec_NoDuplicates=CumSumSortedSteadyStateDistVec_NoDuplicates(1:firstIndex);
%         %         CumSumSortedWeightedValues_NoDuplicates=CumSumSortedWeightedValues_NoDuplicates(1:firstIndex);
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
        LorenzCurve(:,i)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedStationaryDistVec,npoints); 
    end
else
%     AggVars=zeros(1,length(FnsToEvaluate));
    LorenzCurve=zeros(npoints,length(FnsToEvaluate));

    StationaryDistVec=reshape(StationaryDist,[N_a*N_z*N_j,1]);
    
    sizePolicyIndexes=size(PolicyIndexes);
    if sizePolicyIndexes(2:end)~=[N_a,N_z,N_j] % If not in vectorized form
        PolicyIndexes=reshape(PolicyIndexes,[sizePolicyIndexes(1),N_a,N_z,N_j]);
    end
    
    a_gridvals=CreateGridvals(n_a,a_grid,1);
    z_gridvals=CreateGridvals(n_z,z_grid,1);
    dPolicy_gridvals=zeros(N_a*N_z,N_j);
    for jj=1:N_j
        dPolicy_gridvals(:,jj)=CreateGridvals_Policy(PolicyIndexes(:,:,jj),n_d,[],n_a,n_z,d_grid,[],2,1);
    end
    
    for i=1:length(FnsToEvaluate)
        Values=zeros(N_a,N_z,N_j);
        
        for a_c=1:N_a
            a_val=a_gridvals(a_c,:);
            %                 a_ind=ind2sub_homemade_gpu([n_a],a_c);
            %                 for jj1=1:l_a
            %                     if jj1==1
            %                         a_val(jj1)=a_grid(a_ind(jj1));
            %                     else
            %                         a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
            %                     end
            %                 end
            for z_c=1:N_z
                z_val=z_gridvals(z_c,:);
                %                     s_ind=ind2sub_homemade_gpu([n_z],z_c);
                %                     for jj2=1:l_z
                %                         if jj2==1
                %                             z_val(jj2)=z_grid(s_ind(jj2));
                %                         else
                %                             z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                %                         end
                %                     end
                az_c=sub2ind_homemade([N_a,N_z],[a_c,z_c]);
                for jj=1:N_j
                    d_val=dPolicy_gridvals(az_c,jj);
%                     [aprime_ind]=PolicyIndexes(:,a_c,z_c,jj);
%                     for kk2=1:l_a
%                         if kk2==1
%                             aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                         else
%                             aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                         end
%                     end
                    % Includes check for cases in which no parameters are actually required
                    if isempty(FnsToEvaluateParamNames(i).Names)
                        tempv=[d_val,a_val,z_val];
                        tempcell=cell(1,length(tempv));
                        for temp_c=1:length(tempv)
                            tempcell{temp_c}=tempv(temp_c);
                        end
                    else
                        FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names,jj);
                        tempv=[d_val,a_val,z_val,FnToEvaluateParamsVec];
                        tempcell=cell(1,length(tempv));
                        for temp_c=1:length(tempv)
                            tempcell{temp_c}=tempv(temp_c);
                        end
                    end
                    Values(a_c,z_c,jj)=FnsToEvaluate{i}(tempcell{:});
                end
            end
        end

        Values=reshape(Values,[N_a*N_z*N_j,1]);
        
        WeightedValues=Values.*StationaryDistVec;
%         AggVars(i)=sum(WeightedValues);
        
        
        [~,SortedValues_index] = sort(Values);
        
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        SortedWeightedValues=WeightedValues(SortedValues_index);
        
        CumSumSortedStationaryDistVec=cumsum(SortedStationaryDistVec);
        
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
        LorenzCurve(:,i)=LorenzCurve_subfunction_PreSorted(SortedWeightedValues,CumSumSortedStationaryDistVec,npoints);
    end
    
end


end