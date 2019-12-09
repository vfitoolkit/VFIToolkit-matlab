function MeanMedianStdDev=EvalFnOnAgentDist_MeanMedianStdDev_Case2(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid,Parallel)
% Evaluates the mean value (weighted sum/integral), median value, and standard deviation for each element of FnsToEvaluate

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3,'gpuArray'); % 3 columns: Mean, Median, and Standard Deviation

    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    PolicyValues=PolicyInd2Val_Case2(PolicyIndexes,n_d,n_a,n_z,d_grid,Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        Values=EvalFnOnAgentDist_Grid_Case2(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        % Mean
        MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        SortedValues(median_index)
        MeanMedianStdDev(i,2)=SortedValues(median_index);
        % Standard Deviation
        MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
    end
else
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3); % 3 columns: Mean, Median, and Standard Deviation
%     d_val=zeros(l_d,1);
%     a_val=zeros(l_a,1);
%     z_val=zeros(l_z,1);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
%     [d_gridvals, ~, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,2,2);
    [d_gridvals, ~]=CreateGridvals_Policy(PolicyIndexes,n_d,n_a,n_a,n_z,d_grid,a_grid,2, 2);
    a_gridvals=CreateGridvals(n_a,a_grid,2);
    z_gridvals=CreateGridvals(n_z,z_grid,2);
    
    for i=1:length(FnsToEvaluate)
%         % Includes check for cases in which no parameters are actually required
%         if isempty(FnsToEvaluateParamNames) % check for 'SSvalueParamNames={}'
%             Values=zeros(N_a,N_z);
%             for j1=1:N_a
%                 a_ind=ind2sub_homemade_gpu([n_a],j1);
%                 for jj1=1:l_a
%                     if jj1==1
%                         a_val(jj1)=a_grid(a_ind(jj1));
%                     else
%                         a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                     end
%                 end
%                 for j2=1:N_z
%                     s_ind=ind2sub_homemade_gpu([n_z],j2);
%                     for jj2=1:l_z
%                         if jj2==1
%                             z_val(jj2)=z_grid(s_ind(jj2));
%                         else
%                             z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                         end
%                     end
%                     d_ind=PolicyIndexes(1:l_d,j1,j2);
%                     for kk1=1:l_d
%                         if kk1==1
%                             d_val(kk1)=d_grid(d_ind(kk1));
%                         else
%                             d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                         end
%                     end
%                     Values(j1,j2)=FnsToEvaluate{i}(d_val,a_val,z_val);
%                 end
%             end
%             Values=reshape(Values,[N_a*N_z,1]);
%             
%             % Mean
%             MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
%             % Median
%             [SortedValues,SortedValues_index] = sort(Values);
%             SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
%             MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
%             % Standard Deviation
%             MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
%         else
%             FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
%             Values=zeros(N_a,N_z);
%             for j1=1:N_a
%                 a_ind=ind2sub_homemade_gpu([n_a],j1);
%                 for jj1=1:l_a
%                     if jj1==1
%                         a_val(jj1)=a_grid(a_ind(jj1));
%                     else
%                         a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                     end
%                 end
%                 for j2=1:N_z
%                     s_ind=ind2sub_homemade_gpu([n_z],j2);
%                     for jj2=1:l_z
%                         if jj2==1
%                             z_val(jj2)=z_grid(s_ind(jj2));
%                         else
%                             z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                         end
%                     end
%                     d_ind=PolicyIndexes(1:l_d,j1,j2);
%                     for kk1=1:l_d
%                         if kk1==1
%                             d_val(kk1)=d_grid(d_ind(kk1));
%                         else
%                             d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                         end
%                     end
%                     Values(j1,j2)=FnsToEvaluate{i}(d_val,a_val,z_val,FnToEvaluateParamsVec);
%                 end
%             end
%             Values=reshape(Values,[N_a*N_z,1]);
%             
%             % Mean
%             MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
%             % Median            
%             [SortedValues,SortedValues_index] = sort(Values);
%             SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
%             median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
%             SortedValues(median_index)
%             MeanMedianStdDev(i,2)=SortedValues(median_index);
%             % Standard Deviation
%             MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
%         end
        if isempty(FnsToEvaluateParamNames(i).Names) % check for 'FnsToEvaluateParamNames={}'
            Values=zeros(N_a*N_z,1);
            for ii=1:N_a*N_z
                %        j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
                %                     a_val=a_gridvals{j1,:};
                %                     z_val=z_gridvals{j2,:};
                %                     d_val=d_gridvals{j1+(j2-1)*N_a,:};
                %                     aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
                %                     Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:});
            end
        else
            Values=zeros(N_a*N_z,1);
            FnToEvaluateParamsCell=num2cell(CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names));
            for ii=1:N_a*N_z
%                 j1j2=ind2sub_homemade([N_a,N_z],ii); % Following two lines just do manual implementation of this.
                j1=rem(ii-1,N_a)+1;
                j2=ceil(ii/N_a);
%                 a_val=a_gridvals{j1,:};
%                 z_val=z_gridvals{j2,:};
%                 d_val=d_gridvals{j1+(j2-1)*N_a,:};
%                 aprime_val=aprime_gridvals{j1+(j2-1)*N_a,:};
%                 Values(ii)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,SSvalueParamsVec);
                Values(ii)=FnsToEvaluate{i}(d_gridvals{j1+(j2-1)*N_a,:},a_gridvals{j1,:},z_gridvals{j2,:},FnToEvaluateParamsCell{:});
            end
        end
        
        % Mean
        MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        MeanMedianStdDev(i,2)=SortedValues(median_index);
        % Standard Deviation
        MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
    end
end

end

