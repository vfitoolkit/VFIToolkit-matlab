function MeanMedianStdDev=EvalFnOnAgentDist_MeanMedianStdDev_Case1(StationaryDist, PolicyIndexes, FnsToEvaluate, Parameters,FnsToEvaluateParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
% Evaluates the mean value (weighted sum/integral), median value, and standard deviation for each element of SSvaluesFn

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
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3,'gpuArray'); % 3 columns: Mean, Median, and Standard Deviation
    
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(FnsToEvaluate)
        % Includes check for cases in which no parameters are actually required
        if isempty(FnsToEvaluateParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
            FnToEvaluateParamsVec=[];
        else
            FnToEvaluateParamsVec=CreateVectorFromParams(Parameters,FnsToEvaluateParamNames(i).Names);
        end
        Values=EvalFnOnAgentDist_Grid_Case1(FnsToEvaluate{i}, FnToEvaluateParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        % Mean
        MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        median_index=find(cumsum(SortedStationaryDistVec)>=0.5,1,'first');
        MeanMedianStdDev(i,2)=SortedValues(median_index);
        % SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
        % Standard Deviation
        MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
    end
else
    MeanMedianStdDev=zeros(length(FnsToEvaluate),3); % 3 columns: Mean, Median, and Standard Deviation
%     d_val=zeros(1,l_d);
%     aprime_val=zeros(1,l_a);
%     a_val=zeros(1,l_a);
%     z_val=zeros(1,l_z);
    StationaryDistVec=reshape(StationaryDist,[N_a*N_z,1]);
    
    [d_gridvals, aprime_gridvals, a_gridvals, z_gridvals]=CreateGridvals(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid,z_grid,1,2);
    
    for i=1:length(FnsToEvaluate)
%         % Includes check for cases in which no parameters are actually required
%         if isempty(FnsToEvaluateParamNames(i).Names) % check for 'SSvalueParamNames={}'
%             Values=zeros(N_a,N_z);
%             if l_d==0
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         z_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 z_val(jj2)=z_grid(z_ind(jj2));
%                             else
%                                 z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         [aprime_ind]=PolicyIndexes(:,j1,j2);
%                         
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(aprime_val,a_val,z_val);
%                         
%                     end
%                 end
%             else
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         z_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 z_val(jj2)=z_grid(z_ind(jj2));
%                             else
%                                 z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_ind=PolicyIndexes(1:l_d,j1,j2);
%                         aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
%                         for kk1=1:l_d
%                             if kk1==1
%                                 d_val(kk1)=d_grid(d_ind(kk1));
%                             else
%                                 d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                             end
%                         end
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
% 
%                         Values(j1,j2)=FnsToEvaluate{i}(d_val,aprime_val,a_val,z_val);
%                     end
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
%             if l_d==0
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_s
%                         z_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 z_val(jj2)=z_grid(z_ind(jj2));
%                             else
%                                 z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         [aprime_ind]=PolicyIndexes(:,j1,j2);
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(aprime_val,a_val,z_val,FnToEvaluateParamsVec);
%                     end
%                 end
%             else
%                 for j1=1:N_a
%                     a_ind=ind2sub_homemade_gpu([n_a],j1);
%                     for jj1=1:l_a
%                         if jj1==1
%                             a_val(jj1)=a_grid(a_ind(jj1));
%                         else
%                             a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%                         end
%                     end
%                     for j2=1:N_z
%                         z_ind=ind2sub_homemade_gpu([n_z],j2);
%                         for jj2=1:l_z
%                             if jj2==1
%                                 z_val(jj2)=z_grid(z_ind(jj2));
%                             else
%                                 z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
%                             end
%                         end
%                         d_ind=PolicyIndexes(1:l_d,j1,j2);
%                         aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
%                         for kk1=1:l_d
%                             if kk1==1
%                                 d_val(kk1)=d_grid(d_ind(kk1));
%                             else
%                                 d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
%                             end
%                         end
%                         for kk2=1:l_a
%                             if kk2==1
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2));
%                             else
%                                 aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
%                             end
%                         end
%                         Values(j1,j2)=FnsToEvaluate{i}(d_val,aprime_val,a_val,z_val,FnToEvaluateParamsVec(:));
%                     end
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
%         end

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
        
        % Mean
        MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
        % Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedStationaryDistVec=StationaryDistVec(SortedValues_index);
        MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedStationaryDistVec)>0.5));
        % Standard Deviation
        MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
    end
    
end


