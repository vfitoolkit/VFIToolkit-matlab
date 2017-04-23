function SSvalues_MeanMedianStdDev=SSvalues_MeanMedianStdDev_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters, SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)

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
    
    SSvalues_MeanMedianStdDev=zeros(length(SSvaluesFn),3, 'gpuArray'); % 3 columns: Mean, Median, and Standard Deviation
        
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(SSvaluesFn)
        % Includes check for cases in which no parameters are actually required
        if isempty(SSvalueParamNames)% || strcmp(SSvalueParamNames(1),'')) % check for 'SSvalueParamNames={}'
            SSvalueParamsVec=[];
        else
            SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names);
        end
        
        Values=ValuesOnSSGrid_Case1(SSvaluesFn{i}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        
        %Mean
        SSvalues_MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
        
        %Median
        [SortedValues,SortedValues_index] = sort(Values);
        SortedSteadyStateDistVec=StationaryDistVec(SortedValues_index);
        
        SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedSteadyStateDistVec)>0.5));
        
        %Standard Deviation
        SSvalues_MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-SSvalues_MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
    end
    
else
    SSvalues_MeanMedianStdDev=zeros(length(SSvaluesFn),3); % 3 columns: Mean, Median, and Standard Deviation
    d_val=zeros(l_d,1);
    aprime_val=zeros(l_a,1);
    a_val=zeros(l_a,1);
    s_val=zeros(l_z,1);
    
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d+l_a,N_a,N_z]);    
    
    for i=1:length(SSvaluesFn)
        % Includes check for cases in which no parameters are actually required
        if isempty(SSvalueParamNames) % check for 'SSvalueParamNames={}'
            if l_d==0
                Values=zeros(N_a,N_z);
                for j1=1:N_a
                    a_ind=ind2sub_homemade([n_a],j1);
                    for jj1=1:l_a
                        if jj1==1
                            a_val(jj1)=a_grid(a_ind(jj1));
                        else
                            a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                        end
                    end
                    for j2=1:N_z
                        s_ind=ind2sub_homemade([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values(j1,j2)=SSvaluesFn{i}(aprime_val,a_val,s_val);
                    end
                end
                
                Values=reshape(Values,[N_a*N_z,1]);
                
                %Mean
                SSvalues_MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
                
                %Median
                [SortedValues,SortedValues_index] = sort(Values);
                SortedSteadyStateDistVec=StationaryDistVec(SortedValues_index);
                
                SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedSteadyStateDistVec)>0.5));
                
                %Standard Deviation
                SSvalues_MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-SSvalues_MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
            else
                Values=zeros(N_a,N_z);
                for j1=1:N_a
                    a_ind=ind2sub_homemade([n_a],j1);
                    for jj1=1:l_a
                        if jj1==1
                            a_val(jj1)=a_grid(a_ind(jj1));
                        else
                            a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                        end
                    end
                    for j2=1:N_z
                        s_ind=ind2sub_homemade([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        d_ind=PolicyIndexesKron(1:l_d,j1,j2);
                        for kk=1:l_d
                            if kk==1
                                d_val(kk)=d_grid(d_ind(kk));
                            else
                                d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                            end
                        end
                        aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,s_val);
                    end
                end
                
                Values=reshape(Values,[N_a*N_z,1]);
                
                %Mean
                SSvalues_MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
                
                %Median
                [SortedValues,SortedValues_index] = sort(Values);
                SortedSteadyStateDistVec=StationaryDistVec(SortedValues_index);
                
                SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedSteadyStateDistVec)>0.5));
                
                %Standard Deviation
                SSvalues_MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-SSvalues_MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
            end
        else
            SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);
            if l_d==0
                Values=zeros(N_a,N_z);
                for j1=1:N_a
                    a_ind=ind2sub_homemade([n_a],j1);
                    for jj1=1:l_a
                        if jj1==1
                            a_val(jj1)=a_grid(a_ind(jj1));
                        else
                            a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                        end
                    end
                    for j2=1:N_z
                        s_ind=ind2sub_homemade([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values(j1,j2)=SSvaluesFn{i}(aprime_val,a_val,s_val,SSvalueParamsVec);
                    end
                end
                
                Values=reshape(Values,[N_a*N_z,1]);
                
                %Mean
                SSvalues_MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
                
                %Median
                [SortedValues,SortedValues_index] = sort(Values);
                SortedSteadyStateDistVec=StationaryDistVec(SortedValues_index);
                
                SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedSteadyStateDistVec)>0.5));
                
                %Standard Deviation
                SSvalues_MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-SSvalues_MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
            else
                Values=zeros(N_a,N_z);
                for j1=1:N_a
                    a_ind=ind2sub_homemade([n_a],j1);
                    for jj1=1:l_a
                        if jj1==1
                            a_val(jj1)=a_grid(a_ind(jj1));
                        else
                            a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                        end
                    end
                    for j2=1:N_z
                        s_ind=ind2sub_homemade([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        d_ind=PolicyIndexesKron(1:l_d,j1,j2);
                        for kk=1:l_d
                            if kk==1
                                d_val(kk)=d_grid(d_ind(kk));
                            else
                                d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
                            end
                        end
                        aprime_ind=PolicyIndexesKron(l_d+1:l_d+l_a,j1,j2);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,s_val,SSvalueParamsVec);
                    end
                end
                
                Values=reshape(Values,[N_a*N_z,1]);
                
                %Mean
                SSvalues_MeanMedianStdDev(i,1)=sum(Values.*StationaryDistVec);
                
                %Median
                [SortedValues,SortedValues_index] = sort(Values);
                SortedSteadyStateDistVec=StationaryDistVec(SortedValues_index);
                
                SSvalues_MeanMedianStdDev(i,2)=min(SortedValues(cumsum(SortedSteadyStateDistVec)>0.5));
                
                %Standard Deviation
                SSvalues_MeanMedianStdDev(i,3)=sqrt(sum(StationaryDistVec.*((Values-SSvalues_MeanMedianStdDev(i,1).*ones(N_a*N_z,1)).^2)));
            end
        end
    end
    
end

