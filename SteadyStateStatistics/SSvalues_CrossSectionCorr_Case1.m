function SSvalues_CrossSectionCorr=SSvalues_CrossSectionCorr_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters,SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
% Evaluates the cross-sectional correlation for each column of SSvaluesFn
% eg. if you give a SSvaluesFn which is 2x3 (functions) then you will get
% three correlation coefficients (one for the pair of functions that
% constitute each column).

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
    SSvalues_CrossSectionCorr=zeros(length(SSvaluesFn),1,'gpuArray');
    
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
        
        Values1=ValuesOnSSGrid_Case1(SSvaluesFn{1,i}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values1=reshape(Values1,[N_a*N_z,1]);
        Values2=ValuesOnSSGrid_Case1(SSvaluesFn{2,i}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values2=reshape(Values2,[N_a*N_z,1]);
        
        SSvalues_Mean1=sum(Values1.*StationaryDistVec);
        SSvalues_Mean2=sum(Values2.*StationaryDistVec);
        SSvalues_StdDev1=sqrt(sum(StationaryDistVec.*((Values1-SSvalues_Mean1.*ones(N_a*N_z,1)).^2)));
        SSvalues_StdDev2=sqrt(sum(StationaryDistVec.*((Values2-SSvalues_Mean2.*ones(N_a*N_z,1)).^2)));
        
        Numerator=sum((Values1-SSvalues_Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-SSvalues_Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
        SSvalues_CrossSectionCorr(i)=Numerator/(SSvalues_StdDev1*SSvalues_StdDev2);
    end
    
else
    SSvalues_CrossSectionCorr=zeros(length(SSvaluesFn),1);
    d_val=zeros(l_d,1);
    aprime_val=zeros(l_a,1);
    a_val=zeros(l_a,1);
    s_val=zeros(l_z,1);
    
    for i=1:length(SSvaluesFn)
        % Includes check for cases in which no parameters are actually required
        if isempty(SSvalueParamNames) % check for 'SSvalueParamNames={}'
            if l_d==0
                Values1=zeros(N_a,N_z);
                Values2=zeros(N_a,N_z);
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
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        [aprime_ind]=PolicyIndexes(:,j1,j2);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values1(j1,j2)=SSvaluesFn{1,i}(aprime_val,a_val,s_val);
                        Values2(j1,j2)=SSvaluesFn{2,i}(aprime_val,a_val,s_val);
                    end
                end
            else
                Values1=zeros(N_a,N_z);
                Values2=zeros(N_a,N_z);
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
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        d_ind=PolicyIndexes(1:l_d,j1,j2);
                        aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
                        for kk1=1:l_d
                            if kk1==1
                                d_val(kk1)=d_grid(d_ind(kk1));
                            else
                                d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                            end
                        end
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values1(j1,j2)=SSvaluesFn{1,i}(d_val,aprime_val,a_val,s_val);
                        Values2(j1,j2)=SSvaluesFn{2,i}(d_val,aprime_val,a_val,s_val);
                    end
                end
            end
        else
            SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);
            if l_d==0
                Values1=zeros(N_a,N_z);
                Values2=zeros(N_a,N_z);
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
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        [aprime_ind]=PolicyIndexes(:,j1,j2);
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values1(j1,j2)=SSvaluesFn{1,i}(aprime_val,a_val,s_val,SSvalueParamsVec);
                        Values2(j1,j2)=SSvaluesFn{2,i}(aprime_val,a_val,s_val,SSvalueParamsVec);
                    end
                end
            else
                Values1=zeros(N_a,N_z);
                Values2=zeros(N_a,N_z);
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
                                s_val(jj2)=z_grid(s_ind(jj2));
                            else
                                s_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                            end
                        end
                        d_ind=PolicyIndexes(1:l_d,j1,j2);
                        aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
                        for kk1=1:l_d
                            if kk1==1
                                d_val(kk1)=d_grid(d_ind(kk1));
                            else
                                d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                            end
                        end
                        for kk2=1:l_a
                            if kk2==1
                                aprime_val(kk2)=a_grid(aprime_ind(kk2));
                            else
                                aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                            end
                        end
                        Values1(j1,j2)=SSvaluesFn{1,i}(d_val,aprime_val,a_val,s_val,SSvalueParamsVec);
                        Values2(j1,j2)=SSvaluesFn{2,i}(d_val,aprime_val,a_val,s_val,SSvalueParamsVec);
                    end
                end
            end
        end
        
    end
    Values1=reshape(Values1,[N_a*N_z,1]);
    Values2=reshape(Values2,[N_a*N_z,1]);
    
    SSvalues_Mean1=sum(Values1.*StationaryDistVec);
    SSvalues_Mean2=sum(Values2.*StationaryDistVec);
    SSvalues_StdDev1=sqrt(sum(StationaryDistVec.*((Values1-SSvalues_Mean1.*ones(N_a*N_z,1)).^2)));
    SSvalues_StdDev2=sqrt(sum(StationaryDistVec.*((Values2-SSvalues_Mean2.*ones(N_a*N_z,1)).^2)));
    
    Numerator=sum((Values1-SSvalues_Mean1*ones(N_a*N_z,1,'gpuArray')).*(Values2-SSvalues_Mean2*ones(N_a*N_z,1,'gpuArray')).*StationaryDistVec);
    SSvalues_CrossSectionCorr(i)=Numerator/(SSvalues_StdDev1*SSvalues_StdDev2);
end

end