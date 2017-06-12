function SSvalues_AggVars=SSvalues_AggVars_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters,SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, Parallel)
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

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
    SSvalues_AggVars=zeros(length(SSvaluesFn),1,'gpuArray');
        
    PolicyValues=PolicyInd2Val_Case1(PolicyIndexes,n_d,n_a,n_z,d_grid,a_grid, Parallel);
    permuteindexes=[1+(1:1:(l_a+l_z)),1];
    PolicyValuesPermute=permute(PolicyValues,permuteindexes); %[n_a,n_s,l_d+l_a]
    
    for i=1:length(SSvaluesFn)
        % Includes check for cases in which no parameters are actually required
        if isempty(SSvalueParamNames)  % check for 'SSvalueParamNames={}'
            SSvalueParamsVec=[];
        else
            SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names);
        end
        Values=ValuesOnSSGrid_Case1(SSvaluesFn{i}, SSvalueParamsVec,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel);
        Values=reshape(Values,[N_a*N_z,1]);
        % When evaluating value function (which may sometimes give -Inf
        % values) on StationaryDistVec (which at those points will be
        % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
        temp=Values.*StationaryDistVec;
        SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
    end
else
    SSvalues_AggVars=zeros(length(SSvaluesFn),1);
    d_val=zeros(1,l_d);
    aprime_val=zeros(1,l_a);
    a_val=zeros(1,l_a);
    z_val=zeros(1,l_z);
    
    for i=1:length(SSvaluesFn)
        % Includes check for cases in which no parameters are actually required
        if isempty(SSvalueParamNames(i).Names) % check for 'SSvalueParamNames={}'
            Values=zeros(N_a,N_z);
            if l_d==0
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
                        z_ind=ind2sub_homemade_gpu([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                z_val(jj2)=z_grid(z_ind(jj2));
                            else
                                z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
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
                        Values(j1,j2)=SSvaluesFn{i}(aprime_val,a_val,z_val);
                        
                    end
                end
            else
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
                        z_ind=ind2sub_homemade_gpu([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                z_val(jj2)=z_grid(z_ind(jj2));
                            else
                                z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
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
                        Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val);
                    end
                end
            end
            Values=reshape(Values,[N_a*N_z,1]);
            
            SSvalues_AggVars(i)=sum(Values.*StationaryDistVec);
        else
            SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames(i).Names);
            Values=zeros(N_a,N_z);
            if l_d==0
                for j1=1:N_a
                    a_ind=ind2sub_homemade_gpu([n_a],j1);
                    for jj1=1:l_a
                        if jj1==1
                            a_val(jj1)=a_grid(a_ind(jj1));
                        else
                            a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
                        end
                    end
                    for j2=1:N_s
                        z_ind=ind2sub_homemade_gpu([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                z_val(jj2)=z_grid(z_ind(jj2));
                            else
                                z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
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
                        Values(j1,j2)=SSvaluesFn{i}(aprime_val,a_val,z_val,SSvalueParamsVec);
                    end
                end
            else
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
                        z_ind=ind2sub_homemade_gpu([n_z],j2);
                        for jj2=1:l_z
                            if jj2==1
                                z_val(jj2)=z_grid(z_ind(jj2));
                            else
                                z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
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
                        Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,SSvalueParamsVec(:));
                    end
                end
            end
            Values=reshape(Values,[N_a*N_z,1]);
            
            % When evaluating value function (which may sometimes give -Inf
            % values) on StationaryDistVec (which at those points will be
            % 0) we get 'NaN'. Use temp as intermediate variable just eliminate those.
            temp=Values.*StationaryDistVec;
            SSvalues_AggVars(i)=sum(temp(~isnan(temp)));
        end
        
    end
    
end


end