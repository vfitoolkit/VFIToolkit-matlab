function SSvalues_AggVars=SSvalues_AggVars_Case1(StationaryDist, PolicyIndexes, SSvaluesFn, Parameters,SSvalueParamNames, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val, Parallel)
% Evaluates the aggregate value (weighted sum/integral) for each element of SSvaluesFn

SSvalueParamsVec=CreateVectorFromParams(Parameters,SSvalueParamNames);

if n_d(1)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

% Check if the SSvaluesFn depends on pi_s, if not then can do it all much
% faster. (I have been unable to figure out how to really take advantage of GPU
% when there is pi_s).
nargin_vec=zeros(numel(SSvaluesFn),1);
for ii=1:numel(SSvaluesFn)
    nargin_vec(ii)=nargin(SSvaluesFn{ii});
end
if max(nargin_vec)==(l_d+2*l_a+l_z+1+length(SSvalueParamsVec)) && Parallel==2
    SSvalues_AggVars=SSvalues_AggVars_Case1_NoPi(StationaryDist, PolicyIndexes, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, p_val, Parallel);
    return 
end

N_a=prod(n_a);
N_s=prod(n_z);

if Parallel==2 % This Parallel==2 case is only used if there is pi_s
    SSvalues_AggVars=zeros(length(SSvaluesFn),1,'gpuArray');
    d_val=zeros(l_d,1,'gpuArray');
    aprime_val=zeros(l_a,1,'gpuArray');
    a_val=zeros(l_a,1,'gpuArray');
    z_val=zeros(l_z,1,'gpuArray');
    
    SteadyStateDistVec=reshape(StationaryDist,[N_a*N_s,1]);

    for i=1:length(SSvaluesFn)
    
% %      % HOW CAN I DEAL WITH pi_s when using arrayfun????
                
        Values=zeros(N_a,N_s,'gpuArray');
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
                s_ind=ind2sub_homemade_gpu([n_z],j2);
                for jj2=1:l_z
                    if jj2==1
                        z_val(jj2)=z_grid(s_ind(jj2));
                    else
                        z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                    end
                end
                if l_d==0
                    [aprime_ind]=PolicyIndexes(:,j1,j2);
                else
                    d_ind=PolicyIndexes(1:l_d,j1,j2);
                    aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
                    for kk1=1:l_d
                        if kk1==1
                            d_val(kk1)=d_grid(d_ind(kk1));
                        else
                            d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                        end
                    end
                end
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_ind(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                Values(j1,j2)=SSvaluesFn{i}(d_val(:),aprime_val(:),a_val(:),z_val(:),pi_z,p_val);
            end
        end
        Values=reshape(Values,[N_a*N_s,1]);
        
        SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
    end
    

else
    SSvalues_AggVars=zeros(length(SSvaluesFn),1);
    d_val=zeros(l_d,1);
    aprime_val=zeros(l_a,1);
    a_val=zeros(l_a,1);
    z_val=zeros(l_z,1);
    SteadyStateDistVec=reshape(StationaryDist,[N_a*N_s,1]);
    
    for i=1:length(SSvaluesFn)
        Values=zeros(N_a,N_s);
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
                s_ind=ind2sub_homemade_gpu([n_z],j2);
                for jj2=1:l_z
                    if jj2==1
                        z_val(jj2)=z_grid(s_ind(jj2));
                    else
                        z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
                    end
                end
                if l_d==0
                    [aprime_ind]=PolicyIndexes(:,j1,j2);
                else
                    d_ind=PolicyIndexes(1:l_d,j1,j2);
                    aprime_ind=PolicyIndexes(l_d+1:l_d+l_a,j1,j2);
                    for kk1=1:l_d
                        if kk1==1
                            d_val(kk1)=d_grid(d_ind(kk1));
                        else
                            d_val(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
                        end
                    end
                end
                for kk2=1:l_a
                    if kk2==1
                        aprime_val(kk2)=a_grid(aprime_ind(kk2));
                    else
                        aprime_val(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
                    end
                end
                Values(j1,j2)=SSvaluesFn{i}(d_val,aprime_val,a_val,z_val,pi_z,p_val);
            end
        end
        Values=reshape(Values,[N_a*N_s,1]);
        
        SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
    end

end


end