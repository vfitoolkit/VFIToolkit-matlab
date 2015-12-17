function SSvalues_AggVars=SSvalues_AggVars_Case2_vec(StationaryDist, PolicyIndexes, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, pi_z,p_val,Parallel)

if nargin<12 %Default is to assume SteadyStateDist exists on the gpu
    Parallel=2;
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);

% Check if the SSvaluesFn depends on pi_s, if not then can do it all much
% faster. (I have been unable to figure out how to really take advantage of GPU
% when there is pi_s).
nargin_vec=zeros(numel(SSvaluesFn),1);
for ii=1:numel(SSvaluesFn)
    nargin_vec(ii)=nargin(SSvaluesFn{ii});
end
if max(nargin_vec)==(l_d+l_a+l_z+1+length(SSvalueParamsVec)) && Parallel==2
    SSvalues_AggVars=SSvalues_AggVars_Case2_NoPi(StationaryDist, PolicyIndexes, SSvaluesFn, SSvalueParamsVec, n_d, n_a, n_z, d_grid, a_grid, z_grid, p_val, Parallel);
    return 
end

N_a=prod(n_a);
N_z=prod(n_z);

if Parallel==2
    PolicyIndexesKron=gather(reshape(PolicyIndexes,[l_d,N_a,N_z]));
    SteadyStateDistVec=gather(reshape(StationaryDist,[N_a*N_z,1]));
    d_grid=gather(d_grid);
    a_grid=gather(a_grid);
    z_grid=gather(z_grid);
    pi_z=gather(pi_z);
else 
    PolicyIndexesKron=reshape(PolicyIndexes,[l_d,N_a,N_z]);
    SteadyStateDistVec=reshape(StationaryDist,[N_a*N_z,1]);
end

d_val=zeros(l_d,1);
a_val=zeros(l_a,1);
z_val=zeros(l_z,1);
Values=zeros(N_a,N_z,length(SSvaluesFn));
% Should be able to speed this up with a parfor
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
                z_val(jj2)=z_grid(s_ind(jj2));
            else
                z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
            end
        end
        d_ind=PolicyIndexesKron(:,j1,j2);
        for kk=1:l_d
            if kk==1
                d_val(kk)=d_grid(d_ind(kk));
            else
                d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
            end
        end
        for i=1:length(SSvaluesFn)
            Values(j1,j2,i)=SSvaluesFn{i}(d_val,a_val,z_val,pi_z,p_val);
        end
    end
end
Values=reshape(Values,[N_a*N_z,length(SSvaluesFn)]);
SSvalues_AggVars=sum(Values.*(SteadyStateDistVec*ones(1,length(SSvaluesFn))),1);

% SSvalues_AggVars=zeros(length(SSvaluesFn),1);
% parfor i=1:length(SSvaluesFn) 
%     % This parfor loop can probably be written much faster (seems like I should par across N_a, 
%     % and make length(SSvalueFn) the innermost loop to take advantage that d_val, a_val, z_val, pi_z are same for all the SSvalueFn entries)
%     d_val=zeros(l_d,1);
%     a_val=zeros(l_a,1);
%     z_val=zeros(l_z,1);
%     Values=zeros(N_a,N_z);
%     for j1=1:N_a
%         a_ind=ind2sub_homemade([n_a],j1);
%         for jj1=1:l_a
%             if jj1==1
%                 a_val(jj1)=a_grid(a_ind(jj1));
%             else
%                 a_val(jj1)=a_grid(a_ind(jj1)+sum(n_a(1:jj1-1)));
%             end
%         end
%         for j2=1:N_z
%             s_ind=ind2sub_homemade([n_z],j2);
%             for jj2=1:l_z
%                 if jj2==1
%                     z_val(jj2)=z_grid(s_ind(jj2));
%                 else
%                     z_val(jj2)=z_grid(s_ind(jj2)+sum(n_z(1:jj2-1)));
%                 end
%             end
%             d_ind=PolicyIndexesKron(:,j1,j2);
%             for kk=1:l_d
%                 if kk==1
%                     d_val(kk)=d_grid(d_ind(kk));
%                 else
%                     d_val(kk)=d_grid(d_ind(kk)+sum(n_d(1:kk-1)));
%                 end
%             end
%             Values(j1,j2)=SSvaluesFn{i}(d_val,a_val,z_val,pi_z,p_val);
%         end
%     end
%     Values=reshape(Values,[N_a*N_z,1]);
%     SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
% end

