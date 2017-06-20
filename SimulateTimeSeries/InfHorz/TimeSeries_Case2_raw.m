function TimeSeriesKron=TimeSeries_Case2_raw(SteadyStateDistKron, PolicyIndexesKron, TimeSeriesFn, n_d, n_a, n_s, d_grid, a_grid, s_grid, pi_s)
%INCOMPLETE
num_d_vars=length(n_d);
num_a_vars=length(n_a);
num_s_vars=length(n_s);
N_a=prod(n_a);
N_s=prod(n_s);

SSvalues_AggVars=zeros(length(SSvaluesFn),1);
d_values=zeros(num_d_vars,1);
a_values=zeros(num_a_vars,1);
s_values=zeros(num_s_vars,1);
SteadyStateDistVec=reshape(SteadyStateDistKron,[N_a*N_s,1]);
for i=1:length(SSvaluesFn)
    Values=zeros(N_a,N_z);
    for j1=1:N_a
        a_indexes=ind2sub_homemade([n_a],j1);
        for jj1=1:num_a_vars
            if jj1==1
                a_values(jj1)=a_grid(a_indexes(jj1));
            else
                a_values(jj1)=a_grid(a_indexes(jj1)+sum(n_a(1:jj1-1)));
            end
        end
        for j2=1:N_s
            s_indexes=ind2sub_homemade([n_s],j2);
            for jj2=1:num_s_vars
                if jj2==1
                    s_values(jj2)=s_grid(s_indexes(jj2));
                else
                    s_values(jj2)=s_grid(s_indexes(jj2)+sum(n_s(1:jj2-1)));
                end
            end
            d_indexes=ind2sub_homemade([n_d],PolicyIndexesKron(j1,j2));
            for kk=1:num_d_vars
                if kk==1
                    d_values(kk)=d_grid(d_indexes(kk));
                else
                    d_values(kk)=d_grid(d_indexes(kk)+sum(n_d(1:kk-1)));
                end
            end
            %SSvalues_AggVars(i)=SSvalues_AggVars(i)+SteadyStateDistKron(j1+(j2-1)*N_a)*SSvaluesFn{i}(d_values,a_values,s_values,pi_s,p_values);
            Values(j1,j2)=SSvaluesFn{i}(d_values,d_indexes,a_values,a_indexes,s_values,s_indexes,pi_s,p_values);
        end
    end
    Values=reshape(Values,[N_a*N_s,1]);
    SSvalues_AggVars(i)=sum(Values.*SteadyStateDistVec);
end

end

