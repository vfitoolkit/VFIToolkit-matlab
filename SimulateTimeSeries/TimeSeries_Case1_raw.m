function TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, PolicyIndexesKron, TimeSeriesFn, n_d, n_a, n_s, d_grid, a_grid, s_grid)

if n_d(1)==0
    num_d_vars=0;
else
    num_d_vars=length(n_d);
end
num_a_vars=length(n_a);
num_s_vars=length(n_s);

periods=length(TimeSeriesIndexesKron(1,:));

TimeSeriesKron=zeros(length(TimeSeriesFn),periods);
d_values=zeros(num_d_vars,1);
aprime_values=zeros(num_a_vars,1);
a_values=zeros(num_a_vars,1);
s_values=zeros(num_s_vars,1);
d_indexes=zeros(num_d_vars,1); aprime_indexes=zeros(num_a_vars,1);

for t=1:periods
    a_c=TimeSeriesIndexesKron(1,t);
    a_indexes=ind2sub_homemade([n_a],a_c);
    for jj1=1:num_a_vars
        if jj1==1
            a_values(jj1)=a_grid(a_indexes(jj1));
        else
            a_values(jj1)=a_grid(a_indexes(jj1)+sum(n_a(1:jj1-1)));
        end
    end
    
    s_c=TimeSeriesIndexesKron(2,t);
    s_indexes=ind2sub_homemade([n_s],s_c);
    for jj2=1:num_s_vars
        if jj2==1
            s_values(jj2)=s_grid(s_indexes(jj2));
        else
            s_values(jj2)=s_grid(s_indexes(jj2)+sum(n_s(1:jj2-1)));
        end
    end
    
    if num_d_vars==0
        [aprime_indexes]=PolicyIndexesKron(:,a_c,s_c);
    else
        temp=PolicyIndexesKron(:,a_c,s_c);
        d_indexes=temp(1); aprime_indexes=temp(2);
        for kk1=1:num_d_vars
            if kk1==1
                d_values(kk1)=d_grid(d_indexes(kk1));
            else
                d_values(kk1)=d_grid(d_indexes(kk1)+sum(n_d(1:kk1-1)));
            end
        end
    end
    for kk2=1:num_a_vars
        if kk2==1
            aprime_values(kk2)=a_grid(aprime_indexes(kk2));
        else
            aprime_values(kk2)=a_grid(aprime_indexes(kk2)+sum(n_a(1:kk2-1)));
        end
    end
    for i=1:length(TimeSeriesFn)
        TimeSeriesKron(i,t)=TimeSeriesFn{i}(d_values,aprime_values,a_values,s_values);
%        TimeSeriesKron(i,t)=TimeSeriesFn{i}(d_values,d_indexes,aprime_values,aprime_indexes,a_values,a_indexes,s_values,s_indexes);
    end
end

end