function TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, PolicyIndexesKron, TimeSeriesFn, n_d, n_a, n_z, d_grid, a_grid, z_grid)

if n_d(1)==0
    num_d_vars=0;
else
    num_d_vars=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

periods=length(TimeSeriesIndexesKron(1,:));

TimeSeriesKron=zeros(length(TimeSeriesFn),periods);
d_values=zeros(num_d_vars,1);
aprime_values=zeros(l_a,1);
a_val=zeros(l_a,1);
z_val=zeros(l_z,1);
d_ind=zeros(num_d_vars,1); aprime_ind=zeros(l_a,1);

for t=1:periods
    a_c=TimeSeriesIndexesKron(1,t);
    a_indexes=ind2sub_homemade([n_a],a_c);
    for jj1=1:l_a
        if jj1==1
            a_val(jj1)=a_grid(a_indexes(jj1));
        else
            a_val(jj1)=a_grid(a_indexes(jj1)+sum(n_a(1:jj1-1)));
        end
    end
    
    z_c=TimeSeriesIndexesKron(2,t);
    z_ind=ind2sub_homemade([n_z],z_c);
    for jj2=1:l_z
        if jj2==1
            z_val(jj2)=z_grid(z_ind(jj2));
        else
            z_val(jj2)=z_grid(z_ind(jj2)+sum(n_z(1:jj2-1)));
        end
    end
    
    if num_d_vars==0
        [aprime_ind]=PolicyIndexesKron(:,a_c,z_c);
    else
        temp=PolicyIndexesKron(:,a_c,z_c);
        d_ind=temp(1); aprime_ind=temp(2);
        for kk1=1:num_d_vars
            if kk1==1
                d_values(kk1)=d_grid(d_ind(kk1));
            else
                d_values(kk1)=d_grid(d_ind(kk1)+sum(n_d(1:kk1-1)));
            end
        end
    end
    for kk2=1:l_a
        if kk2==1
            aprime_values(kk2)=a_grid(aprime_ind(kk2));
        else
            aprime_values(kk2)=a_grid(aprime_ind(kk2)+sum(n_a(1:kk2-1)));
        end
    end
    for i=1:length(TimeSeriesFn)
        TimeSeriesKron(i,t)=TimeSeriesFn{i}(d_values,aprime_values,a_val,z_val);
    end
end

end