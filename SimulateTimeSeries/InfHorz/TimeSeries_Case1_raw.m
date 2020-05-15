function TimeSeriesKron=TimeSeries_Case1_raw(TimeSeriesIndexesKron, PolicyIndexesKron, TimeSeriesFn, n_d, n_a, n_z, d_grid, a_grid, z_grid)

if prod(n_d)==0
    l_d=0;
else
    l_d=length(n_d);
end
l_a=length(n_a);
l_z=length(n_z);

periods=length(TimeSeriesIndexesKron(1,:));

TimeSeriesKron=zeros(length(TimeSeriesFn),periods);
d_vals=zeros(l_d,1);
aprime_vals=zeros(l_a,1);
a_val=zeros(l_a,1);
z_val=zeros(l_z,1);

% Note this is poorly coded, should treat the l_d==0 as a seperate for
% loop, rather than lots of if statements in the for loop. Would be faster.
for t=1:periods
    a_ind=TimeSeriesIndexesKron(1,t);
    a_sub=ind2sub_homemade([n_a],a_ind);
    for jj1=1:l_a
        if jj1==1
            a_val(jj1)=a_grid(a_sub(jj1));
        else
            a_val(jj1)=a_grid(a_sub(jj1)+sum(n_a(1:jj1-1)));
        end
    end
    
    z_ind=TimeSeriesIndexesKron(2,t);
    z_sub=ind2sub_homemade([n_z],z_ind);
    for jj2=1:l_z
        if jj2==1
            z_val(jj2)=z_grid(z_sub(jj2));
        else
            z_val(jj2)=z_grid(z_sub(jj2)+sum(n_z(1:jj2-1)));
        end
    end
    
    if l_d==0
        [aprime_ind]=PolicyIndexesKron(a_ind,z_ind);
        aprime_sub=ind2sub_homemade(n_a,aprime_ind);
    else
        temp=PolicyIndexesKron(:,a_ind,z_ind);
        d_ind=temp(1); aprime_ind=temp(2);
        d_sub=ind2sub_homemade(n_d,d_ind);
        aprime_sub=ind2sub_homemade(n_a,aprime_ind);
        for kk1=1:l_d
            if kk1==1
                d_vals(kk1)=d_grid(d_sub(kk1));
            else
                d_vals(kk1)=d_grid(d_sub(kk1)+sum(n_d(1:kk1-1)));
            end
        end
    end
    for kk2=1:l_a
        if kk2==1
            aprime_vals(kk2)=a_grid(aprime_sub(kk2));
        else
            aprime_vals(kk2)=a_grid(aprime_sub(kk2)+sum(n_a(1:kk2-1)));
        end
    end
    if l_d==0
        for i=1:length(TimeSeriesFn)
            TimeSeriesKron(i,t)=TimeSeriesFn{i}(aprime_vals,a_val,z_val);
        end
    else
        for i=1:length(TimeSeriesFn)
            TimeSeriesKron(i,t)=TimeSeriesFn{i}(d_vals,aprime_vals,a_val,z_val);
        end
    end
end

end