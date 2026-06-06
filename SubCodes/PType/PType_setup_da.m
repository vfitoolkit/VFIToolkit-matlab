function [n_d_temp,n_a_temp,d_grid_temp,a_grid_temp]=PType_setup_da(iistr,n_d,n_a,d_grid,a_grid)

if isstruct(n_d)
    n_d_temp=n_d.(iistr);
else
    n_d_temp=n_d;
end
if isstruct(n_a)
    n_a_temp=n_a.(iistr);
else
    n_a_temp=n_a;
end

if isstruct(d_grid)
    d_grid_temp=d_grid.(iistr);
else
    d_grid_temp=d_grid;
end
if isstruct(a_grid)
    a_grid_temp=a_grid.(iistr);
else
    a_grid_temp=a_grid;
end

end