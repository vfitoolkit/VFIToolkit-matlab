function Values=ValuesOnSSGrid_Case2(ValuesFn,ValuesFnParams,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,Parallel)

if Parallel~=2
    disp('ValuesOnSSGrid_Case2() only works for Parallel==2')
end

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
N_a=prod(n_a);
N_z=prod(n_z);

ParamCell=cell(length(ValuesFnParams),1);
for ii=1:length(ValuesFnParams)
    ParamCell(ii,1)={ValuesFnParams(ii)};
end

if l_a>=1
    a1vals=a_grid(1:n_a(1));    
end
if l_a>=2
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
end
if l_a>=3
    a3vals=shiftdim(a_grid(n_a(1)+n_a(2)+1:n_a(1)+n_a(2)+n_a(3)),-2);
end
if l_z>=1
    z1vals=shiftdim(z_grid(1:n_z(1)),-l_a);
end
if l_z>=2
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-l_a-1);
end
if l_z>=3
    z3vals=shiftdim(z_grid(n_z(1)+n_z(2)+1:n_z(1)+n_z(2)+n_z(3)),-l_a-2);    
end

if l_a+l_z==2
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,1);
    end
    if l_d>=2
        d2vals=PolicyValuesPermute(:,:,2);
    end
    if l_d>=3
        d3vals=PolicyValuesPermute(:,:,3);
    end
end
if l_a+l_z==3
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,1);
    end
    if l_d>=2
        d2vals=PolicyValuesPermute(:,:,:,2);
    end
    if l_d>=3
        d3vals=PolicyValuesPermute(:,:,:,3);
    end
end
if l_a+l_z==4
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,1);
    end
    if l_d>=2
        d2vals=PolicyValuesPermute(:,:,:,:,2);
    end
    if l_d>=3
        d3vals=PolicyValuesPermute(:,:,:,:,3);
    end
end
if l_a+l_z==5
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,1);
    end
    if l_d>=2
        d2vals=PolicyValuesPermute(:,:,:,:,:,2);
    end
    if l_d>=3
        d3vals=PolicyValuesPermute(:,:,:,:,:,3);
    end
end
if l_a+l_z==6
    if l_d>=1
        d1vals=PolicyValuesPermute(:,:,:,:,:,:,1);
    end
    if l_d>=2
        d2vals=PolicyValuesPermute(:,:,:,:,:,:,2);
    end
    if l_d>=3
        d3vals=PolicyValuesPermute(:,:,:,:,:,:,3);
    end
end

if l_d==1 && l_a==1 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    Values=arrayfun(ValuesFn, d1vals,a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==1
    Values=arrayfun(ValuesFn, d1vals,a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals, zvals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals, z1vals, z2vals, ParamCell{:}); 
elseif l_d==2 && l_a==2 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals, z1vals, z2vals, z3vals, ParamCell{:}); 
elseif l_d==2 && l_a==3 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals,a3vals, zvals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals,a3vals, z1vals, z2vals, ParamCell{:}); 
elseif l_d==2 && l_a==3 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals,a3vals, z1vals, z2vals, z3vals, ParamCell{:}); 
elseif l_d==3 && l_a==1 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals,a2vals, zvals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals,a2vals, z1vals, z2vals, ParamCell{:}); 
elseif l_d==3 && l_a==2 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals,a2vals, z1vals, z2vals, z3vals, ParamCell{:}); 
elseif l_d==3 && l_a==3 && l_z==1
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals,a2vals,a3vals, zvals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==2
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals,a2vals,a3vals, z1vals, z2vals, ParamCell{:}); 
elseif l_d==3 && l_a==3 && l_z==3
    Values=arrayfun(ValuesFn, d1vals, d2vals, d3vals, a1vals,a2vals,a3vals, z1vals, z2vals, z3vals, ParamCell{:}); 
end

Values=reshape(Values,[N_a,N_z]);


end