function Values=ValuesOnSSGrid_Case2(ValuesFn,ValuesFnParams,PolicyValuesPermute,n_d,n_a,n_z,a_grid,z_grid,p_val,Parallel)
% IS THERE a way to make this work if the function also depends on pi_z???


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

p_valCell=cell(length(p_val),1);
for ii=1:length(p_val)
    p_valCell(ii,1)={p_val(ii)};
end


if l_d==1 && l_a==1 && l_z==1
    dvals=PolicyValuesPermute(:,:,1);
    avals=a_grid;
    zvals=shiftdim(z_grid,-1);
    Values=arrayfun(ValuesFn, dvals, avals, zvals,p_valCell{:}, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    dvals=PolicyValuesPermute(:,:,:,1);
    avals=a_grid;
    z1vals=shiftdim(z_grid(1:n_z(1)),-1);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-2);
    Values=arrayfun(ValuesFn, dvals, avals, z1vals,z2vals,p_valCell{:}, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    dvals=PolicyValuesPermute(:,:,:,1);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    zvals=shiftdim(z_grid,-2);
    Values=arrayfun(ValuesFn, dvals,a1vals,a2vals, zvals,p_valCell{:}, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    dvals=PolicyValuesPermute(:,:,:,:,1);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    z1vals=shiftdim(z_grid(1:n_z(1)),-2);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
    Values=arrayfun(ValuesFn, dvals, a1vals,a2vals, z1vals,z2vals,p_valCell{:}, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    d1vals=PolicyValuesPermute(:,:,1);
    d2vals=PolicyValuesPermute(:,:,2);
    avals=a_grid;
    zvals=shiftdim(z_grid,-1);
    Values=arrayfun(ValuesFn, d1vals, d2vals, avals, zvals,p_valCell{:}, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    d1vals=PolicyValuesPermute(:,:,:,1);
    d2vals=PolicyValuesPermute(:,:,:,2);
    avals=a_grid;
    z1vals=shiftdim(z_grid(1:n_z(1)),-1);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-2);
    Values=arrayfun(ValuesFn, d1vals, d2vals, avals, z1vals,z2vals,p_valCell{:}, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    d1vals=PolicyValuesPermute(:,:,:,1);
    d2vals=PolicyValuesPermute(:,:,:,2);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    zvals=shiftdim(z_grid,-2);
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals, zvals,p_valCell{:}, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    d1vals=PolicyValuesPermute(:,:,:,:,1);
    d2vals=PolicyValuesPermute(:,:,:,:,2);
    a1vals=a_grid(1:n_a(1));
    a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
    z1vals=shiftdim(z_grid(1:n_z(1)),-2);
    z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
    Values=arrayfun(ValuesFn, d1vals, d2vals, a1vals,a2vals, z1vals, z2vals, p_valCell{:}, ParamCell{:});
end

Values=reshape(Values,[N_a,N_z]);


end