function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC2_Par2(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_z=prod(n_z);

l_d=length(n_d); % won't get here if l_d=0
l_z=length(n_z); % won't get here if l_z=0
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
if Level==1
    N_a1prime=length(a1prime_grid);
    N_a2prime=length(a2prime_grid);
    N_a1a2prime=N_a1prime*N_a2prime;
    a1prime_grid=repmat(a1prime_grid',1,N_a2prime);
    a2prime_grid=repelem(a2prime_grid',1,N_a1prime);
elseif Level==2
    N_a1a2prime=max(size(a1prime_grid,2),size(a2prime_grid,2)); % max(), they will either both be same size, or one is singular
end


if l_z==1
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), ReturnFnParamsCell{:});
    end
elseif l_z==2
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), ReturnFnParamsCell{:});
    end
elseif l_z==3
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), ReturnFnParamsCell{:});
    end
elseif l_z==4
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), shiftdim(z_gridvals(:,4),-4), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), shiftdim(z_gridvals(:,4),-4), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), shiftdim(z_gridvals(:,4),-4), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime_grid, shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), shiftdim(z_gridvals(:,4),-4), ReturnFnParamsCell{:});
    end
end


if Level==1
    Fmatrix=reshape(Fmatrix,[N_d,N_a1a2prime,N_a1,N_a2,N_z]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1a2prime,N_a1,N_a2,N_z]);
end



end


