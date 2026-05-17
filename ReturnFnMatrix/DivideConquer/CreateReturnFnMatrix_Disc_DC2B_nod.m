function Fmatrix=CreateReturnFnMatrix_Disc_DC2B_nod(ReturnFn, n_z, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec, Level)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_z=prod(n_z);

l_z=length(n_z); % won't get here if l_z=0
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

if Level==1
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
elseif Level==2
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
end
N_a2prime=N_a2;

if l_z==1
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), ReturnFnParamsCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), ReturnFnParamsCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), ReturnFnParamsCell{:});
elseif l_z==4
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), shiftdim(z_gridvals(:,1),-4), shiftdim(z_gridvals(:,2),-4), shiftdim(z_gridvals(:,3),-4), shiftdim(z_gridvals(:,4),-4), ReturnFnParamsCell{:});
end

if Level==1
    Fmatrix=reshape(Fmatrix,[N_a1prime,N_a2prime,N_a1,N_a2,N_z]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_a1prime*N_a2prime,N_a1*N_a2,N_z]);
end


end
