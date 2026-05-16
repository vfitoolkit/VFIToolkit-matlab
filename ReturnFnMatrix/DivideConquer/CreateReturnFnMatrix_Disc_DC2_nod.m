function Fmatrix=CreateReturnFnMatrix_Disc_DC2_nod(ReturnFn, n_z, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec, Level)
% 2D divide-and-conquer

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_z=prod(n_z);

l_z=length(n_z); % won't get here if l_z=0
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

if Level==1
    N_a1prime=length(a1prime_grid);
    N_a2prime=length(a2prime_grid);
    N_a1a2prime=N_a1prime*N_a2prime;
    a1prime_grid=repmat(a1prime_grid,N_a2prime,1);
    a2prime_grid=repelem(a2prime_grid,N_a1prime,1);
elseif Level==2
    N_a1a2prime=size(a1prime_grid,1);
end


if l_z==1
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, a2prime_grid, shiftdim(a1_grid,-1), shiftdim(a2_grid,-2), shiftdim(z_gridvals(:,1),-3), ReturnFnParamsCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, a2prime_grid, shiftdim(a1_grid,-1), shiftdim(a2_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), ReturnFnParamsCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, a2prime_grid, shiftdim(a1_grid,-1), shiftdim(a2_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), ReturnFnParamsCell{:});
elseif l_z==4
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, a2prime_grid, shiftdim(a1_grid,-1), shiftdim(a2_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), ReturnFnParamsCell{:});
end

Fmatrix=reshape(Fmatrix,[N_a1a2prime,N_a1,N_a2,N_z]);

end


