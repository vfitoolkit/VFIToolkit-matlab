function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A(ReturnFn, n_d, n_z, N_j, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,Level)
% fastOLG DC2A: parallelize over age (j) for the divide-and-conquer step in
% the first endogenous state, iterating over the second endogenous state.
% Output dims (Level=1): (d, a1prime, a2prime, a1, a2, j, z)
% Output dims (Level=2): (d*a1prime*a2prime, a1*a2, j, z)
% Caller pre-shifts z_gridvals_J to [1,1,1,1,1,N_j,N_z,l_z].

l_d=length(n_d); % won't get here if l_d=0
l_a1=1; l_a2=1; % (or else won't get here)
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_z=length(n_z); % won't get here if l_z=0
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-1-l_a1-l_a2-l_a1-l_a2)};
end

N_d=prod(n_d);
N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_z=prod(n_z);
N_a2prime=N_a2;

if Level==1
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2); % Because l_a1=1
end

if l_z==1
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), ReturnFnParamsCell{:});
    end
elseif l_z==2
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), ReturnFnParamsCell{:});
    end
elseif l_z==3
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), ReturnFnParamsCell{:});
    end
elseif l_z==4
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), z_gridvals_J(1,1,1,1,1,:,:,4), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), z_gridvals_J(1,1,1,1,1,:,:,4), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), z_gridvals_J(1,1,1,1,1,:,:,4), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), z_gridvals_J(1,1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,1,:,:,3), z_gridvals_J(1,1,1,1,1,:,:,4), ReturnFnParamsCell{:});
    end
end

if Level==1 || Level==3
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_j,N_z]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2,N_j,N_z]);
end


end
