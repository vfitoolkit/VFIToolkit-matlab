function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod(ReturnFn, n_z, N_j, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals_J, ReturnFnParamsAgeMatrix,Level)
% fastOLG DC2A (no d): parallelize over age (j) for the divide-and-conquer
% step in the first endogenous state, iterating over the second.
% Output dims (Level=1): (a1prime, a2prime, a1, a2, j, z)
% Output dims (Level=2): (a1prime*a2prime, a1*a2, j, z)
% Caller pre-shifts z_gridvals_J to [1,1,1,1,N_j,N_z,l_z].

l_a1=1; l_a2=1; % (or else won't get here)
l_z=length(n_z); % won't get here if l_z=0
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_a1-l_a2-l_a1-l_a2)};
end

N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_z=prod(n_z);
N_a2prime=N_a2;

if Level==1
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
elseif Level==2
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
end

if l_z==1
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), z_gridvals_J(1,1,1,1,:,:,1), ReturnFnParamsCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), z_gridvals_J(1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,:,:,2), ReturnFnParamsCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), z_gridvals_J(1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,:,:,3), ReturnFnParamsCell{:});
elseif l_z==4
    Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), z_gridvals_J(1,1,1,1,:,:,1), z_gridvals_J(1,1,1,1,:,:,2), z_gridvals_J(1,1,1,1,:,:,3), z_gridvals_J(1,1,1,1,:,:,4), ReturnFnParamsCell{:});
end

if Level==1
    Fmatrix=reshape(Fmatrix,[N_a1prime,N_a2prime,N_a1,N_a2,N_j,N_z]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_a1prime*N_a2prime,N_a1*N_a2,N_j,N_z]);
end


end
