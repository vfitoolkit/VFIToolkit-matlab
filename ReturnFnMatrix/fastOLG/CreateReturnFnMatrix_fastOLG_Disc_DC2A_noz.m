function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A_noz(ReturnFn, n_d, N_j, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, ReturnFnParamsAgeMatrix,Level)
% fastOLG DC2A (no z): parallelize over age (j) for the divide-and-conquer
% step in the first endogenous state, iterating over the second.
% Output dims (Level=1): (d, a1prime, a2prime, a1, a2, j)
% Output dims (Level=2): (d*a1prime*a2prime, a1*a2, j)

l_d=length(n_d); % won't get here if l_d=0
l_a1=1; l_a2=1; % (or else won't get here)
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_d-l_a1-l_a2-l_a1-l_a2)};
end

N_d=prod(n_d);
N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_a2prime=N_a2;

if Level==1
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2); % Because l_a1=1
end

if l_d==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
elseif l_d==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
elseif l_d==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
elseif l_d==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
end

if Level==1 || Level==3
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_j]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2,N_j]);
end


end
