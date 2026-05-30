function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_DC2A_nod_noz(ReturnFn, N_j, a1prime_grid, a2prime_grid, a1_grid, a2_grid, ReturnFnParamsAgeMatrix,Level)
% fastOLG DC2A (no d, no z): parallelize over age (j) for the divide-and-conquer
% step in the first endogenous state, iterating over the second.
% Output dims (Level=1): (a1prime, a2prime, a1, a2, j)
% Output dims (Level=2): (a1prime*a2prime, a1*a2, j)

l_a1=1; l_a2=1; % (or else won't get here)

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_a1-l_a2-l_a1-l_a2)};
end

N_a1=length(a1_grid);
N_a2=length(a2_grid);
N_a2prime=N_a2;

if Level==1
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
elseif Level==2
    N_a1prime=size(a1prime_grid,1); % Because l_a1=1
end

Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), ReturnFnParamsCell{:});

if Level==1
    Fmatrix=reshape(Fmatrix,[N_a1prime,N_a2prime,N_a1,N_a2,N_j]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_a1prime*N_a2prime,N_a1*N_a2,N_j]);
end


end
