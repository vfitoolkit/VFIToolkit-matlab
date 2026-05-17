function Fmatrix=CreateReturnFnMatrix_Disc_DC2B_nod_noz(ReturnFn, a1prime_grid, a2prime_grid, a1_grid, a2_grid, ReturnFnParamsVec, Level)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_a1=length(a1_grid);
N_a2=length(a2_grid);

N_a1prime=size(a1prime_grid,1); % Because l_a=1
N_a2prime=N_a2;

Fmatrix=arrayfun(ReturnFn, a1prime_grid, shiftdim(a2prime_grid,-1), shiftdim(a1_grid,-2), shiftdim(a2_grid,-3), ReturnFnParamsCell{:});

if Level==1
    Fmatrix=reshape(Fmatrix,[N_a1prime,N_a2prime,N_a1,N_a2]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_a1prime*N_a2prime,N_a1*N_a2]);
end


end
