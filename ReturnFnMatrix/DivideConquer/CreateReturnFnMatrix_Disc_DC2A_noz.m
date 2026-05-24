function Fmatrix=CreateReturnFnMatrix_Disc_DC2A_noz(ReturnFn, n_d, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, ReturnFnParamsVec, Level, Refine)
% Refine=1 at Level=1 or Level=3 collapses N_d into the N_a1prime row (useful when d is singular, e.g. inside a d2_c loop with special_n_d2=ones, so downstream can treat output like the _nod variant).
% Refine=1 at Level=2 keeps N_a1 and N_a2 as separate dims (useful for broadcasting an EV that has the level1iidiff axis as singleton).
% Refine=0 keeps the default shapes.

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a1=length(a1_grid);
N_a2=length(a2_grid);

l_d=length(n_d); % won't get here if l_d=0
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end

if Level==1
    N_a1prime=size(a1prime_grid,1); % Because l_a=1
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2); % Because l_a=1
elseif Level==5 % Level 2 inputs, but for doing semiz without d1, so d2 is singular inside the loop over d2
    N_a1prime=size(a1prime_grid,1);
    a1prime_grid=shiftdim(a1prime_grid,-1); % extra -1 for the singular d2
end
N_a2prime=N_a2;

if l_d==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
elseif l_d==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
elseif l_d==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
elseif l_d==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), ReturnFnParamsCell{:});
end

if Level==1 || Level==3 % For GI
    if Refine==0
        Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2]);
    elseif Refine==1
        Fmatrix=reshape(Fmatrix,[N_d*N_a1prime,N_a2prime,N_a1,N_a2]); % collapse N_d into N_a1prime row
    end
elseif Level==2 || Level==5 % Level 5 = singular d2 inside d2_c loop
    if Refine==0
        Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2]);
    elseif Refine==1
        Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1,N_a2]); % keep N_a1, N_a2 separate for broadcasting
    end
end



end
