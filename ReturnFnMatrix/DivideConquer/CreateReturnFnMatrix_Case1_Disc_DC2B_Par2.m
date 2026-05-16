function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC2B_Par2(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid, a2prime_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParamsVec, Level, Refine)
% Refine=1 keeps N_a1 and N_a2 as separate dims at Level=2 (useful for broadcasting an EV that has the level1iidiff axis as singleton).
% Refine=0 stacks them into N_a1*N_a2 as before. Refine has no effect at Level=1 or Level=3 (those keep N_a1, N_a2 separate already).

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
    N_a1prime=size(a1prime_grid,1);
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2
    N_a1prime=size(a1prime_grid,2);
elseif Level==3
    N_a1prime=size(a1prime_grid,2);
    % a1prime_grid unchanged
    % Level 3 has level 2 inputs but level 1 outputs, used for GI
end
N_a2prime=N_a2;


if l_z==1
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), ReturnFnParamsCell{:});
    end
elseif l_z==2
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), ReturnFnParamsCell{:});
    end
elseif l_z==3
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), ReturnFnParamsCell{:});
    end
elseif l_z==4
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), shiftdim(z_gridvals(:,4),-5), ReturnFnParamsCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), shiftdim(z_gridvals(:,4),-5), ReturnFnParamsCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), shiftdim(z_gridvals(:,4),-5), ReturnFnParamsCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a2prime_grid,-2), shiftdim(a1_grid,-3), shiftdim(a2_grid,-4), shiftdim(z_gridvals(:,1),-5), shiftdim(z_gridvals(:,2),-5), shiftdim(z_gridvals(:,3),-5), shiftdim(z_gridvals(:,4),-5), ReturnFnParamsCell{:});
    end
end

if Level==1
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_z]);
elseif Level==2 % For level 2
    if Refine==0
        Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2,N_z]);
    elseif Refine==1
        Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1,N_a2,N_z]); % keep N_a1, N_a2 separate for broadcasting
    end
elseif Level==3 % For GI
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_z]);
end



end


