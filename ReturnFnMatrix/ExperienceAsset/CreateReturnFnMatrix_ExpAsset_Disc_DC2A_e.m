function Fmatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_e(ReturnFn, n_d1, n_d2, n_a2, n_z, n_e, d_gridvals, a1prime_grid, a2prime_gridvals, a1_grid, a2_gridvals, a3_grid, z_gridvals, e_gridvals, ReturnFnParamsVec, Level)
% _e variant of CreateReturnFnMatrix_ExpAsset_Disc_DC2A: with both Markov z and i.i.d. e.
% Output dim ordering: 1=d, 2=a1prime, 3=a2prime, 4=a1, 5=a2, 6=a3, 7=z, 8=e.
%
% Level==1: column a1prime → output [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z, N_e].
% Level==2: multi-D a1prime → output [N_d*N_a1prime*N_a2prime, N_a1*N_a2*N_a3, N_z, N_e].
% Level==3: multi-D a1prime → output [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3, N_z, N_e].
% Supports l_d<=4, l_a2<=3, l_z<=4, l_e<=4.

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

if n_d1(1)==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end
N_d=prod(n_d);
N_a2=prod(n_a2);
N_a3=size(a3_grid,1);
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d);
l_a2=length(n_a2);
l_z=length(n_z);
l_e=length(n_e);

if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variables')
end
if l_a2>3
    error('Using GPU for the return fn does not allow for more than three of folded a2 variables (DC2A_e ExpAsset)')
end
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variables (you have length(n_z)>4)')
end
if l_e>4
    error('Using GPU for the return fn does not allow for more than four of e variables (you have length(n_e)>4)')
end

if Level==1
    N_a1prime=size(a1prime_grid,1);
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2);
end
N_a2prime=N_a2;
N_a1=size(a1_grid,1);

% a2prime at dim 3
if l_a2>=1
    a2prime1vals=shiftdim(a2prime_gridvals(:,1),-2);
    if l_a2>=2
        a2prime2vals=shiftdim(a2prime_gridvals(:,2),-2);
        if l_a2>=3
            a2prime3vals=shiftdim(a2prime_gridvals(:,3),-2);
        end
    end
end

% a1 at dim 4
a1vals=shiftdim(a1_grid,-3);

% a2 at dim 5
if l_a2>=1
    a21vals=shiftdim(a2_gridvals(:,1),-4);
    if l_a2>=2
        a22vals=shiftdim(a2_gridvals(:,2),-4);
        if l_a2>=3
            a23vals=shiftdim(a2_gridvals(:,3),-4);
        end
    end
end

% a3 at dim 6
a3vals=shiftdim(a3_grid,-5);

% z at dim 7
z1vals=shiftdim(z_gridvals(:,1),-6);
if l_z>=2
    z2vals=shiftdim(z_gridvals(:,2),-6);
    if l_z>=3
        z3vals=shiftdim(z_gridvals(:,3),-6);
        if l_z>=4
            z4vals=shiftdim(z_gridvals(:,4),-6);
        end
    end
end

% e at dim 8
e1vals=shiftdim(e_gridvals(:,1),-7);
if l_e>=2
    e2vals=shiftdim(e_gridvals(:,2),-7);
    if l_e>=3
        e3vals=shiftdim(e_gridvals(:,3),-7);
        if l_e>=4
            e4vals=shiftdim(e_gridvals(:,4),-7);
        end
    end
end

if l_e==1
    if l_z==1
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
            end
        end
    end
elseif l_e==2
    if l_z==1
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
            end
        end
    end
elseif l_e==3
    if l_z==1
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
            end
        end
    end
elseif l_e==4
    if l_z==1
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==2
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==3
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        end
    elseif l_z==4
        if l_d==1
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==2
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==3
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        elseif l_d==4
            if l_a2==1
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==2
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            elseif l_a2==3
                Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
            end
        end
    end
end

if Level==1 || Level==3
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_a3,N_z,N_e]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2*N_a3,N_z,N_e]);
end

end
