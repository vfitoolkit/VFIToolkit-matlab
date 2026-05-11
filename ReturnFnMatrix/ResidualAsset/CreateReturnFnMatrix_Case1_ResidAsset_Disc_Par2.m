function Fmatrix=CreateReturnFnMatrix_Case1_ResidAsset_Disc_Par2(ReturnFn, n_d, n_a, n_r, n_z, d_grid, a_grid, r_grid, z_gridvals, ReturnFnParamsVec,Refine) % Refine is an optional input

if size(d_grid,2)==1 % stacked-column % IN FUTURE, CHANGE INPUT TO BE d_gridvals
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else
    d_gridvals=d_grid;
end

if ~exist('Refine','var')
    Refine=0;
end

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a=prod(n_a);
N_r=prod(n_r);
N_z=prod(n_z);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a); 
l_r=length(n_r); 
l_z=length(n_z);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>3
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_r>2
    error('Using GPU for the return fn does not allow for more than two of r variable (you have length(n_r)>2)')
end
if l_z>5
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>5)')
end


if l_a>=1
    aprime1vals=shiftdim(a_grid(1:n_a(1)),-1);
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_a-1);
    if l_a>=2
        aprime2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-1-1);
        if l_a>=3
            aprime3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-1-2);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-1-2);
            if l_a>=4
                aprime4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-1-3);
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-1-3);
            end
        end
    end
end


if l_r>=1
    r1vals=shiftdim(r_grid(1:n_r(1)),-l_a-l_a-1);
    if l_r>=2
        r2vals=shiftdim(r_grid(n_r(1)+1:sum(n_r(1:2))),-l_a-l_a-1-1);
    end
end
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-1-l_a-l_a-l_r);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-1-l_a-l_a-l_r);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-1-l_a-l_a-l_r);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-1-l_a-l_a-l_r);
                if l_z>=5
                    z5vals=shiftdim(z_gridvals(:,5),-1-l_a-l_a-l_r);
                end
            end
        end
    end
end

% Note: if l_r=0 you should never end up here anyway
if l_r==1
    if l_d==0 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==1
        d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    end
elseif l_r==2
    if l_d==0 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==0 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==1
        d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==1 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==2 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==3 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    elseif l_d==4 && l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, r1vals, r2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ReturnFnParamsCell{:});
    end
end

if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a,N_r,N_z]);
else
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a,N_r,N_z]); % This is the difference when using Refine
    else
        Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a,N_r,N_z]);
    end
end

end


