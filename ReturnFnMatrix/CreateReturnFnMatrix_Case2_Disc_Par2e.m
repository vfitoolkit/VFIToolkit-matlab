function Fmatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn,n_d, n_a, n_z, n_e, d_gridvals, a_gridvals, z_gridvals, e_gridvals, ReturnFnParamsVec)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a);
l_z=length(n_z);
l_e=length(n_e);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>5
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
if l_e>5
    error('Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)')
end

if l_a>=1
    a1vals=shiftdim(a_gridvals(:,1),-1);
    if l_a>=2
        a2vals=shiftdim(a_gridvals(:,2),-1);
        if l_a>=3
            a3vals=shiftdim(a_gridvals(:,3),-1);
            if l_a>=4
                a4vals=shiftdim(a_gridvals(:,4),-1);
            end
        end
    end
end
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-2);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-2);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-2);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-2);
                if l_z>=5
                    z5vals=shiftdim(z_gridvals(:,5),-2);
                end
            end
        end
    end
end
if l_e>=1
    e1vals=shiftdim(e_gridvals(:,1),-3);
    if l_e>=2
        e2vals=shiftdim(e_gridvals(:,2),-3);
        if l_e>=3
            e3vals=shiftdim(e_gridvals(:,3),-3);
            if l_e>=4
                e4vals=shiftdim(e_gridvals(:,4),-3);
                if l_e>=5
                    e5vals=shiftdim(e_gridvals(:,5),-3);
                end
            end
        end
    end
end



if l_e==1
    if l_d==1
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ReturnFnParamsCell{:});
        end
    end
elseif l_e==2
    if l_d==1
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ReturnFnParamsCell{:});
        end
    end
elseif l_e==3
    if l_d==1
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ReturnFnParamsCell{:});
        end
    end
elseif l_e==4
    if l_d==1
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ReturnFnParamsCell{:});
        end
    end
elseif l_e==5
    if l_d==1
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ReturnFnParamsCell{:});
        end
    end
end

Fmatrix=reshape(Fmatrix,[N_d,N_a,N_z,N_e]); % this is probably obsolete given they are all based on gridvals


end


