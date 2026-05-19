function Fmatrix=CreateReturnFnMatrix_Case2_Disc(ReturnFn,n_d, n_a, n_z, d_gridvals, a_gridvals, z_gridvals, ReturnFnParamsVec)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_d=length(n_d);
l_a=length(n_a);
l_z=length(n_z);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>8
    error('Using GPU for the return fn does not allow for more than eight of semiz and z variables')
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
                    if l_z>=6
                        z6vals=shiftdim(z_gridvals(:,6),-2);
                        if l_z>=7
                            z7vals=shiftdim(z_gridvals(:,7),-2);
                            if l_z>=8
                                z8vals=shiftdim(z_gridvals(:,8),-2);
                            end
                        end
                    end
                end
            end
        end
    end
end

if l_d==1
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    end
elseif l_d==2
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    end
elseif l_d==3
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    end
elseif l_d==4
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ReturnFnParamsCell{:});
        elseif l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ReturnFnParamsCell{:});
        elseif l_z==6
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ReturnFnParamsCell{:});
        elseif l_z==7
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ReturnFnParamsCell{:});
        elseif l_z==8
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ReturnFnParamsCell{:});
        end
    end
end


Fmatrix=reshape(Fmatrix,[N_d,N_a,N_z]);


end
