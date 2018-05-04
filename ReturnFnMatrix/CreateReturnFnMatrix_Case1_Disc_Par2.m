function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid,ReturnFnParamsVec)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParamsVec),1);
for ii=1:length(ReturnFnParamsVec)
    ParamCell(ii,1)={ReturnFnParamsVec(ii)};
end


N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a); 
l_z=length(n_z);
if l_a>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4): (in CreateReturnFnMatrix_Case1_Disc_Parallel2). Email me (robertdkirkby@gmail.com) and I will add this functionality (have been too lazy till now).')
end
if l_z>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4): (in CreateReturnFnMatrix_Case1_Disc_Parallel2). Email me (robertdkirkby@gmail.com) and I will add this functionality (have been too lazy till now).')
end

if N_d==0
    if l_a==1
        aprimevals=a_grid;
        avals=shiftdim(a_grid,-1);
        if l_z==1
            zvals=shiftdim(z_grid,-2);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(z_grid(1:n_z(1)),-2);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, z1vals,z2vals,ParamCell{:});
        elseif l_z==3
            z1vals=shiftdim(z_grid(1:n_z(1)),-2);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-4);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_z==4
            z1vals=shiftdim(z_grid(1:n_z(1)),-2);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-3);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-4);
            z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-5);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    elseif l_a==2 
        a1primevals=a_grid(1:n_a(1));
        a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-1);
        a1vals=shiftdim(a_grid(1:n_a(1)),-2);
        a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
        if l_z==1
            zvals=shiftdim(z_grid,-4);
            Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(z_grid(1:n_z(1)),-4);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
        elseif l_z==3
            z1vals=shiftdim(z_grid(1:n_z(1)),-4);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_z==4
            z1vals=shiftdim(z_grid(1:n_z(1)),-4);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
            z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-7);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    elseif l_a==3
        a1primevals=a_grid(1:n_a(1));
        a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
        a1vals=shiftdim(a_grid(1:n_a(1)),-3);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-4);
        a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-5);
        if l_z==1
            zvals=shiftdim(z_grid,-6);
            Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(z_grid(1:n_z(1)),-6);
            z2vals=shiftdim(z_grid(n_z(1)+1:sum(n_z(1:2))),-7);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
        elseif l_z==3
            z1vals=shiftdim(z_grid(1:n_z(1)),-6);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_z==4
            z1vals=shiftdim(z_grid(1:n_z(1)),-6);
            z2vals=shiftdim(z_grid(n_z(1)+1:sum(n_z(1:2))),-7);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
            z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-9);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    elseif l_a==4
        a1primevals=a_grid(1:n_a(1));
        a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
        a4primevals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
        a1vals=shiftdim(a_grid(1:n_a(1)),-4);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-5);
        a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-6);
        a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-7);
        if l_z==1
            zvals=shiftdim(z_grid,-8);
            Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(z_grid(1:n_z(1)),-8);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            z1vals=shiftdim(z_grid(1:n_z(1)),-8);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            z1vals=shiftdim(z_grid(1:n_z(1)),-8);
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
            z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
            z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-11);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    end
    
    if ~(l_a==1 && l_z==1)
        Fmatrix=reshape(Fmatrix,[N_a,N_a,N_z]);
    end
    
else
    l_d=length(n_d); 
    if l_d>4
        disp('ERROR: Using GPU the return fn does not allow for more than four of d variable (length(n_d)>4): (in CreateReturnFnMatrix_Case1_Disc_Parallel2)')
    end
    
    if l_d==1 
        dvals=d_grid; dvals(1,1,1)=d_grid(1);
        if l_a==1
            aprimevals=shiftdim(a_grid,-1);
            avals=shiftdim(a_grid,-2);
            if l_z==1
                zvals=shiftdim(z_grid,-3);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-3);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-3);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-5);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-3);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-4);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-5);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-6);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2 
            a1primevals=shiftdim(a_grid(1:n_a(1)),-1);
            a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-2);
            a1vals=shiftdim(a_grid(1:n_a(1)),-3);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
            if l_z==1
                zvals=shiftdim(z_grid,-5);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-5);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-5);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-5);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-8);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_a(1)),-1);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-2);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-3);
            a1vals=shiftdim(a_grid(1:n_a(1)),-4);
            a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-5);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-6);
            if l_z==1
                zvals=shiftdim(z_grid,-7);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-10);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_a(1)),-1);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-2);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-3);
            a4primevals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-4);
            a1vals=shiftdim(a_grid(1:n_a(1)),-5);
            a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-6);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-7);
            a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-8);
            if l_z==1
                zvals=shiftdim(z_grid,-7);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-10);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    elseif l_d==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        if l_a==1
            aprimevals=shiftdim(a_grid,-2);
            avals=shiftdim(a_grid,-3);
            if l_z==1
                zvals=shiftdim(z_grid,-4);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-4);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-4);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-4);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-5);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2
            a1primevals=shiftdim(a_grid(1:n_a(1)),-2);
            a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-3);
            a1vals=shiftdim(a_grid(1:n_a(1)),-4);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-5);
            if l_z==1
                zvals=shiftdim(z_grid,-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-6);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-6);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-6);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_a(1)),-2);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-3);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-4);
            a1vals=shiftdim(a_grid(1:n_a(1)),-5);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-6);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-7);
            if l_z==1
                zvals=shiftdim(z_grid,-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-8);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-8);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-8);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_a(1)),-2);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-3);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-4);
            a4primevals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-5);
            a1vals=shiftdim(a_grid(1:n_a(1)),-6);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-7);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-8);
            a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-9);
            if l_z==1
                zvals=shiftdim(z_grid,-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-10);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-10);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-10);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    elseif l_d==3
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(2)+1:n_d(2)+n_d(3)),-2);
        if l_a==1
            aprimevals=shiftdim(a_grid,-3);
            avals=shiftdim(a_grid,-4);
            if l_z==1
                zvals=shiftdim(z_grid,-5);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-5);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-5);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-5);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2
            a1primevals=shiftdim(a_grid(1:n_a(1)),-3);
            a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-4);
            a1vals=shiftdim(a_grid(1:n_a(1)),-5);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-6);
            if l_z==1
                zvals=shiftdim(z_grid,-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-7);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_a(1)),-3);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-4);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-5);
            a1vals=shiftdim(a_grid(1:n_a(1)),-6);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-7);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-8);
            if l_z==1
                zvals=shiftdim(z_grid,-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-9);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-9);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-10);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-9);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-10);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-11);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_a(1)),-3);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-4);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-5);
            a4primevals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-6);
            a1vals=shiftdim(a_grid(1:n_a(1)),-7);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-8);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-9);
            a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-10);
            if l_z==1
                zvals=shiftdim(z_grid,-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-11);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-11);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-12);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-11);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-12);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-13);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-14);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    elseif l_d==4
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(2)+1:n_d(2)+n_d(3)),-2);
        d4vals=shiftdim(d_grid(n_d(3)+1:n_d(3)+n_d(4)),-3);
        if l_a==1
            aprimevals=shiftdim(a_grid,-4);
            avals=shiftdim(a_grid,-5);
            if l_z==1
                zvals=shiftdim(z_grid,-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-6);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-6);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-6);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2
            a1primevals=shiftdim(a_grid(1:n_a(1)),-4);
            a2primevals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-5);
            a1vals=shiftdim(a_grid(1:n_a(1)),-6);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-7);
            if l_z==1
                zvals=shiftdim(z_grid,-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-8);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-8);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-8);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_a(1)),-4);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-5);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-6);
            a1vals=shiftdim(a_grid(1:n_a(1)),-7);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-8);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-9);
            if l_z==1
                zvals=shiftdim(z_grid,-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-10);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-10);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-10);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_a(1)),-4);
            a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-5);
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-6);
            a4primevals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-7);
            a1vals=shiftdim(a_grid(1:n_a(1)),-8);
            a2vals=shiftdim(a_grid(n_a(1)+1:n_a(1)+n_a(2)),-9);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-10);
            a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-11);
            if l_z==1
                zvals=shiftdim(z_grid,-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(z_grid(1:n_z(1)),-12);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(z_grid(1:n_z(1)),-12);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-13);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-14);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(z_grid(1:n_z(1)),-12);
                z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-13);
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-14);
                z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-15);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    end
    
    Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a,N_z]);
    
end



% if N_d==0
% %     aprime_dim=gpuArray.ones(prod(n_a),1,1); aprime_dim(:,1,1)=1:1:prod(n_a);
% %     a_dim=gpuArray.ones(1,prod(n_a),1);      a_dim(1,:,1)=1:1:prod(n_a);
% %     z_dim=gpuArray.ones(1,1,prod(n_z));      z_dim(1,1,:)=1:1:prod(n_z);
%     
%     z_gridvals=gpuArray.zeros(N_z,length(n_z));
%     for i1=1:N_z
%         sub=zeros(1,length(n_z));
%         sub(1)=rem(i1-1,n_z(1))+1;
%         for ii=2:length(n_z)-1
%             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%         end
%         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%         
%         if length(n_z)>1
%             sub=sub+[0,cumsum(n_z(1:end-1))];
%         end
%         z_gridvals(i1,:)=z_grid(sub);
%     end
%     
%     a_gridvals=gpuArray.zeros(N_a,length(n_a));
%     for i2=1:N_a
%         sub=zeros(1,length(n_a));
%         sub(1)=rem(i2-1,n_a(1)+1);
%         for ii=2:length(n_a)-1
%             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%         end
%         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%         
%         if length(n_a)>1
%             sub=sub+[0,cumsum(n_a(1:end-1))];
%         end
%         a_gridvals(i2,:)=a_grid(sub);
%     end
%     
%     aprime_gridvals=a_gridvals;
%     a_gridvals=shiftdim(a_gridvals,-1);
%     z_gridvals2=shiftdim(z_gridvals,-2);
%     
%     % TempReturnFn.m is created by CreateCleanedReturnFn() as part of ValueFnIter_Case1()
%     Fmatrix=arrayfun(ReturnFn, aprime_gridvals, a_gridvals, z_gridvals2);
%     
% else % This version generates return function (d,aprime,a,z), rather than (daprime,a,z)
%     
%     z_gridvals=gpuArray.zeros(N_z,length(n_z));
%     for i1=1:N_z
%         sub=zeros(1,length(n_z));
%         sub(1)=rem(i1-1,n_z(1))+1;
%         for ii=2:length(n_z)-1
%             sub(ii)=rem(ceil(i1/prod(n_z(1:ii-1)))-1,n_z(ii))+1;
%         end
%         sub(length(n_z))=ceil(i1/prod(n_z(1:length(n_z)-1)));
%         
%         if length(n_z)>1
%             sub=sub+[0,cumsum(n_z(1:end-1))];
%         end
%         z_gridvals(i1,:)=z_grid(sub);
%     end
%     
%     a_gridvals=gpuArray.zeros(N_a,length(n_a));
%     for i2=1:N_a
%         sub=zeros(1,length(n_a));
%         sub(1)=rem(i2-1,n_a(1)+1);
%         for ii=2:length(n_a)-1
%             sub(ii)=rem(ceil(i2/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
%         end
%         sub(length(n_a))=ceil(i2/prod(n_a(1:length(n_a)-1)));
%         
%         if length(n_a)>1
%             sub=sub+[0,cumsum(n_a(1:end-1))];
%         end
%         a_gridvals(i2,:)=a_grid(sub);
%     end
% 
%     d_gridvals=gpuArray.zeros(N_d,length(n_d));
%     for i1=1:N_d
%         sub=zeros(1,length(n_d));
%         sub(1)=rem(i1-1,n_d(1))+1;
%         for ii=2:length(n_d)-1
%             sub(ii)=rem(ceil(i1/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
%         end
%         sub(length(n_d))=ceil(i1/prod(n_d(1:length(n_d)-1)));
%         
%         if length(n_d)>1
%             sub=sub+[0,cumsum(n_d(1:end-1))];
%         end
%         d_gridvals(i1,:)=d_grid(sub);
%     end
%     
%     aprime_gridvals=shiftdim(a_gridvals,-1);
%     a_gridvals=shiftdim(a_gridvals,-2);
%     z_gridvals2=shiftdim(z_gridvals,-3);
%     
%     % TempReturnFn.m is created by CreateCleanedReturnFn() as part of ValueFnIter_Case1()
%     Fmatrix=arrayfun(ReturnFn, d_gridvals, aprime_gridvals, a_gridvals, z_gridvals2);
% end



end


