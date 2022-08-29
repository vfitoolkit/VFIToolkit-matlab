function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2_LowMem2(ReturnFn, n_d, n_aprime, n_a, n_z, d_grid, a_grid, avals, zvals,ReturnFnParamsVec,Refine) % Refine is an optional input
%If there is no d variable, just input n_d=0 and d_grid=0

if ~exist('Refine','var')
    Refine=0;
end

ParamCell=cell(length(ReturnFnParamsVec),1);
for ii=1:length(ReturnFnParamsVec)
    ParamCell(ii,1)={ReturnFnParamsVec(ii)};
end

N_d=prod(n_d);
N_aprime=prod(n_aprime);
N_a=prod(n_a); % THIS WILL EQUAL 1 FOR LowMem2
N_z=prod(n_z); % THIS WILL EQUAL 1 FOR LowMem2

l_a=length(n_a);
l_z=length(n_z);
if l_a>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

if N_d==0
    if l_a==1
        aprimevals=a_grid;
        avals=shiftdim(avals,-1);
        if l_z==1
            zvals=shiftdim(zvals,-2);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(zvals(1:n_z(1)),-2);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-3);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, z1vals,z2vals,ParamCell{:});
        elseif l_z==3
            z1vals=shiftdim(zvals(1:n_z(1)),-2);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-3);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-4);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_z==4
            z1vals=shiftdim(zvals(1:n_z(1)),-2);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-3);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-4);
            z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-5);
            Fmatrix=arrayfun(ReturnFn, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    elseif l_a==2 
        a1primevals=a_grid(1:n_aprime(1));
        a2primevals=shiftdim(a_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2)),-1);
        a1vals=shiftdim(avals(1:n_a(1)),-2);
        a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-3);
        if l_z==1
            zvals=shiftdim(zvals,-4);
            Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(zvals(1:n_z(1)),-4);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-5);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
        elseif l_z==3
            z1vals=shiftdim(zvals(1:n_z(1)),-4);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-5);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_z==4
            z1vals=shiftdim(zvals(1:n_z(1)),-4);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-5);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
            z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-7);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    elseif l_a==3
        a1primevals=a_grid(1:n_aprime(1));
        a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-1);
        a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-2);
        a1vals=shiftdim(avals(1:n_a(1)),-3);
        a2vals=shiftdim(avals(n_a(1)+1:sum(n_a(1:2))),-4);
        a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-5);
        if l_z==1
            zvals=shiftdim(zvals,-6);
            Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(zvals(1:n_z(1)),-6);
            z2vals=shiftdim(zvals(n_z(1)+1:sum(n_z(1:2))),-7);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
        elseif l_z==3
            z1vals=shiftdim(zvals(1:n_z(1)),-6);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_z==4
            z1vals=shiftdim(zvals(1:n_z(1)),-6);
            z2vals=shiftdim(zvals(n_z(1)+1:sum(n_z(1:2))),-7);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
            z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-9);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    elseif l_a==4
        a1primevals=a_grid(1:n_aprime(1));
        a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-1);
        a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-2);
        a4primevals=shiftdim(a_grid(sum(n_aprime(1:3))+1:sum(n_aprime(1:4))),-3);
        a1vals=shiftdim(avals(1:n_a(1)),-4);
        a2vals=shiftdim(avals(n_a(1)+1:sum(n_a(1:2))),-5);
        a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-6);
        a4vals=shiftdim(avals(sum(n_a(1:3))+1:sum(n_a(1:4))),-7);
        if l_z==1
            zvals=shiftdim(zvals,-8);
            Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
        elseif l_z==2
            z1vals=shiftdim(zvals(1:n_z(1)),-8);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            z1vals=shiftdim(zvals(1:n_z(1)),-8);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            z1vals=shiftdim(zvals(1:n_z(1)),-8);
            z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
            z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
            z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-11);
            Fmatrix=arrayfun(ReturnFn, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
        end
    end
    
    if ~(l_a==1 && l_z==1)
        Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_z]);
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
            avals=shiftdim(avals,-2);
            if l_z==1
                zvals=shiftdim(zvals,-3);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-3);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-4);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-3);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-4);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-5);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-3);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-4);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-5);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-6);
                Fmatrix=arrayfun(ReturnFn, dvals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2 
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-1);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2)),-2);
            a1vals=shiftdim(avals(1:n_a(1)),-3);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-4);
            if l_z==1
                zvals=shiftdim(zvals,-5);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-5);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-6);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-5);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-5);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-8);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-1);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-2);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-3);
            a1vals=shiftdim(avals(1:n_a(1)),-4);
            a2vals=shiftdim(avals(n_a(1)+1:sum(n_a(1:2))),-5);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-6);
            if l_z==1
                zvals=shiftdim(zvals,-7);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-10);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-1);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-2);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-3);
            a4primevals=shiftdim(a_grid(sum(n_aprime(1:3))+1:sum(n_aprime(1:4))),-4);
            a1vals=shiftdim(avals(1:n_a(1)),-5);
            a2vals=shiftdim(avals(n_a(1)+1:sum(n_a(1:2))),-6);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-7);
            a4vals=shiftdim(avals(sum(n_a(1:3))+1:sum(n_a(1:4))),-8);
            if l_z==1
                zvals=shiftdim(zvals,-7);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-10);
                Fmatrix=arrayfun(ReturnFn, dvals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    elseif l_d==2
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        if l_a==1
            aprimevals=shiftdim(a_grid,-2);
            avals=shiftdim(avals,-3);
            if l_z==1
                zvals=shiftdim(zvals,-4);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-4);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-5);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-4);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-5);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-4);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-5);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-6);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-2);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2)),-3);
            a1vals=shiftdim(avals(1:n_a(1)),-4);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-5);
            if l_z==1
                zvals=shiftdim(zvals,-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-6);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-6);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-6);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-2);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-3);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-4);
            a1vals=shiftdim(avals(1:n_a(1)),-5);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-6);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-7);
            if l_z==1
                zvals=shiftdim(zvals,-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-8);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-8);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-8);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-2);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-3);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-4);
            a4primevals=shiftdim(a_grid(sum(n_aprime(1:3))+1:sum(n_aprime(1:4))),-5);
            a1vals=shiftdim(avals(1:n_a(1)),-6);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-7);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-8);
            a4vals=shiftdim(avals(sum(n_a(1:3))+1:sum(n_a(1:4))),-9);
            if l_z==1
                zvals=shiftdim(zvals,-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-10);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-10);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-10);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    elseif l_d==3
        d1vals=d_grid(1:n_d(1));
        d2vals=shiftdim(d_grid(n_d(1)+1:n_d(1)+n_d(2)),-1);
        d3vals=shiftdim(d_grid(n_d(2)+1:n_d(2)+n_d(3)),-2);
        if l_a==1
            aprimevals=shiftdim(a_grid,-3);
            avals=shiftdim(avals,-4);
            if l_z==1
                zvals=shiftdim(zvals,-5);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-5);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-5);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-5);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-6);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-7);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-3);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2)),-4);
            a1vals=shiftdim(avals(1:n_a(1)),-5);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-6);
            if l_z==1
                zvals=shiftdim(zvals,-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-7);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-8);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-9);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-3);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-4);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-5);
            a1vals=shiftdim(avals(1:n_a(1)),-6);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-7);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-8);
            if l_z==1
                zvals=shiftdim(zvals,-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-9);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-9);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-10);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-9);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-10);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-11);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-3);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-4);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-5);
            a4primevals=shiftdim(a_grid(sum(n_aprime(1:3))+1:sum(n_aprime(1:4))),-6);
            a1vals=shiftdim(avals(1:n_a(1)),-7);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-8);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-9);
            a4vals=shiftdim(avals(sum(n_a(1:3))+1:sum(n_a(1:4))),-10);
            if l_z==1
                zvals=shiftdim(zvals,-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-11);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-11);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-12);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-11);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-12);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-13);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-14);
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
                zvals=shiftdim(zvals,-6);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-6);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-6);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-6);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-7);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-8);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprimevals, avals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==2
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-4);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:n_aprime(1)+n_aprime(2)),-5);
            a1vals=shiftdim(avals(1:n_a(1)),-6);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-7);
            if l_z==1
                zvals=shiftdim(zvals,-8);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals, a1vals,a2vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-8);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-8);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-8);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-9);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-10);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==3
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-4);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-5);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-6);
            a1vals=shiftdim(avals(1:n_a(1)),-7);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-8);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-9);
            if l_z==1
                zvals=shiftdim(zvals,-10);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-10);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-11);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-10);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-10);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-11);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-12);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        elseif l_a==4
            a1primevals=shiftdim(a_grid(1:n_aprime(1)),-4);
            a2primevals=shiftdim(a_grid(n_aprime(1)+1:sum(n_aprime(1:2))),-5);
            a3primevals=shiftdim(a_grid(sum(n_aprime(1:2))+1:sum(n_aprime(1:3))),-6);
            a4primevals=shiftdim(a_grid(sum(n_aprime(1:3))+1:sum(n_aprime(1:4))),-7);
            a1vals=shiftdim(avals(1:n_a(1)),-8);
            a2vals=shiftdim(avals(n_a(1)+1:n_a(1)+n_a(2)),-9);
            a3vals=shiftdim(avals(sum(n_a(1:2))+1:sum(n_a(1:3))),-10);
            a4vals=shiftdim(avals(sum(n_a(1:3))+1:sum(n_a(1:4))),-11);
            if l_z==1
                zvals=shiftdim(zvals,-12);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, zvals,ParamCell{:});
            elseif l_z==2
                z1vals=shiftdim(zvals(1:n_z(1)),-12);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-13);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
            elseif l_z==3
                z1vals=shiftdim(zvals(1:n_z(1)),-12);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-13);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-14);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
            elseif l_z==4
                z1vals=shiftdim(zvals(1:n_z(1)),-12);
                z2vals=shiftdim(zvals(n_z(1)+1:n_z(1)+n_z(2)),-13);
                z3vals=shiftdim(zvals(sum(n_z(1:2))+1:sum(n_z(1:3))),-14);
                z4vals=shiftdim(zvals(sum(n_z(1:3))+1:sum(n_z(1:4))),-15);
                Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1primevals, a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
            end
        end
    end
    
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_aprime,N_a,N_z]);
    else
        Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_z]);
    end

end

end


