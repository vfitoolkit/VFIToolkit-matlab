function Fmatrix=CreateReturnFnMatrix_Case1_Disc_nod_Par2(ReturnFn, n_a, n_z, a_grid, z_gridvals, ReturnFnParams) % Refine is not relevant without d

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a); 
l_z=length(n_z);
if l_a>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>5
    error('ERROR: Using GPU for the return fn does not allow for more than five of z variable (you have length(n_z)>5)')
end

if l_a>=1
    aprime1vals=a_grid(1:n_a(1));
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_a);
    if l_a>=2
        aprime2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-1);
        if l_a>=3
            aprime3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-2);
            if l_a>=4
                aprime4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-3);
            end
        end
    end
end
% joint z_grid
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-l_a-l_a);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-l_a-l_a);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-l_a-l_a);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-l_a-l_a);
                if l_z>=5
                    z5vals=shiftdim(z_gridvals(:,5),-l_a-l_a);
                end
            end
        end
    end
end

if l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
end

Fmatrix=reshape(Fmatrix,[N_a,N_a,N_z]);

end


