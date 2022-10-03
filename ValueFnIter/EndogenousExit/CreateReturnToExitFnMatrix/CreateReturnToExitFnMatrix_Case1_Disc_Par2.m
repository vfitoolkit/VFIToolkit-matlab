function Fmatrix=CreateReturnToExitFnMatrix_Case1_Disc_Par2(ReturnFn, n_a, n_z, a_grid, z_grid, ReturnFnParams)
% Is same as 'CreateReturnFnMatrix' codes, but only on (a,z)

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    if size(ReturnFnParams(ii))~=[1,1]
        disp('ERROR: Using GPU for the return fn does not allow for any of ReturnFnParams to be anything but a scalar')
    end
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a); 
l_z=length(n_z);
if l_a>4
    disp('ERROR: Using GPU for the return to exit fn does not allow for more than four of a variable (you have length(n_a)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end
if l_z>4
    disp('ERROR: Using GPU for the return to exit fn does not allow for more than four of z variable (you have length(n_z)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end

if nargin(ReturnFn)~=l_a+l_z+length(ReturnFnParams)
    disp('ERROR: Number of inputs to ReturnToExitFn does not fit with size of ReturnToExitFnParamNames')
end

if l_a>=1
    a1vals=a_grid(1:n_a(1));
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        if l_a>=3
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
            if l_a>=4
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
            end
        end
    end
end
if all(size(z_grid)==[sum(n_z),1]) % kroneker product z_grid
    if l_z>=1
        z1vals=shiftdim(z_grid(1:n_z(1)),-l_a);
        if l_z>=2
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-l_a-1);
            if l_z>=3
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-l_a-2);
                if l_z>=4
                    z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-l_a-3);
                    if l_z>=5
                        z5vals=shiftdim(z_grid(sum(n_z(1:4))+1:sum(n_z(1:5))),-l_a-4);
                    end
                end
            end
        end
    end
elseif all(size(z_grid)==[prod(n_z),l_z]) % joint z_grid
    if l_z>=1
        z1vals=shiftdim(z_grid(:,1),-l_a);
        if l_z>=2
            z2vals=shiftdim(z_grid(:,2),-l_a);
            if l_z>=3
                z3vals=shiftdim(z_grid(:,3),-l_a);
                if l_z>=4
                    z4vals=shiftdim(z_grid(:,4),-l_a);
                    if l_z>=5
                        z5vals=shiftdim(z_grid(:,5),-l_a);
                    end
                end
            end
        end
    end
end

if l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1vals, z1vals, ParamCell{:});
elseif l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
end

Fmatrix=reshape(Fmatrix,[N_a,N_z]);

end


