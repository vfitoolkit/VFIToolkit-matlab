function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2_SDP(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParams, SDP1, SDP2,SDP3)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a); 
l_z=length(n_z);
if l_d>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end
if l_a>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end
if l_z>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4): (in CreateReturnFnMatrix_Case1_Disc_Par2)')
end

if l_d>=1
    d1vals=d_grid(1:n_d(1)); 
    if l_d>=2
        d2vals=shiftdim(d_grid(n_d(1)+1:sum(n_d(1:2))),-1);
        if l_d>=3
            d3vals=shiftdim(d_grid(sum(n_d(1:2))+1:sum(n_d(1:3))),-2);
            if l_d>=4
                d4vals=shiftdim(d_grid(sum(n_d(1:3))+1:sum(n_d(1:4))),-3);
            end
        end
    end
end
if l_a>=1
    aprime1vals=shiftdim(a_grid(1:n_a(1)),-l_d);
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_a-l_d);
    if l_a>=2
        aprime2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-l_d-1);
        if l_a>=3
            aprime3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-2);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-l_d-2);
            if l_a>=4
                aprime4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-3);
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-l_d-3);
            end
        end
    end
end
if all(size(z_grid)==[sum(n_z),1]) % kroneker product z_grid
    if l_z>=1
        z1vals=shiftdim(z_grid(1:n_z(1)),-l_d-l_a-l_a);
        if l_z>=2
            z2vals=shiftdim(z_grid(n_z(1)+1:n_z(1)+n_z(2)),-l_d-l_a-l_a-1);
            if l_z>=3
                z3vals=shiftdim(z_grid(sum(n_z(1:2))+1:sum(n_z(1:3))),-l_d-l_a-l_a-2);
                if l_z>=4
                    z4vals=shiftdim(z_grid(sum(n_z(1:3))+1:sum(n_z(1:4))),-l_d-l_a-l_a-3);
                    if l_z>=5
                        z5vals=shiftdim(z_grid(sum(n_z(1:4))+1:sum(n_z(1:5))),-l_d-l_a-l_a-4);
                    end
                end
            end
        end
    end
elseif all(size(z_grid)==[prod(n_z),l_z]) % joint z_grid
    if l_z>=1
        z1vals=shiftdim(z_grid(:,1),-l_d-l_a-l_a);
        if l_z>=2
            z2vals=shiftdim(z_grid(:,2),-l_d-l_a-l_a);
            if l_z>=3
                z3vals=shiftdim(z_grid(:,3),-l_d-l_a-l_a);
                if l_z>=4
                    z4vals=shiftdim(z_grid(:,4),-l_d-l_a-l_a);
                    if l_z>=5
                        z5vals=shiftdim(z_grid(:,5),-l_d-l_a-l_a);
                    end
                end
            end
        end
    end
end


if exists(SDP3)
    if l_d==0 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1
        d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2,SDP3, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2,SDP3, ParamCell{:});
    end
elseif exists(SDP2)
    if l_d==0 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1
        d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1,SDP2, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1,SDP2, ParamCell{:});
    end
else % Just SDP1
    if l_d==0 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==0 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==1
        d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==1 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==2 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==3 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, SDP1, ParamCell{:});
    elseif l_d==4 && l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, SDP1, ParamCell{:});
    end
end



%%
if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a,N_z]);
else
    Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a,N_z]);
end


end


