function Fmatrix=CreateReturnFnMatrix_Case2_Disc_Par2e(ReturnFn,n_d, n_a, n_z, n_e, d_gridvals, a_gridvals, z_gridvals, e_gridvals, ReturnFnParams)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    if size(ReturnFnParams(ii))~=[1,1]
        error('Using GPU for the return fn does not allow for any of ReturnFnParams to be anything but a scalar')
    end
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

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
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>5
    error('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
if l_e>5
    error('ERROR: Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)')
end

if nargin(ReturnFn)~=l_d+l_a+l_z+l_e+length(ReturnFnParams)
    error('ERROR: Number of inputs to ReturnFn does not fit with size of ReturnFnParams')
end


if l_d>=1
    d1vals=d_gridvals(:,1);
    if l_d>=2
        d2vals=d_gridvals(:,2);
        if l_d>=3
            d3vals=d_gridvals(:,3);
            if l_d>=4
                d4vals=d_gridvals(:,4);
            end
        end
    end
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
            d1vals(1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        end     
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,ParamCell{:});
        end
    end
elseif l_e==2
    if l_d==1
        if l_a==1 && l_z==1
            d1vals(1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        end     
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,ParamCell{:});
        end
    end
elseif l_e==3
    if l_d==1
        if l_a==1 && l_z==1
            d1vals(1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        end     
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,ParamCell{:});
        end
    end
elseif l_e==4
    if l_d==1
        if l_a==1 && l_z==1
            d1vals(1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        end     
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,ParamCell{:});
        end
    end
elseif l_e==5
    if l_d==1
        if l_a==1 && l_z==1
            d1vals(1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        end     
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals,ParamCell{:});
        end
    end
end

Fmatrix=reshape(Fmatrix,[N_d,N_a,N_z,N_e]); % this is probably obsolete given they are all based on gridvals


end


