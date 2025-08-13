function Fmatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_Par2(ReturnFn, n_d, n_a1,n_a2, n_z, d_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParams,Refine) % Refine is an optional input
% Note: d is both d1 and d2 

if ~exist('Refine','var')
    Refine=0;
end

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_z=prod(n_z);

l_d=length(n_d);
if N_d==0
    error('With an experience asset there must be a decision variable')
end
l_a1=length(n_a1); 
l_a2=length(n_a2); 
l_z=length(n_z);
if l_d>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a1>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_a2>1
    error('ERROR: Using GPU for the return fn does not allow for more than one experience asset variable')
end
if l_z>5
    error('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>5)')
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
if l_a1>=1
    aprime1vals=shiftdim(a1_grid(1:n_a1(1)),-l_d);
    a1vals=shiftdim(a1_grid(1:n_a1(1)),-l_a1-l_d);
    if l_a1>=2
        aprime2vals=shiftdim(a1_grid(n_a1(1)+1:sum(n_a1(1:2))),-l_d-1);
        a2vals=shiftdim(a1_grid(n_a1(1)+1:sum(n_a1(1:2))),-l_a1-l_d-1);
        if l_a1>=3
            aprime3vals=shiftdim(a1_grid(sum(n_a1(1:2))+1:sum(n_a1(1:3))),-l_d-2);
            a3vals=shiftdim(a1_grid(sum(n_a1(1:2))+1:sum(n_a1(1:3))),-l_a1-l_d-2);
            if l_a1>=4
                aprime4vals=shiftdim(a1_grid(sum(n_a1(1:3))+1:sum(n_a1(1:4))),-l_d-3);
                a4vals=shiftdim(a1_grid(sum(n_a1(1:3))+1:sum(n_a1(1:4))),-l_a1-l_d-3);
            end
        end
    end
end
expassetvals=shiftdim(a2_grid,-l_d-l_a1-l_a1);
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-l_d-l_a1-l_a1-l_a2);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-l_d-l_a1-l_a1-l_a2);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-l_d-l_a1-l_a1-l_a2);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-l_d-l_a1-l_a1-l_a2);
                if l_z>=5
                    z5vals=shiftdim(z_gridvals(:,5),-l_d-l_a1-l_a1-l_a2);
                end
            end
        end
    end
end


if l_d==1 && l_a1==1 && l_z==1
    d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==1 && l_a1==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a1==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a1==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a1==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a1==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==1 && l_a1==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a1==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a1==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a1==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a1==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==1 && l_a1==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a1==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a1==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a1==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a1==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==1 && l_a1==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a1==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a1==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a1==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a1==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==2 && l_a1==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a1==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a1==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a1==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a1==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==2 && l_a1==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a1==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a1==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a1==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a1==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==2 && l_a1==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a1==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a1==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a1==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a1==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==2 && l_a1==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a1==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a1==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});  
elseif l_d==2 && l_a1==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});  
elseif l_d==3 && l_a1==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==3 && l_a1==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a1==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a1==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a1==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a1==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==3 && l_a1==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a1==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a1==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a1==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a1==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==3 && l_a1==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a1==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a1==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a1==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a1==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==3 && l_a1==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a1==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a1==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});  
elseif l_d==3 && l_a1==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});  
elseif l_d==4 && l_a1==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==4 && l_a1==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a1==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a1==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a1==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a1==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==4 && l_a1==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a1==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a1==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a1==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a1==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==4 && l_a1==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a1==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a1==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a1==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a1==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals, ParamCell{:});
elseif l_d==4 && l_a1==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a1==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a1==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a1==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, expassetvals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
end

N_a1=prod(n_a1);
N_a2=prod(n_a2);

% Note: cannot have N_d=0 with experience asset
if Refine==1    
    Fmatrix=reshape(Fmatrix,[N_d,N_a1,N_a1*N_a2,N_z]); % This is the difference when using Refine
    % NOT ACTUALLY USABLE AS NEEDS ME TO CHANGE INPUTS SO n_d1 AND n_d2 ARE SEPARATE INPUTS (INSTEAD OF JUST n_d)
    % [Easy, but will break stuff and don't feel like fixing it all just now]
    % Fmatrix=reshape(Fmatrix,[N_d1,N_d2,N_a1,N_a1*N_a2,N_z]); % This is the difference when using Refine
else
    Fmatrix=reshape(Fmatrix,[N_d*N_a1,N_a1*N_a2,N_z]);
end


end


