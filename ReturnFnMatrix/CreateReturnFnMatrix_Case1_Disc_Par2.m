function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2(ReturnFn, n_d, n_a, n_z, d_grid, a_grid, z_gridvals, ReturnFnParams,Refine) % Refine is an optional input
% would be good to make this version assume d (and d_gridvals) and use
% alternative version when no d variable but problem is that so many codes
% depend on this one would take ages to modify them all (all value fn
% commands without d would break, and all those with d would need to be
% switched to d_gridvals)

if size(d_grid,2)==1 % stacked-column % IN FUTURE, CHANGE INPUT TO BE d_gridvals
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else
    d_gridvals=d_grid;
end

if ~exist('Refine','var')
    Refine=0;
end

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
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
if l_z>5
    error('ERROR: Using GPU for the return fn does not allow for more than five of z variable (you have length(n_z)>5)')
end

if l_a>=1
    a1primevals=shiftdim(a_grid(1:n_a(1)),-1);
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_a-1);
    if l_a>=2
        a2primevals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-1-1);
        if l_a>=3
            a3primevals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-1-2);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-1-2);
            if l_a>=4
                a4primevals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-1-3);
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-1-3);
            end
        end
    end
end
% joint z_grid
if l_z>=1
    z1vals=shiftdim(z_gridvals(:,1),-1-l_a-l_a);
    if l_z>=2
        z2vals=shiftdim(z_gridvals(:,2),-1-l_a-l_a);
        if l_z>=3
            z3vals=shiftdim(z_gridvals(:,3),-1-l_a-l_a);
            if l_z>=4
                z4vals=shiftdim(z_gridvals(:,4),-1-l_a-l_a);
                if l_z>=5
                    z5vals=shiftdim(z_gridvals(:,5),-1-l_a-l_a);
                end
            end
        end
    end
end

if l_d==0 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==1
    % d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});  
elseif l_d==2 && l_a==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});  
elseif l_d==3 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});  
elseif l_d==3 && l_a==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});  
elseif l_d==4 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==5
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, ParamCell{:});
end

if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a,N_z]);
else
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a,N_z]); % This is the difference when using Refine
    else
        Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a,N_z]);
    end
end

end


