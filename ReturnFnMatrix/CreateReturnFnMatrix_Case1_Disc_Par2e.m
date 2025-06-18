function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2e(ReturnFn, n_d, n_a, n_z, n_e, d_grid, a_grid, z_grid, e_grid, ReturnFnParams, Refine)
% If there is no d variable, just input n_d=0
% Same as CreateReturnFnMatrix_Case1_Disc_Par2, except with e variable

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

if l_a>=1
    aprime1vals=shiftdim(a_grid(1:n_a(1)),-1);
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_a-1);
    if l_a>=2
        aprime2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1-1);
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-1-1);
        if l_a>=3
            aprime3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-1-2);
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-1-2);
            if l_a>=4
                aprime4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-1-3);
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-1-3);
            end
        end
    end
end

% joint z_grid
if l_z>=1
    z1vals=shiftdim(z_grid(:,1),-1-l_a-l_a);
    if l_z>=2
        z2vals=shiftdim(z_grid(:,2),-1-l_a-l_a);
        if l_z>=3
            z3vals=shiftdim(z_grid(:,3),-1-l_a-l_a);
            if l_z>=4
                z4vals=shiftdim(z_grid(:,4),-1-l_a-l_a);
                if l_z>=5
                    z5vals=shiftdim(z_grid(:,5),-1-l_a-l_a);
                end
            end
        end
    end
end

% joint e_grid
if l_e>=1
    e1vals=shiftdim(e_grid(:,1),-1-l_a-l_a-1);
    if l_e>=2
        e2vals=shiftdim(e_grid(:,2),-1-l_a-l_a-1);
        if l_e>=3
            e3vals=shiftdim(e_grid(:,3),-1-l_a-l_a-1);
            if l_e>=4
                e4vals=shiftdim(e_grid(:,4),-1-l_a-l_a-1);
                if l_e>=5
                    e5vals=shiftdim(e_grid(:,5),-1-l_a-l_a-1);
                end
            end
        end
    end
end

if l_e==1
    if l_d==0
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        end
    elseif l_d==1
        if l_a==1 && l_z==1
            d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals, e1vals, ParamCell{:});
        elseif  l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif  l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif  l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif  l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif  l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
        elseif  l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif  l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif  l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif  l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif  l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
        elseif  l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif  l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif  l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif  l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif  l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
        elseif  l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif  l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif  l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif  l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals, ParamCell{:});
        end
    end
elseif l_e==2
    if l_d==0
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        end
    elseif l_d==1
        if l_a==1 && l_z==1
            d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif  l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals, ParamCell{:});
        end
    end
elseif l_e==3
    if l_d==0
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        end
    elseif l_d==1
        if l_a==1 && l_z==1
            d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif  l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals, ParamCell{:});
        end
    end
elseif l_e==4
    if l_d==0
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        end
    elseif l_d==1
        if l_a==1 && l_z==1
            d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif  l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals, ParamCell{:});
        end
    end
elseif l_e==5
    if l_d==0
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        end
    elseif l_d==1
        if l_a==1 && l_z==1
            d_gridvals(:,1)(1,1,1,1)=d_grid(1); % Requires special treatment
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        end
    elseif l_d==2
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        end
    elseif l_d==3
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif  l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        end
    elseif l_d==4
        if l_a==1 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==1 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==2 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==3 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        elseif l_a==4 && l_z==5
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals, e1vals,e2vals,e3vals,e4vals,e5vals, ParamCell{:});
        end
    end
end


if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a,N_z,N_e]);
else
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a,N_z,N_e]);
    else
        Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a,N_z,N_e]);
    end
end



end


