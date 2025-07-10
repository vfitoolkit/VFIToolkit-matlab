function Fmatrix=CreateReturnFnMatrix_Case2_Disc_Par2(ReturnFn,n_d, n_a, n_z, d_grid, a_grid, z_grid, ReturnFnParams)
%If there is no d variable, just input n_d=0 and d_grid=0

if size(d_grid,2)==1 % stacked-column % IN FUTURE, CHANGE INPUT TO BE d_gridvals
    d_gridvals=CreateGridvals(n_d,d_grid,1);
else
    d_gridvals=d_grid;
end

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

l_d=length(n_d);
l_a=length(n_a); 
l_z=length(n_z);
if l_d>4
    error('=Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4): (in CreateReturnFnMatrix_Case2_Disc_Par2)')
end
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4): (in CreateReturnFnMatrix_Case2_Disc_Par2)')
end
if l_z>8
    error('Using GPU for the return fn does not allow for more than four of z variable (plus four of semiz): (in CreateReturnFnMatrix_Case2_Disc_Par2)')
end

if nargin(ReturnFn)~=l_d+l_a+l_z+length(ReturnFnParams)
    error('Number of inputs to ReturnFn does not fit with size of ReturnFnParams')
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
    a1vals=shiftdim(a_grid(1:n_a(1)),-1);
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-1-1);
        if l_a>=3
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-1-2);
            if l_a>=4
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-1-3);
            end
        end
    end
end

% joint z_grid
if l_z>=1
    z1vals=shiftdim(z_grid(:,1),-1-l_a);
    if l_z>=2
        z2vals=shiftdim(z_grid(:,2),-1-l_a);
        if l_z>=3
            z3vals=shiftdim(z_grid(:,3),-1-l_a);
            if l_z>=4
                z4vals=shiftdim(z_grid(:,4),-1-l_a);
                if l_z>=5
                    z5vals=shiftdim(z_grid(:,5),-1-l_a);
                    if l_z>=6
                        z6vals=shiftdim(z_grid(:,6),-l_d-l_a);
                        if l_z>=7
                            z7vals=shiftdim(z_grid(:,7),-1-l_a);
                            if l_z>=8
                                z8vals=shiftdim(z_grid(:,8),-1-l_a);
                            end
                        end
                    end
                end
            end
        end
    end
end

if l_d==1
    if l_a==1 && l_z==1
        % d1vals(1,1,1)=d_grid(1); % Requires special treatment
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,ParamCell{:});
    elseif l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==1 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==1 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==1 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,ParamCell{:});
    elseif l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==2 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==2 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==2 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,ParamCell{:});
    elseif l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==3 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==3 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==3 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,ParamCell{:});
    elseif l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==4 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==4 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==4 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    end
elseif l_d==2
    if l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,ParamCell{:});
    elseif l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==1 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==1 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==1 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,ParamCell{:});
    elseif l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==2 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==2 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==2 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,ParamCell{:});
    elseif l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==3 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==3 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==3 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,ParamCell{:});
    elseif l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==4 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==4 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==4 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    end
elseif l_d==3
    if l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,ParamCell{:});
    elseif l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==1 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==1 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==1 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,ParamCell{:});
    elseif l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==2 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==2 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==2 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,ParamCell{:});
    elseif l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==3 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==3 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==3 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,ParamCell{:});
    elseif l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==4 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==4 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==4 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    end
elseif l_d==4
    if l_a==1 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,ParamCell{:});
    elseif l_a==1 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==1 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==1 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==1 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==1 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==1 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==1 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==2 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,ParamCell{:});
    elseif l_a==2 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==2 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==2 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==2 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==2 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==2 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==2 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==3 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,ParamCell{:});
    elseif l_a==3 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==3 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==3 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==3 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==3 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==3 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==3 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    elseif l_a==4 && l_z==1
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,ParamCell{:});
    elseif l_a==4 && l_z==2
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,ParamCell{:});
    elseif l_a==4 && l_z==3
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,ParamCell{:});
    elseif l_a==4 && l_z==4
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,ParamCell{:});
    elseif l_a==4 && l_z==5
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,ParamCell{:});
    elseif l_a==4 && l_z==6
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,ParamCell{:});
    elseif l_a==4 && l_z==7
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,ParamCell{:});
    elseif l_a==4 && l_z==8
        Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals,z5vals,z6vals,z7vals,z8vals,ParamCell{:});
    end
end


Fmatrix=reshape(Fmatrix,[N_d,N_a,N_z]);


end


