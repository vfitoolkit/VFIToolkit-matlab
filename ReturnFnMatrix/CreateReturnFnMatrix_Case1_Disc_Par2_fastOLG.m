function Fmatrix=CreateReturnFnMatrix_Case1_Disc_Par2_fastOLG(ReturnFn, n_d, n_a, n_z, N_j, d_grid, a_grid, z_grid, ReturnFnParamsAgeMatrix)
%If there is no d variable, just input n_d=0 and d_grid=0

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

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);

if nargin(ReturnFn)~=l_d+l_a+l_a+l_z+nReturnFnParams
    disp('ERROR: Number of inputs to ReturnFn does not fit with size of ReturnFnParams')
end

ParamCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ParamCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_d-l_a-l_a)};
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
if l_z>=1 % Note that z_grid is an abnormal shape due to fastOLG so the z dimension is after the j dimension (for the return function params, and possibly to allow z_grid to also depend on j)
    z1vals=shiftdim(z_grid(:,1:n_z(1)),-l_d-l_a-l_a);
    if l_z>=2
        z2vals=permute(z_grid(:,n_z(1)+1:n_z(1)+n_z(2)),[1,3,2]); % Because of fastOLG we want to leave the j dimension where it is  % Keep 1st where it is, move 2nd to 3rd
        z2vals=shiftdim(z2vals,-l_d-l_a-l_a); % Note that the permute already does the role usually played by -1
        if l_z>=3
            z3vals=permute(z_grid(:,sum(n_z(1:2))+1:sum(n_z(1:3))),[1,3,4,2]); % Keep 1st where it is, move 2nd to 4th
            z3vals=shiftdim(z3vals,-l_d-l_a-l_a); % Note that the permute already does the role usually played by -2
            if l_z>=4
                z4vals=permute(z_grid(:,sum(n_z(1:3))+1:sum(n_z(1:4))),[1,3,4,5,2]); % Keep 1st where it is, move 2nd to 5th
                z4vals=shiftdim(z4vals,-l_d-l_a-l_a); % Note that the permute already does the role usually played by -3
            end
        end
    end
end

% Note that the dimensions of the following are d-aprime-a-j-z

if l_d==0 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==0 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==1
    d1vals(1,1,1,1)=d_grid(1); % Requires special treatment
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==1 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==2 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});  
elseif l_d==3 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==3 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});  
elseif l_d==4 && l_a==1 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==1 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals, a1vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==2 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==3 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ParamCell{:});
elseif l_d==4 && l_a==4 && l_z==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ParamCell{:});
end

if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a*N_j,N_z]);
else
    Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a*N_j,N_z]);
end


end


