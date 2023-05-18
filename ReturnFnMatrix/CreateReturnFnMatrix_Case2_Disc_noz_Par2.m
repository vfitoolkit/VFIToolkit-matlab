function Fmatrix=CreateReturnFnMatrix_Case2_Disc_noz_Par2(ReturnFn,n_d, n_a, d_grid, a_grid, ReturnFnParams)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    if size(ReturnFnParams(ii))~=[1,1]
        disp('ERROR: Using GPU for the return fn does not allow for any of ReturnFnParams to be anything but a scalar')
    end
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_a=prod(n_a);

l_d=length(n_d);
l_a=length(n_a); 
if l_d>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4): (in CreateReturnFnMatrix_Case2_Disc_Par2)')
end
if l_a>4
    disp('ERROR: Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4): (in CreateReturnFnMatrix_Case2_Disc_Par2)')
end

if nargin(ReturnFn)~=l_d+l_a+length(ReturnFnParams)
    disp('ERROR: Number of inputs to ReturnFn does not fit with size of ReturnFnParams')
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
    a1vals=shiftdim(a_grid(1:n_a(1)),-l_d);
    if l_a>=2
        a2vals=shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-1);
        if l_a>=3
            a3vals=shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-2);
            if l_a>=4
                a4vals=shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-3);
            end
        end
    end
end

% d1vals(1,1,1)=d_grid(1);

if l_d==1 && l_a==1
    d1vals(1,1,1)=d_grid(1); % Requires special treatment
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, ParamCell{:});
elseif l_d==1 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, ParamCell{:});
elseif l_d==1 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==1 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==2 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, ParamCell{:});
elseif l_d==2 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, ParamCell{:});
elseif l_d==2 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==2 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==3 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, ParamCell{:});
elseif l_d==3 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, ParamCell{:});
elseif l_d==3 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==3 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
elseif l_d==4 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, ParamCell{:});
elseif l_d==4 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, ParamCell{:});
elseif l_d==4 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, ParamCell{:});
elseif l_d==4 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, ParamCell{:});
end
Fmatrix=reshape(Fmatrix,[N_d,N_a]);



end


