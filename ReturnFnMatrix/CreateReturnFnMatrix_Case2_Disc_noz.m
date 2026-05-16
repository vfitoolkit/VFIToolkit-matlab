function Fmatrix=CreateReturnFnMatrix_Case2_Disc_noz(ReturnFn,n_d, n_a, d_gridvals, a_gridvals, ReturnFnParamsVec)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a=prod(n_a);

l_d=length(n_d);
l_a=length(n_a);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
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



if l_d==1 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d1vals,d2vals,d3vals,d4vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
end
Fmatrix=reshape(Fmatrix,[N_d,N_a]);



end


