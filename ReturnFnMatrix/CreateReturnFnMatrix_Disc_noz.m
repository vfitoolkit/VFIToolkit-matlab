function Fmatrix=CreateReturnFnMatrix_Disc_noz(ReturnFn, n_d, n_a, d_gridvals, a_grid, ReturnFnParamsVec,Refine) % Refine is an optional input
% If no d variable, just input n_d=0 and d_grid=[]

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_d=prod(n_d);
N_a=prod(n_a);

l_d=length(n_d);
if N_d==0
    l_d=0;
end
l_a=length(n_a);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
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

if l_d==0 && l_a==1
    Fmatrix=arrayfun(ReturnFn, aprime1vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==0 && l_a==2
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==0 && l_a==3
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==0 && l_a==4
    Fmatrix=arrayfun(ReturnFn, aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==1 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==2 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==3 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==1
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals, a1vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==2
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==3
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_d==4 && l_a==4
    Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime1vals,aprime2vals,aprime3vals,aprime4vals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
end

if l_d==0
    Fmatrix=reshape(Fmatrix,[N_a,N_a]);
else
    if Refine==1
        Fmatrix=reshape(Fmatrix,[N_d,N_a,N_a]); % This is the difference when using Refine
    else
        Fmatrix=reshape(Fmatrix,[N_d*N_a,N_a]);
    end
end

end


