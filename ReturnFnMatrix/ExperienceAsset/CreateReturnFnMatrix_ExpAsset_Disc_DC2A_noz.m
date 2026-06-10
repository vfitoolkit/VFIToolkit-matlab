function Fmatrix=CreateReturnFnMatrix_ExpAsset_Disc_DC2A_noz(ReturnFn, n_d1, n_d2, n_a2, n_a3, d_gridvals, a1prime_grid, a2prime_gridvals, a1_grid, a2_gridvals, a3_gridvals, ReturnFnParamsVec, Level)
% _noz variant of CreateReturnFnMatrix_ExpAsset_Disc_DC2A: no exogenous z state.
%
% Note: d_gridvals is both d1 and d2 (unless n_d1=0 so there is no d1, in which case is just d2)
% a1: standard endogenous state which will have DC applied
% a2: standard endogenous state
% a3: experienceasset

% Level==1: a1prime_grid input column [N_a1prime, 1]; output [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3].
% Level==2: a1prime_grid input [N_d, maxgap+1, N_a2prime, 1, N_a2, N_a3]; output [N_d*N_a1prime*N_a2prime, N_a1*N_a2*N_a3] (3D for direct max).
% Level==3: a1prime_grid input same as Level=2; output [N_d, N_a1prime, N_a2prime, N_a1, N_a2, N_a3].

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

if n_d1(1)==0
    n_d=n_d2;
else
    n_d=[n_d1,n_d2];
end
N_d=prod(n_d);
N_a2=prod(n_a2);
N_a3=prod(n_a3);

l_d=length(n_d);
l_a2=length(n_a2);
l_a3=length(n_a3);

if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variables (you have length(n_d1)+length(n_d2)>4)')
end
if l_a2>3
    error('Using GPU for the return fn does not allow for more than three of folded a2 variables (DC2A noz ExpAsset)')
end
if l_a3>2
    error('experienceasset currently supports length(n_a3) in {1,2}')
end

if Level==1
    N_a1prime=size(a1prime_grid,1);
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2 || Level==3
    N_a1prime=size(a1prime_grid,2);
end
N_a2prime=N_a2;
N_a1=size(a1_grid,1);

if l_a2>=1
    a2prime1vals=shiftdim(a2prime_gridvals(:,1),-2);
    if l_a2>=2
        a2prime2vals=shiftdim(a2prime_gridvals(:,2),-2);
        if l_a2>=3
            a2prime3vals=shiftdim(a2prime_gridvals(:,3),-2);
        end
    end
end

a1vals=shiftdim(a1_grid,-3);

if l_a2>=1
    a21vals=shiftdim(a2_gridvals(:,1),-4);
    if l_a2>=2
        a22vals=shiftdim(a2_gridvals(:,2),-4);
        if l_a2>=3
            a23vals=shiftdim(a2_gridvals(:,3),-4);
        end
    end
end

if l_a3==1
    a3vals=shiftdim(a3_gridvals(:,1),-5);
elseif l_a3==2
    a3vals_1=shiftdim(a3_gridvals(:,1),-5);
    a3vals_2=shiftdim(a3_gridvals(:,2),-5);
end

if l_d==1
    if l_a2==1
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==2
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==3
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    end
elseif l_d==2
    if l_a2==1
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==2
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==3
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    end
elseif l_d==3
    if l_a2==1
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==2
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==3
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    end
elseif l_d==4
    if l_a2==1
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals, a1vals, a21vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==2
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals, a1vals, a21vals,a22vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    elseif l_a2==3
        if l_a3==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals, ReturnFnParamsCell{:});
        elseif l_a3==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, a2prime1vals,a2prime2vals,a2prime3vals, a1vals, a21vals,a22vals,a23vals, a3vals_1,a3vals_2, ReturnFnParamsCell{:});
        end
    end
end

if Level==1 || Level==3
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a2prime,N_a1,N_a2,N_a3]);
elseif Level==2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime*N_a2prime,N_a1*N_a2*N_a3]);
end

end
