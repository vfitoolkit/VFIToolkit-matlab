function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_noz(ReturnFn, n_d, n_a, N_j, d_gridvals, aprime_grid, a_grid, ReturnFnParamsAgeMatrix)
% Plain (non-DC, non-GI) fastOLG return matrix, with d, no z, no e.
% Handles l_d in {1..4}, l_a in {1..4}.
% d_gridvals: (N_d, l_d). aprime_grid/a_grid: concatenated 1D vectors [a1_grid; a2_grid; ...]
% Output: (N_d*N_aprime, N_a, N_j) where N_aprime=N_a=prod(n_a).

l_d=length(n_d);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_a=length(n_a);
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end

N_d=prod(n_d);
N_a=prod(n_a);
N_aprime=N_a;

% d on dim 1 (joint, all l_d slices share dim 1 by being column vecs of size N_d)
% aprime on dims l_d+1..l_d+l_a; a on dims l_d+l_a+1..l_d+2*l_a; j on dim l_d+2*l_a+1
if l_a>=1
    a1primevals=shiftdim(aprime_grid(1:n_a(1)),-l_d);                                          % dim l_d+1
    a1vals     =shiftdim(a_grid(1:n_a(1)),-l_d-l_a);                                            % dim l_d+l_a+1
    if l_a>=2
        a2primevals=shiftdim(aprime_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-1);                        % dim l_d+2
        a2vals     =shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-l_a-1);                         % dim l_d+l_a+2
        if l_a>=3
            a3primevals=shiftdim(aprime_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-2);             % dim l_d+3
            a3vals     =shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-l_a-2);              % dim l_d+l_a+3
            if l_a>=4
                a4primevals=shiftdim(aprime_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-3);         % dim l_d+4
                a4vals     =shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-l_a-3);          % dim l_d+l_a+4
            end
        end
    end
end

% Age-matrix params shifted to j dim (dim l_d+2*l_a+1)
nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_d-2*l_a)};
end

if l_d==1
    if l_a==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, ReturnFnParamsCell{:});
    elseif l_a==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, ReturnFnParamsCell{:});
    elseif l_a==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
    elseif l_a==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
    end
elseif l_d==2
    if l_a==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, ReturnFnParamsCell{:});
    elseif l_a==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, ReturnFnParamsCell{:});
    elseif l_a==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
    elseif l_a==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
    end
elseif l_d==3
    if l_a==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, ReturnFnParamsCell{:});
    elseif l_a==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, ReturnFnParamsCell{:});
    elseif l_a==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
    elseif l_a==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
    end
elseif l_d==4
    if l_a==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, ReturnFnParamsCell{:});
    elseif l_a==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, ReturnFnParamsCell{:});
    elseif l_a==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
    elseif l_a==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
    end
end

Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_j]);

end
