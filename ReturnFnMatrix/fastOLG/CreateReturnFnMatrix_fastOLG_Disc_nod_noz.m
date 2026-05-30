function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_nod_noz(ReturnFn, n_a, N_j, aprime_grid, a_grid, ReturnFnParamsAgeMatrix)
% Plain (non-DC, non-GI) fastOLG return matrix, no d, no z, handles l_a in {1,2,3,4}.
% aprime_grid and a_grid are concatenated 1D vectors per dim: [a1_grid; a2_grid; ...].
% For plain, typically aprime_grid == a_grid.
% Output: (N_aprime, N_a, N_j) where N_aprime=N_a=prod(n_a).

l_a=length(n_a);
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end

N_a=prod(n_a);
N_aprime=N_a;

% aprime occupies dims 1..l_a; a occupies dims l_a+1..2*l_a; age (j) on dim 2*l_a+1
if l_a>=1
    a1primevals=aprime_grid(1:n_a(1));                                            % dim 1
    a1vals     =shiftdim(a_grid(1:n_a(1)),-l_a);                                  % dim l_a+1
    if l_a>=2
        a2primevals=shiftdim(aprime_grid(n_a(1)+1:sum(n_a(1:2))),-1);             % dim 2
        a2vals     =shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-1);              % dim l_a+2
        if l_a>=3
            a3primevals=shiftdim(aprime_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);  % dim 3
            a3vals     =shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-2);   % dim l_a+3
            if l_a>=4
                a4primevals=shiftdim(aprime_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);  % dim 4
                a4vals     =shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-3);   % dim l_a+4
            end
        end
    end
end

% Age-matrix params: each parameter (one column) shifts to dim 2*l_a+1
nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-2*l_a)};
end

if l_a==1
    Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, ReturnFnParamsCell{:});
elseif l_a==2
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, ReturnFnParamsCell{:});
elseif l_a==3
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, ReturnFnParamsCell{:});
elseif l_a==4
    Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, ReturnFnParamsCell{:});
end

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_j]);

end
