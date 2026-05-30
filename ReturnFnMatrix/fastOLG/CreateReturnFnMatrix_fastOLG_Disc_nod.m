function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_nod(ReturnFn, n_a, n_z, N_j, aprime_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix)
% Plain (non-DC, non-GI) fastOLG return matrix, no d, with z, no e.
% Handles l_a in {1..4}, l_z in {1..4}.
% aprime_grid/a_grid: concatenated 1D vectors [a1_grid; a2_grid; ...]
% z_gridvals_J: any shape with prod(size)==N_j*N_z*l_z (reshapes to canonical (N_j,N_z,l_z) internally).
% Output: (N_aprime, N_a, N_j, N_z) where N_aprime=N_a=prod(n_a).

l_a=length(n_a);
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
l_z=length(n_z);
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

N_a=prod(n_a);
N_aprime=N_a;
N_z=prod(n_z);

% Restore canonical (N_j,N_z,l_z) regardless of caller pre-shifting
z_gridvals_J=reshape(z_gridvals_J,[N_j,N_z,l_z]);

% aprime on dims 1..l_a, a on dims l_a+1..2*l_a (no d, so no leading offset)
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

% z slices: j on dim 2*l_a+1, z (joint) on dim 2*l_a+2
z_shape=[ones(1,2*l_a) N_j N_z];
if l_z>=1, z1vals=reshape(z_gridvals_J(:,:,1),z_shape); end
if l_z>=2, z2vals=reshape(z_gridvals_J(:,:,2),z_shape); end
if l_z>=3, z3vals=reshape(z_gridvals_J(:,:,3),z_shape); end
if l_z>=4, z4vals=reshape(z_gridvals_J(:,:,4),z_shape); end

% Age-matrix params: shift to dim 2*l_a+1 (the j dim, aligned with z's j)
nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-2*l_a)};
end

if l_a==1
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals, ReturnFnParamsCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    end
elseif l_a==2
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals, ReturnFnParamsCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    end
elseif l_a==3
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ReturnFnParamsCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    end
elseif l_a==4
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ReturnFnParamsCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ReturnFnParamsCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
    end
end

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_j,N_z]);

end
