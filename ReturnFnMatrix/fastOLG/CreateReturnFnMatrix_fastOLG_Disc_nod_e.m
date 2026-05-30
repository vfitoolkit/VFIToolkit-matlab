function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_nod_e(ReturnFn, n_a, n_z, n_e, N_j, aprime_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix)
% Plain (non-DC, non-GI) fastOLG return matrix, no d, with z, with e.
% Handles l_a in {1..4}, l_z in {1..4}, l_e in {1..4}.
% Output: (N_aprime, N_a, N_j, N_z, N_e) where N_aprime=N_a=prod(n_a).

l_a=length(n_a);
if l_a>4, error('Using GPU for the return fn does not allow for more than four of a variable'), end
l_z=length(n_z);
if l_z>4, error('Using GPU for the return fn does not allow for more than four of z variable'), end
l_e=length(n_e);
if l_e>4, error('Using GPU for the return fn does not allow for more than four of e variable'), end

N_a=prod(n_a); N_aprime=N_a; N_z=prod(n_z); N_e=prod(n_e);

z_gridvals_J=reshape(z_gridvals_J,[N_j,N_z,l_z]);
e_gridvals_J=reshape(e_gridvals_J,[N_j,N_e,l_e]);

% aprime on dims 1..l_a, a on dims l_a+1..2*l_a, j on dim 2*l_a+1, z on 2*l_a+2, e on 2*l_a+3
if l_a>=1
    a1primevals=aprime_grid(1:n_a(1));
    a1vals     =shiftdim(a_grid(1:n_a(1)),-l_a);
    if l_a>=2
        a2primevals=shiftdim(aprime_grid(n_a(1)+1:sum(n_a(1:2))),-1);
        a2vals     =shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_a-1);
        if l_a>=3
            a3primevals=shiftdim(aprime_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-2);
            a3vals     =shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_a-2);
            if l_a>=4
                a4primevals=shiftdim(aprime_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-3);
                a4vals     =shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_a-3);
            end
        end
    end
end

z_shape=[ones(1,2*l_a) N_j N_z];
if l_z>=1, z1vals=reshape(z_gridvals_J(:,:,1),z_shape); end
if l_z>=2, z2vals=reshape(z_gridvals_J(:,:,2),z_shape); end
if l_z>=3, z3vals=reshape(z_gridvals_J(:,:,3),z_shape); end
if l_z>=4, z4vals=reshape(z_gridvals_J(:,:,4),z_shape); end

e_shape=[ones(1,2*l_a) N_j 1 N_e];
if l_e>=1, e1vals=reshape(e_gridvals_J(:,:,1),e_shape); end
if l_e>=2, e2vals=reshape(e_gridvals_J(:,:,2),e_shape); end
if l_e>=3, e3vals=reshape(e_gridvals_J(:,:,3),e_shape); end
if l_e>=4, e4vals=reshape(e_gridvals_J(:,:,4),e_shape); end

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-2*l_a)};
end

if l_a==1
    if l_z==1
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==2
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==3
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==4
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_a==2
    if l_z==1
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==2
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==3
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==4
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_a==3
    if l_z==1
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==2
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==3
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==4
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_a==4
    if l_z==1
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==2
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==3
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    elseif l_z==4
        if     l_e==1, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals, ReturnFnParamsCell{:});
        elseif l_e==2, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals, ReturnFnParamsCell{:});
        elseif l_e==3, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals, ReturnFnParamsCell{:});
        elseif l_e==4, Fmatrix=arrayfun(ReturnFn, a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, e1vals,e2vals,e3vals,e4vals, ReturnFnParamsCell{:});
        end
    end
end

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_j,N_z,N_e]);

end
