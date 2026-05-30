function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc_e(ReturnFn, n_d, n_a, n_z, n_e, N_j, d_gridvals, aprime_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix)
% Plain (non-DC, non-GI) fastOLG return matrix, with d, with z, with e.
% Handles l_d in {1..4}, l_a in {1..4}, l_z in {1..4}, l_e in {1..4}.
% Output: (N_d*N_aprime, N_a, N_j, N_z, N_e) where N_aprime=N_a=prod(n_a).
%
% NOTE: With l_d x l_a x l_z x l_e = 256 (d,a,z,e)-combinations the explicit if/elseif
% explosion exceeds 500 lines for no benefit. This file uses cell-expansion to build the
% arrayfun argument list once, which is semantically identical to one explicit arrayfun
% call per combination. The MATLAB syntax `arrayfun(f, cellargs{:})` expands the cell
% into a comma-separated arg list, exactly as if it had been written out by hand.

l_d=length(n_d);
if l_d>4, error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)'), end
l_a=length(n_a);
if l_a>4, error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)'), end
l_z=length(n_z);
if l_z>4, error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)'), end
l_e=length(n_e);
if l_e>4, error('Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)'), end

N_d=prod(n_d); N_a=prod(n_a); N_aprime=N_a; N_z=prod(n_z); N_e=prod(n_e);

z_gridvals_J=reshape(z_gridvals_J,[N_j,N_z,l_z]);
e_gridvals_J=reshape(e_gridvals_J,[N_j,N_e,l_e]);

% d on dim 1 (all l_d slices share dim 1 via column vec broadcast)
% aprime on dims l_d+1..l_d+l_a; a on dims l_d+l_a+1..l_d+2*l_a
% j on dim l_d+2*l_a+1; z on dim l_d+2*l_a+2; e on dim l_d+2*l_a+3
if l_a>=1
    a1primevals=shiftdim(aprime_grid(1:n_a(1)),-l_d);
    a1vals     =shiftdim(a_grid(1:n_a(1)),-l_d-l_a);
    if l_a>=2
        a2primevals=shiftdim(aprime_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-1);
        a2vals     =shiftdim(a_grid(n_a(1)+1:sum(n_a(1:2))),-l_d-l_a-1);
        if l_a>=3
            a3primevals=shiftdim(aprime_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-2);
            a3vals     =shiftdim(a_grid(sum(n_a(1:2))+1:sum(n_a(1:3))),-l_d-l_a-2);
            if l_a>=4
                a4primevals=shiftdim(aprime_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-3);
                a4vals     =shiftdim(a_grid(sum(n_a(1:3))+1:sum(n_a(1:4))),-l_d-l_a-3);
            end
        end
    end
end

z_shape=[ones(1,l_d+2*l_a) N_j N_z];
if l_z>=1, z1vals=reshape(z_gridvals_J(:,:,1),z_shape); end
if l_z>=2, z2vals=reshape(z_gridvals_J(:,:,2),z_shape); end
if l_z>=3, z3vals=reshape(z_gridvals_J(:,:,3),z_shape); end
if l_z>=4, z4vals=reshape(z_gridvals_J(:,:,4),z_shape); end

e_shape=[ones(1,l_d+2*l_a) N_j 1 N_e];
if l_e>=1, e1vals=reshape(e_gridvals_J(:,:,1),e_shape); end
if l_e>=2, e2vals=reshape(e_gridvals_J(:,:,2),e_shape); end
if l_e>=3, e3vals=reshape(e_gridvals_J(:,:,3),e_shape); end
if l_e>=4, e4vals=reshape(e_gridvals_J(:,:,4),e_shape); end

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_d-2*l_a)};
end

% Build arg list per dim count
dargs=cell(1,l_d);
for k=1:l_d, dargs{k}=d_gridvals(:,k); end

aprimeargs={a1primevals};
if l_a>=2, aprimeargs{end+1}=a2primevals; end
if l_a>=3, aprimeargs{end+1}=a3primevals; end
if l_a>=4, aprimeargs{end+1}=a4primevals; end

aargs={a1vals};
if l_a>=2, aargs{end+1}=a2vals; end
if l_a>=3, aargs{end+1}=a3vals; end
if l_a>=4, aargs{end+1}=a4vals; end

zargs={z1vals};
if l_z>=2, zargs{end+1}=z2vals; end
if l_z>=3, zargs{end+1}=z3vals; end
if l_z>=4, zargs{end+1}=z4vals; end

eargs={e1vals};
if l_e>=2, eargs{end+1}=e2vals; end
if l_e>=3, eargs{end+1}=e3vals; end
if l_e>=4, eargs{end+1}=e4vals; end

Fmatrix=arrayfun(ReturnFn, dargs{:}, aprimeargs{:}, aargs{:}, zargs{:}, eargs{:}, ReturnFnParamsCell{:});

Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_j,N_z,N_e]);

end
