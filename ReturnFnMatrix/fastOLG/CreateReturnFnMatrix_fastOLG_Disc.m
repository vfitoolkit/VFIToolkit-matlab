function Fmatrix=CreateReturnFnMatrix_fastOLG_Disc(ReturnFn, n_d, n_a, n_z, N_j, d_gridvals, aprime_grid, a_grid, z_gridvals_J, ReturnFnParamsAgeMatrix)
% Plain (non-DC, non-GI) fastOLG return matrix, with d, with z, no e.
% Handles l_d in {1..4}, l_a in {1..4}, l_z in {1..4}.
% Output: (N_d*N_aprime, N_a, N_j, N_z) where N_aprime=N_a=prod(n_a).

l_d=length(n_d);
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
l_a=length(n_a);
if l_a>4
    error('Using GPU for the return fn does not allow for more than four of a variable (you have length(n_a)>4)')
end
l_z=length(n_z);
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

N_d=prod(n_d);
N_a=prod(n_a);
N_aprime=N_a;
N_z=prod(n_z);

z_gridvals_J=reshape(z_gridvals_J,[N_j,N_z,l_z]);

% d on dim 1; aprime on dims l_d+1..l_d+l_a; a on dims l_d+l_a+1..l_d+2*l_a; j on dim l_d+2*l_a+1; z on dim l_d+2*l_a+2
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

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ReturnFnParamsCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ReturnFnParamsCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_d-2*l_a)};
end

% Helper macros: build the (aprime,a) arg sub-tuple and the z arg sub-tuple per l_a × l_z
% Then assemble: d_gridvals(:,1..l_d), aprime/a tuple, z tuple, params.
% Explicit if/elseif on l_d × l_a × l_z (64 branches).

if l_d==1
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_d==2
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_d==3
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    end
elseif l_d==4
    if l_a==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals, a1vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals, a1vals,a2vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals, a1vals,a2vals,a3vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    elseif l_a==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals, ReturnFnParamsCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals, ReturnFnParamsCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals, ReturnFnParamsCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1primevals,a2primevals,a3primevals,a4primevals, a1vals,a2vals,a3vals,a4vals, z1vals,z2vals,z3vals,z4vals, ReturnFnParamsCell{:});
        end
    end
end

Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_j,N_z]);

end
