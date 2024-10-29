function Fmatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_Par2e(ReturnFn, n_z, n_e, N_j, aprime_grid, a_grid, z_gridvals_J, e_gridvals_J, ReturnFnParamsAgeMatrix,Level)

l_a=1; % (or else won't get here)
l_z=length(n_z); % won't get here if l_z=0
if l_z>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
l_e=length(n_e); % won't get here if l_e=0
if l_e>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)')
end


nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ParamCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ParamCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_a-l_a)};
end

N_a=length(a_grid); % Because l_a=1
if Level==1
    N_aprime=length(aprime_grid);
elseif Level==2
    N_aprime=size(aprime_grid,1); % Because l_a=1
end
N_z=prod(n_z);
N_e=prod(n_e);

if Level==1
    if l_e==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        end
    elseif l_e==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        end
    elseif l_e==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        end
    elseif l_e==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        end
    end
elseif Level==2
    if l_e==1
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), ParamCell{:});
        end
    elseif l_e==2
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), ParamCell{:});
        end
    elseif l_e==3
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), ParamCell{:});
        end
    elseif l_e==4
        if l_z==1
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        elseif l_z==2
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        elseif l_z==3
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        elseif l_z==4
            Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), z_gridvals_J(1,1,:,:,1), z_gridvals_J(1,1,:,:,2), z_gridvals_J(1,1,:,:,3), z_gridvals_J(1,1,:,:,4), e_gridvals_J(1,1,:,1,:,1), e_gridvals_J(1,1,:,1,:,2), e_gridvals_J(1,1,:,1,:,3), e_gridvals_J(1,1,:,1,:,4), ParamCell{:});
        end
    end
end

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_j,N_z,N_e]);

end


