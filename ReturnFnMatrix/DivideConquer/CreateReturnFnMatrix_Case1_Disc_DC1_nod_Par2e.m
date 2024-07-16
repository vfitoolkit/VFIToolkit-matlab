function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2e(ReturnFn, n_z, n_e, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams,Level)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii)={ReturnFnParams(ii)};
end

N_a=length(a_grid); % Because l_a=1
N_z=prod(n_z);
N_e=prod(n_e);

% l_a=1; % (or else won't get here)
l_z=length(n_z); % won't get here if l_z=0
l_e=length(n_e); % won't get here if l_e=0
if l_z>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
if l_e>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)')
end

if Level==1
    N_aprime=length(aprime_grid); % Because l_a=1
elseif Level==2
    N_aprime=size(aprime_grid,1); % Because l_a=1
    % aprime_grid unchanged
end


if l_e==1
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), ParamCell{:});
    end
elseif l_e==2
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), ParamCell{:});
    end
elseif l_e==3
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), ParamCell{:});
    end
elseif l_e==4
    if l_z==1
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
    elseif l_z==2
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
    elseif l_z==3
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
    elseif l_z==4
        Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), shiftdim(e_gridvals(:,1),-3), shiftdim(e_gridvals(:,2),-3), shiftdim(e_gridvals(:,3),-3), shiftdim(e_gridvals(:,4),-3), ParamCell{:});
    end
end

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_z,N_e]);


end


