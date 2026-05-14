function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_nod_Par2(ReturnFn, n_z, aprime_grid, a_grid, z_gridvals, ReturnFnParamsVec,Level)

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

N_a=length(a_grid); % Because l_a=1
N_z=prod(n_z);

% l_a=1; % (or else won't get here)
l_z=length(n_z); % won't get here if l_z=0
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end

N_aprime=size(aprime_grid,1); % Because l_a=1

if l_z==1
    Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), ReturnFnParamsCell{:});
elseif l_z==2
    Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), ReturnFnParamsCell{:});
elseif l_z==3
    Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), ReturnFnParamsCell{:});
elseif l_z==4
    Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), shiftdim(z_gridvals(:,1),-2), shiftdim(z_gridvals(:,2),-2), shiftdim(z_gridvals(:,3),-2), shiftdim(z_gridvals(:,4),-2), ReturnFnParamsCell{:});
end

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_z]);

end


