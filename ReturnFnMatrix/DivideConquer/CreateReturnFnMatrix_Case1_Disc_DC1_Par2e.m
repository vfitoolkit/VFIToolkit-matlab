function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_Par2e(ReturnFn, n_d, n_z, n_e, d_gridvals, aprime_grid, a_grid, z_gridvals, e_gridvals, ReturnFnParams, Level)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_a=length(a_grid); % Because l_a=1
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d); % won't get here if l_d=0
% l_a=1; % (or else won't get here)
l_z=length(n_z); % won't get here if l_z=0
l_e=length(n_e); % won't get here if l_e=0
if l_d>4
    error('Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_z>4
    error('Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
if l_e>4
    error('Using GPU for the return fn does not allow for more than four of e variable (you have length(n_e)>4)')
end

if Level==1
    N_aprime=length(aprime_grid); % Because l_a=1
    aprime_grid=shiftdim(aprime_grid,-1);
elseif Level==2
    N_aprime=size(aprime_grid,2); % Because l_a=1
    % aprime_grid unchanged
elseif Level==3 % for when doing gridded interpolation layer
    % Same as level 1, but without shiftdim() on aprime_grid
    N_aprime=length(aprime_grid); % Because l_a=1
elseif Level==4
    N_aprime=length(aprime_grid); % Because l_a=1
    aprime_grid=shiftdim(aprime_grid,-1);
    % Level 4 is just version of Level 1 for semi when looping over d2
elseif Level==5
    N_aprime=size(aprime_grid,1); % Because l_a=1
    aprime_grid=shiftdim(aprime_grid,-1);
    % Level 5 is just version of Level 2 for semi when looping over d2
elseif Level==6 % DC+GI with d1
    % Level 2 inputs with Level 3 outputs
    N_aprime=size(aprime_grid,2); % Because l_a=1
end

if l_e==1
    if l_z==1
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        end
    elseif l_z==2
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        end
    elseif l_z==3
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        end
    elseif l_z==4
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), ParamCell{:});
        end
    end
elseif l_e==2
    if l_z==1
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        end
    elseif l_z==2
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        end
    elseif l_z==3
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        end
    elseif l_z==4
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), ParamCell{:});
        end
    end
elseif l_e==3
    if l_z==1
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        end
    elseif l_z==2
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        end
    elseif l_z==3
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        end
    elseif l_z==4
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), ParamCell{:});
        end
    end
elseif l_e==4
    if l_z==1
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        end
    elseif l_z==2
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        end
    elseif l_z==3
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        end
    elseif l_z==4
        if l_d==1
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==2
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==3
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        elseif l_d==4
            Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), aprime_grid, shiftdim(a_grid,-2), shiftdim(z_gridvals(:,1),-3), shiftdim(z_gridvals(:,2),-3), shiftdim(z_gridvals(:,3),-3), shiftdim(z_gridvals(:,4),-3), shiftdim(e_gridvals(:,1),-4), shiftdim(e_gridvals(:,2),-4), shiftdim(e_gridvals(:,3),-4), shiftdim(e_gridvals(:,4),-4), ParamCell{:});
        end
    end
end


if Level==1 % For level 1
    Fmatrix=reshape(Fmatrix,[N_d,N_aprime,N_a,N_z,N_e]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_z,N_e]);
elseif Level==3 % For GI
    Fmatrix=reshape(Fmatrix,[N_d,N_aprime,N_a,N_z,N_e]);
elseif Level==4 % Level 4 is version of Level 1 for looping over d2 with semiz, so there is a single point for n_d
    Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_z,N_e]);
elseif Level==5 % Level 5 is version of Level 2 for looping over d2 with semiz, so there is a single point for n_d
    Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a,N_z,N_e]);
elseif Level==6 % Level 6 is for DC+GI with d1
    Fmatrix=reshape(Fmatrix,[N_d,N_aprime,N_a,N_z,N_e]);
end


end


