function Fmatrix=CreateReturnFnMatrix_Case1_ExpAsset_Disc_DC1_Par2(ReturnFn, n_d, n_z, d_gridvals, a1prime_grid, a1_grid, a2_grid, z_gridvals, ReturnFnParams,Level) 
% Note: d is both d1 and d2 

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii,1)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_z=prod(n_z);

l_d=length(n_d);
if N_d==0
    error('With an experience asset there must be a decision variable')
end
l_a1=1; % hardcoded
l_z=length(n_z);
if l_d>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end
if l_z>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of z variable (you have length(n_z)>4)')
end
N_a1=length(a1_grid);
N_a2=length(a2_grid);

if Level==1
    N_a1prime=length(a1prime_grid); % Because l_a=1
    a1prime_grid=shiftdim(a1prime_grid,-1);
elseif Level==2
    N_a1prime=size(a1prime_grid,2); % Because l_a=1
end

expassetvals=shiftdim(a2_grid,-1-l_a1-l_a1);
if l_z==1
    if l_d==1 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==2 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==3 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==4 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), ParamCell{:});
    end
elseif l_z==2
    if l_d==1 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==2 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==3 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==4 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), ParamCell{:});
    end
elseif l_z==3
    if l_d==1 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==2 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==3 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==4 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), ParamCell{:});
    end
elseif l_z==4
    if l_d==1 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,4),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==2 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,4),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==3 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,4),-1-l_a1-l_a1-1), ParamCell{:});
    elseif l_d==4 && l_a1==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), a1prime_grid, shiftdim(a1_grid,-2), expassetvals, shiftdim(z_gridvals(:,1),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,2),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,3),-1-l_a1-l_a1-1), shiftdim(z_gridvals(:,4),-1-l_a1-l_a1-1), ParamCell{:});
    end
end

% Note: cannot have N_d=0 with experience asset
if Level==1
    Fmatrix=reshape(Fmatrix,[N_d,N_a1prime,N_a1,N_a2,N_z]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_a1prime,N_a1*N_a2,N_z]);
end


end


