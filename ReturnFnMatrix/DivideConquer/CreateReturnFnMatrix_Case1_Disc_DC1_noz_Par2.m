function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_noz_Par2(ReturnFn, n_d, d_gridvals, aprime_grid, a_grid, ReturnFnParams,Level)
%If there is no d variable, just input n_d=0 and d_grid=0

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii)={ReturnFnParams(ii)};
end

N_d=prod(n_d);
N_a=length(a_grid); % Because l_a=1
if Level==1
    N_aprime=length(aprime_grid);
elseif Level==2
    N_aprime=size(aprime_grid,2); % Because l_a=1
end

l_d=length(n_d); % won't get here if l_d=0
l_a=1; % (or else won't get here)
if l_d>4
    error('ERROR: Using GPU for the return fn does not allow for more than four of d variable (you have length(n_d)>4)')
end

if Level==1
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    end
elseif Level==2
    if l_d==1
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1), aprime_grid, shiftdim(a_grid,-2), ParamCell{:});
    elseif l_d==2
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    elseif l_d==3
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    elseif l_d==4
        Fmatrix=arrayfun(ReturnFn, d_gridvals(:,1),d_gridvals(:,2),d_gridvals(:,3),d_gridvals(:,4), shiftdim(aprime_grid,-1), shiftdim(a_grid,-2), ParamCell{:});
    end

end

if Level==1 % For level 1
    Fmatrix=reshape(Fmatrix,[N_d,N_aprime,N_a]);
elseif Level==2 % For level 2
    Fmatrix=reshape(Fmatrix,[N_d*N_aprime,N_a]);
end

end


