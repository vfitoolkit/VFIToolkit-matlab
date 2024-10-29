function Fmatrix=CreateReturnFnMatrix_Case1_Disc_fastOLG_DC1_nod_noz_Par2(ReturnFn, N_j, aprime_grid, a_grid, ReturnFnParamsAgeMatrix,Level)

l_a=1; % (or else won't get here)

nReturnFnParams=size(ReturnFnParamsAgeMatrix,2);
ParamCell=cell(nReturnFnParams,1);
for ii=1:nReturnFnParams
    ParamCell(ii,1)={shiftdim(ReturnFnParamsAgeMatrix(:,ii),-l_a-l_a)};
end

N_a=length(a_grid); % Because l_a=1
if Level==1
    N_aprime=length(aprime_grid); % Because l_a=1
elseif Level==2
    N_aprime=size(aprime_grid,1); % Because l_a=1
end

Fmatrix=arrayfun(ReturnFn, aprime_grid, shiftdim(a_grid,-1), ParamCell{:});

Fmatrix=reshape(Fmatrix,[N_aprime,N_a,N_j]);

end


