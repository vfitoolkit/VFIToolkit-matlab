function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, aprime_grid, a_grid, ReturnFnParams)
% For divide and conquer, with l_a=1 and no d var and no z var

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    ParamCell(ii)={ReturnFnParams(ii)};
end

% l_d=0;
l_a=1; % (or else won't get here)

% N_a=prod(n_a);

% l_d=0, l_a=1
Fmatrix=arrayfun(ReturnFn, aprime_grid, a_grid', ParamCell{:});
% size(Fmatrix)=[n_aprime,1]

end


