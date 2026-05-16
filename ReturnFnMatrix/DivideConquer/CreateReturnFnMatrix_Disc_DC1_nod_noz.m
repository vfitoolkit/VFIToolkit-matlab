function Fmatrix=CreateReturnFnMatrix_Disc_DC1_nod_noz(ReturnFn, aprime_grid, a_grid, ReturnFnParamsVec)
% For divide and conquer, with l_a=1 and no d var and no z var

ReturnFnParamsCell=num2cell(ReturnFnParamsVec)';

% l_d=0;
% l_a=1; % (or else won't get here)

% N_a=prod(n_a);

% l_d=0, l_a=1
Fmatrix=arrayfun(ReturnFn, aprime_grid, a_grid', ReturnFnParamsCell{:});
% size(Fmatrix)=[n_aprime,1]

end


