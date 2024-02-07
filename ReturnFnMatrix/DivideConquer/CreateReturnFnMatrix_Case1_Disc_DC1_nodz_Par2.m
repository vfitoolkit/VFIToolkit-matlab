function Fmatrix=CreateReturnFnMatrix_Case1_Disc_DC1_nodz_Par2(ReturnFn, n_a, a_grid, aprime_grid, ReturnFnParams)
% For divide and conquer, with l_a=1 and no d var and no z var

ParamCell=cell(length(ReturnFnParams),1);
for ii=1:length(ReturnFnParams)
    if ~all(size(ReturnFnParams(ii))==[1,1])
        fprintf('ERROR: Using GPU for the return fn does not allow for any of ReturnFnParams to be anything but a scalar, problem with %i-th parameter',ii)
    end
    ParamCell(ii)={ReturnFnParams(ii)};
end

% l_d=0;
l_a=1; % (or else won't get here)

% N_a=prod(n_a);

if nargin(ReturnFn)~=l_a+l_a+length(ReturnFnParams)
    fprintf('Next line is numbers relevant to the error \n')
    [nargin(ReturnFn),l_a,length(ReturnFnParams)]
    error('ERROR: Number of inputs to ReturnFn does not fit with size of ReturnFnParams')
end

% l_d=0, l_a=1
Fmatrix=arrayfun(ReturnFn, aprime_grid, a_grid', ParamCell{:});
% size(Fmatrix)=[n_aprime,1]

end


