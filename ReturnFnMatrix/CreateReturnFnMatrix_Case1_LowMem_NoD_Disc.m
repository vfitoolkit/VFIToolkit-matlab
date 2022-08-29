function Fmatrix=CreateReturnFnMatrix_Case1_LowMem_NoD_Disc(a_val,z_val,ReturnFn, ReturnFnParamsVec, n_a, a_gridvals, Parallel)
% Gives the return function as a matrix conditional on current values of a and z

N_a=prod(n_a);

ParamCell=cell(length(ReturnFnParamsVec),1);
for ii=1:length(ReturnFnParamsVec)
    ParamCell(ii,1)={ReturnFnParamsVec(ii)};
end

if Parallel==0 || Parallel==1 % Essentially just ignoring the Parallel=1, as can't see it being likely you would want to parallelize at this level.
    Fmatrix=zeros(N_a,1);
    for i1=1:N_a
        tempcell=num2cell([a_gridvals(i1,:),a_val,z_val]);
        Fmatrix(i1)=ReturnFn(tempcell{:},ParamCell{:});
    end
end


end


