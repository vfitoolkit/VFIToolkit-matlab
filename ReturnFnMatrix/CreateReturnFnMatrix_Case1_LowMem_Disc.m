function Fmatrix=CreateReturnFnMatrix_Case1_LowMem_Disc(ReturnFn, ReturnFnParamsVec,a_val,z_val, n_d, n_a, d_gridvals, a_gridvals, Parallel)
%If there is no d variable, just input n_d=0 and d_grid=0

N_d=prod(n_d);
N_a=prod(n_a);

% Essentially just ignoring the Parallel=1, as can't see it being likely you would want to parallelize at this level.
Fmatrix=zeros(N_d*N_a,1);
for i1i2=1:N_d*N_a
    sub=ind2sub_homemade([N_d,N_a],i1i2);
    temp=[d_gridvals(sub(1),:),a_gridvals(sub(2),:),a_val,z_val, ReturnFnParamsVec];
    TempCell=cell(length(temp),1);
    for ii=1:length(temp)
        TempCell(ii,1)={temp(ii)};
    end
    Fmatrix(i1i2)=ReturnFn(TempCell{:});
end

end


