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
%         temp=[a_gridvals(i1,:),a_val,z_val, ReturnFnParamsVec];
%         TempCell=cell(length(temp),1);
%         for ii=1:length(temp)
%             TempCell(ii,1)={temp(ii)};
%         end
%         Fmatrix(i1)=ReturnFn(TempCell{:});
    tempcell=num2cell([a_gridvals(i1,:),a_val,z_val]);
    Fmatrix(i1)=ReturnFn(tempcell{:},ParamCell{:});
    end
    
% elseif Parallel==1
%     Fmatrix=zeros(N_a,1);
%     parfor i1=1:N_a
%         Fmatrix(i1)=ReturnFn(a_gridvals(i1,:),a,z, ReturnFnParamsVec);
%     end

elseif Parallel==2
    disp('WARNING: CreateReturnFnMatrix_Case1_Disc does not really suppport Parallel=2 yet')
    dbstack
    aprime_dim=gpuArray.ones(N_a,1,1); aprime_dim(:,1,1)=1:1:N_a;
    
    ReturnFn_Par=@(i1) ReturnFn(a_gridvals(i1), a_val, z_val);
    Fmatrix=arrayfun(ReturnFn_Par, aprime_dim);

end


end


