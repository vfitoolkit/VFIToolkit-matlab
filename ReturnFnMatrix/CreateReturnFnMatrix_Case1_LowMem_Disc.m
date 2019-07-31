function Fmatrix=CreateReturnFnMatrix_Case1_LowMem_Disc(ReturnFn, ReturnFnParamsVec,a_val,z_val, n_d, n_a, d_gridvals, a_gridvals, Parallel)
%If there is no d variable, just input n_d=0 and d_grid=0

N_d=prod(n_d);
N_a=prod(n_a);

if Parallel==0 || Parallel==1 % Essentially just ignoring the Parallel=1, as can't see it being likely you would want to parallelize at this level.
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
%     for i1=1:N_d
%         for i2=1:N_a
%             i1i2=i1+(i2-1)*N_d;
%             Fmatrix(i1i2)=ReturnFn(d_gridvals(i1,:),a_gridvals(i2,:),a,z);
%         end
%     end
% elseif Parallel==1
%             
%     Fmatrix=zeros(N_d*N_a,1);
%     for i1=1:N_d
%         for i2=1:N_a
%             Fmatrix(i1+(i2-1)*N_d,1)=ReturnFn(d_gridvals(i1,:),a_gridvals(i2,:),a,z);
%         end
%     end
elseif Parallel==2
    disp('WARNING: CreateReturnFnMatrix_Case1_Disc does not really suppport Parallel=2 yet')
    d_dim=gpuArray.ones(N_d,1); d_dim(:,1,1,1)=1:1:N_d;
    aprime_dim=gpuArray.ones(1,N_a); aprime_dim(1,:,1,1)=1:1:N_a;
    ReturnFn_Par=@(i1,i2) ReturnFn(ind2grid_homemade(n_d,i1,d_grid),ind2grid_homemade(n_a,i2,a_grid), a_val,z_val);
    Fmatrix=arrayfun(ReturnFn_Par, d_dim, aprime_dim);
end


end


