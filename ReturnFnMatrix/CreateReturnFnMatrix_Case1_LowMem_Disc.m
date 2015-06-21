function Fmatrix=CreateReturnFnMatrix_Case1_LowMem_Disc(ReturnFn,a,z, n_d, n_a, d_gridvals, a_gridvals, Parallel)
%If there is no d variable, just input n_d=0 and d_grid=0

N_d=prod(n_d);
N_a=prod(n_a);

if Parallel==0 || Parallel==1
    Fmatrix=zeros(N_d*N_a,1);
    for i1=1:N_d
        for i2=1:N_a
            i1i2=i1+(i2-1)*N_d;
            Fmatrix(i1i2)=ReturnFn(d_gridvals(i1,:),a_gridvals(i2,:),a,z);
        end
    end
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
    ReturnFn_Par=@(i1,i2) ReturnFn(ind2grid_homemade(n_d,i1,d_grid),ind2grid_homemade(n_a,i2,a_grid), a,z);
    Fmatrix=arrayfun(ReturnFn_Par, d_dim, aprime_dim);
end


end


