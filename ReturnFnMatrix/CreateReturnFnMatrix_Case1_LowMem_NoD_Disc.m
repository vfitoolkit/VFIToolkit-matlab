function Fmatrix=CreateReturnFnMatrix_Case1_LowMem_NoD_Disc(a,z,ReturnFn, n_a, a_gridvals, Parallel)
%If there is no d variable, just input n_d=0 and d_grid=0

N_a=prod(n_a);

if Parallel==0
    Fmatrix=zeros(N_a,1);
    for i1=1:N_a
        Fmatrix(i1)=ReturnFn(a_gridvals(i1,:),a,z);
    end
    
elseif Parallel==1
    Fmatrix=zeros(N_a,1);
    parfor i1=1:N_a
        Fmatrix(i1)=ReturnFn(a_gridvals(i1,:),a,z);
    end

elseif Parallel==2
    disp('WARNING: CreateReturnFnMatrix_Case1_Disc does not really suppport Parallel=2 yet')
    aprime_dim=gpuArray.ones(N_a,1,1); aprime_dim(:,1,1)=1:1:N_a;
    
    ReturnFn_Par=@(i1) ReturnFn(a_gridvals(i1), a_, z);
    Fmatrix=arrayfun(ReturnFn_Par, aprime_dim);

end


end


