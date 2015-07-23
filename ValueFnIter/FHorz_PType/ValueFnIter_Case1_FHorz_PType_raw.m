function [VKron,PolicyIndexesKron]=ValueFnIter_Case1_FHorz_PType_raw(N_d,N_a,N_z,N_i,N_j, pi_z, beta_j, FmatrixFn_ij)

VKron=zeros(N_a,N_z,N_i,N_j);

if N_d==0
    PolicyIndexesKron=zeros(1,N_a,N_a,N_i,N_j);
    for i=1:N_i
        FmatrixFn_j=@(j) FmatrixFn_ij(i,j);
        [VKron_i,PolicyIndexesKron_i]=ValueFnIter_Case1_FHorz_no_d_raw(N_a, N_z, N_j, pi_z, beta_j, FmatrixFn_j);
        VKron(:,:,i,:)=reshape(VKron_i,[N_a,N_z,1,N_j]);
        PolicyIndexesKron(1,:,:,1,:)=reshape(PolicyIndexesKron_i,[1,N_a,N_z,1,N_j]);
    end
    return
end

PolicyIndexesKron=zeros(2,N_a,N_z,N_i,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
    for i=1:N_i
        FmatrixFn_j=@(j) FmatrixFn_ij(i,j);
        [VKron_i,PolicyIndexesKron_i]=ValueFnIter_Case1_FHorz_raw(N_d,N_a, N_z, N_j, pi_z, beta_j, FmatrixFn_j);
        VKron(:,:,i,:)=reshape(VKron_i,[N_a,N_z,1,N_j]);
        PolicyIndexesKron(:,:,:,1,:)=reshape(PolicyIndexesKron_i,[2,N_a,N_z,1,N_j]);
    end

end