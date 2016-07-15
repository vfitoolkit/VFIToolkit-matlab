function [VKron,PolicyIndexesKron]=ValueFnIter_Case2_FHorz_PType_raw(N_d, N_a, N_z, N_i,N_j, pi_z, Phi_aprimeKronFn_ij, Case2_Type, beta_ij, FmatrixFn_ij)

%disp('Starting Value Fn Iteration')
VKron=zeros(N_a,N_z,N_i,N_j);
PolicyIndexesKron=zeros(N_a,N_z,N_i,N_j); %indexes the optimal choice for d given rest of dimensions a,z

for i=1:N_i
    FmatrixFn_j= @(j) FmatrixFn_ij(i,j);
    Phi_aprimeKronFn_j= @(j) Phi_aprimeKronFn_ij(i,j);
    beta_j= @(j) beta_ij(i,j);
    [VKron_i,PolicyIndexesKron_i]=ValueFnIter_Case2_FHorz_raw(N_d,N_a,N_z,N_j,pi_z,Phi_aprimeKronFn_j,Case2_Type,beta_j,FmatrixFn_j);
    VKron(:,:,i,:)=reshape(VKron_i,[N_a,N_z,1,N_j]);
    PolicyIndexesKron(:,:,i,:)=reshape(PolicyIndexesKron_i,[N_a,N_z,i,N_j]);
end

end