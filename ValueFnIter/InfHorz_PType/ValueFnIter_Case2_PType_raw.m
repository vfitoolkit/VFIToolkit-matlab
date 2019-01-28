function [VKron,PolicyIndexesKron]=ValueFnIter_Case2_PType_raw(Tolerance, V0Kron, N_d, N_a, N_z,N_i, pi_z, Phi_aprimeKron, Case2_Type, beta, FmatrixFn_i, Howards)

VKron=zeros(N_a,N_z,N_i);
PolicyIndexesKron=zeros(N_a,N_z,N_i); %indexes the optimal choice for d given rest of dimensions a,z
for i=1:N_i
    FmatrixKron_i=reshape(FmatrixFn_i(i),[N_d,N_a,N_z]);
    [VKron_i,PolicyIndexesKron_i]=ValueFnIter_Case2_raw(Tolerance, V0Kron(:,:,i),N_d,N_a,N_z,pi_z,Phi_aprimeKron, Case2_Type, beta, FmatrixKron_i, Howards);
    VKron(:,:,i)=VKron_i;
    PolicyIndexesKron(:,:,i)=PolicyIndexesKron_i;
end

end