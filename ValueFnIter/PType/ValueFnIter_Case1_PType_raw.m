function [VKron,PolicyIndexesKron]=ValueFnIter_Case1_PType_raw(Tolerance, V0Kron, N_d,N_a,N_z,N_i, pi_z, beta, FmatrixFn_i, Howards)
%Does exactly the same as ValueFnIter_Case1, but does not reshape input and
%output (so these must already be in kron form) and only returns the policy
%function (no value fn). It also does not bother to check sizes.

VKron=zeros(N_a,N_z,N_i);

if N_d==0
    PolicyIndexesKron=zeros(1,N_a,N_z,N_i); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
    for i=1:N_i
        FmatrixKron=reshape(FmatrixFn_i(i),[N_a,N_a,N_z]);
        [VKron_i,PolicyIndexesKron_i]=ValueFnIter_Case1_no_d_raw(Tolerance, V0Kron(:,:,i), N_a, N_z, pi_z, beta, FmatrixKron, Howards);
        VKron(:,:,i)=VKron_i;
        PolicyIndexesKron(:,:,:,i)=PolicyIndexesKron_i;
    end
    return
end

PolicyIndexesKron=zeros(2,N_a,N_z,N_i); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z

for i=1:N_i
    FmatrixKron=reshape(FmatrixFn_i(i),[N_d*N_a,N_a,N_z]);
    [VKron_i,PolicyIndexesKron_i]=ValueFnIter_Case1_raw(Tolerance, V0Kron(:,:,i), N_d, N_a, N_z, pi_z, beta, FmatrixKron, Howards);
    VKron(:,:,i)=VKron_i;
    PolicyIndexesKron(:,:,:,i)=PolicyIndexesKron_i;
end

end