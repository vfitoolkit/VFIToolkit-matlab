function [V, PolicyIndexes]=ValueFnIter_Case1_PType(Tolerance, n_d,n_a,n_z,n_i, pi_z, beta, FmatrixFn_i, Howards)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_i=prod(n_i);

if isfield(vfoptions,'V0')
    V0Kron=reshape(vfoptions.V0,[N_a,N_z,N_i]);
else
    if vfoptions.parallel==2
        V0Kron=zeros([N_a,N_z,N_i], 'gpuArray');
    else
        V0Kron=zeros([N_a,N_z,N_i]);        
    end
end

[VKron, PolicyIndexesKron]=ValueFnIter_Case1_raw(Tolerance, V0Kron, N_d,N_a,N_z,N_i,pi_z, beta, FmatrixFn_i, Howards);

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z,n_i]);
PolicyIndexes=UnKronPolicyIndexes_Case1_PType(PolicyIndexesKron, n_d, n_a, n_z,n_i);

end