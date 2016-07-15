function [V, PolicyIndexes]=ValueFnIter_Case1_FHorz_PType(n_d,n_a,n_z,n_i,N_j, pi_z, beta_j, FmatrixFn_ij)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_i=prod(n_i);

[VKron, PolicyIndexesKron]=ValueFnIter_Case1_FHorz_PType_raw(N_d,N_a,N_z,N_i,N_j, pi_z, beta_j, FmatrixFn_ij);

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z,N_i,N_j]);
PolicyIndexes=UnKronPolicyIndexes_Case1_FHorz_PType(PolicyIndexesKron, n_d, n_a, n_z, N_i,N_j);

end