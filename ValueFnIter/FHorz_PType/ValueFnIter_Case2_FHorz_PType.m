function [V, PolicyIndexes]=ValueFnIter_Case2_FHorz_PType(n_d, n_a, n_z,n_i, N_j, pi_z, Phi_aprimeKronFn_ij, Case2_Type, beta_ij, FmatrixFn_ij)

N_z=prod(n_z);
N_a=prod(n_a);
N_d=prod(n_d);
N_i=prod(n_i);

disp('Starting Value Fn Iteration')
[VKron, PolicyIndexesKron]=ValueFnIter_Case2_FHorz_PType_raw(n_d, n_a, N_z, N_i,N_j, pi_z, Phi_aprimeKronFn_ij, Case2_Type, beta_ij, FmatrixFn_ij);

%Transform V & PolicyIndexes out of kroneckered form
V=reshape(VKron,[n_a,n_z,N_j]);
PolicyIndexes=UnKronPolicyIndexes_Case2_FHorz_PType(PolicyIndexesKron, n_d, n_a, n_z,n_i,N_j);

end