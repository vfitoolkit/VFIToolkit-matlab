function [V, PolicyIndexes]=ValueFnIter_Case2_FHorz(n_d, n_a, n_z, N_j, pi_z, Phi_aprimeKronFn_j, Case2_Type, beta_j, FmatrixFn_j)

N_z=prod(n_z);
N_a=prod(n_a);
N_d=prod(n_d);

disp('Starting Value Fn Iteration')
[VKron, PolicyIndexesKron]=ValueFnIter_Case2_FHorz_raw(N_d, N_a, N_z, N_j, pi_z, Phi_aprimeKronFn_j, Case2_Type, beta_j, FmatrixFn_j);

%Transform V & PolicyIndexes out of kroneckered form
V=reshape(VKron,[n_a,n_z,N_j]);
PolicyIndexes=UnKronPolicyIndexes_Case2_FHorz(PolicyIndexesKron, n_d, n_a, n_z,N_j);

end