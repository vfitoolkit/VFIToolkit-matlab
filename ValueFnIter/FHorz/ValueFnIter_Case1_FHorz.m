function [V, PolicyIndexes]=ValueFnIter_Case1_FHorz(n_d,n_a,n_z,N_j, pi_z, beta_j, FmatrixFn_j)

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

% %Check the sizes of some of the inputs
% if length(n_z)==1 && n_z(1)==1
%     if size(pi_z)~=[N_z, N_z]
%         disp('Error: pi is not of size N_z-by-N_z')
%     elseif size(Fmatrix)~=[n_d, n_a, n_a]
%         disp('Error: Fmatrix is not of size [n_d, n_a, n_a, n_z]')
%     elseif size(V0)~=[n_a]
%         disp('Error: Starting choice for ValueFn is not of size [n_a,n_z]')
%     end
% else
%     if size(pi_z)~=[N_z, N_z]
%         disp('Error: pi is not of size N_z-by-N_z')
%     elseif size(Fmatrix)~=[n_d, n_a, n_a, n_z]
%         disp('Error: Fmatrix is not of size [n_d, n_a, n_a, n_z]')
%     elseif size(V0)~=[n_a,n_z]
%         disp('Error: Starting choice for ValueFn is not of size [n_a,n_z]')
%     end
% end

[VKron, PolicyIndexesKron]=ValueFnIter_Case1_FHorz_raw(N_d,N_a,N_z, N_j, pi_z, beta_j, FmatrixFn_j);

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z,N_j]);
PolicyIndexes=UnKronPolicyIndexes_Case1_FHorz(PolicyIndexesKron, n_d, n_a, n_z, N_j);

end