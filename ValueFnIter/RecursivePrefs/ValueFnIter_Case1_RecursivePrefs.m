function [V, PolicyIndexes]=ValueFnIter_Case1(Tolerance, V0, n_d,n_a,n_z, pi_z, beta, Fmatrix, Howards)

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

VKron=reshape(V0,[N_a,N_z]);

if n_d==0
    FmatrixKron=reshape(Fmatrix,[N_a,N_a,N_z]);
else
    FmatrixKron=reshape(Fmatrix,[N_d*N_a,N_a,N_z]);
end
[VKron, PolicyIndexesKron]=ValueFnIter_Case1_raw(Tolerance, VKron, N_d,N_a,N_z, pi_z, beta, FmatrixKron, Howards);

%Transforming Value Fn and Optimal Policy Indexes matrices back out of Kronecker Form
V=reshape(VKron,[n_a,n_z]);
PolicyIndexes=UnKronPolicyIndexes_Case1(PolicyIndexesKron, n_d, n_a, n_z);

end