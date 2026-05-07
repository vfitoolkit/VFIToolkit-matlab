function PolicyKron=KronPolicyIndexes_Case2(Policy, n_d, n_a, n_z)
% Input: Policy (l_d,n_a,n_z);
%
% Output: Policy=zeros(N_a,N_z); %indexes the optimal choice for d as function of a,z

% N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
l_d=length(n_d);

% Using Case 2, so GPU only
if l_d==1 % no need to do anything
   PolicyKron=Policy;
elseif l_d>1
   temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
   temp2=gpuArray(cumprod(n_d')); % column vector
   PolicyTemp=(reshape(Policy,[l_d,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
   PolicyKron=reshape(sum(PolicyTemp,1),[N_a,N_z]);
end

end
