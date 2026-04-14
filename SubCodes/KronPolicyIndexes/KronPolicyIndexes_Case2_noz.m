function PolicyKron=KronPolicyIndexes_Case2_noz(Policy, n_d, n_a)
% Input: Policy (l_d,n_a);
%
% Output: Policy=zeros(N_a,1); %indexes the optimal choice for d as function of a

N_a=prod(n_a);
l_d=length(n_d);

% Using Case 2, so GPU only
if l_d==1 % no need to do anything
    PolicyKron=Policy;
elseif l_d>1
    temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
    temp2=gpuArray(cumprod(n_d')); % column vector
    PolicyTemp=(reshape(Policy,[l_d,N_a])-temp*ones(1,N_a,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a,'gpuArray'));
    PolicyKron=reshape(sum(PolicyTemp,1),[N_a,1]);
end

end
