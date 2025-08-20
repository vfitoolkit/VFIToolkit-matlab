function PolicyKron=KronPolicyIndexes_FHorz_Case1_noz(Policy, n_d, n_a, N_j, simoptions)
% Input: Policy (l_d+l_a,n_a,N_j);
%
% Output: Policy=zeros(2,N_a,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a
%                       (N_a,N_j) if there is no d

N_a=prod(n_a);

%%
Policy=reshape(Policy,[size(Policy,1),N_a,N_j]);

if n_d(1)==0
    if isa(Policy,'gpuArray')
        PolicyKron=zeros(N_a,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,jj)=KronPolicyIndexes_Case1_noz(Policy(:,:,jj), n_d, n_a, simoptions);
        end
    else
        PolicyKron=zeros(N_a,N_j);
        for jj=1:N_j
            PolicyKron(:,jj)=KronPolicyIndexes_Case1_noz(Policy(:,:,jj), n_d, n_a, simoptions);
        end
    end
else
    if isa(Policy,'gpuArray')
        PolicyKron=zeros(2,N_a,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,jj)=KronPolicyIndexes_Case1_noz(Policy(:,:,jj), n_d, n_a, simoptions);
        end
    else
        PolicyKron=zeros(2,N_a,N_j);
        for jj=1:N_j
            PolicyKron(:,:,jj)=KronPolicyIndexes_Case1_noz(Policy(:,:,jj), n_d, n_a, simoptions);
        end
    end
end
    

end