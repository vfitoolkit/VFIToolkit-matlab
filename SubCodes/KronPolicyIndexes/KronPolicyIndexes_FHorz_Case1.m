function PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z,N_j)%,options)

%Input: Policy (l_d+l_a,n_a,n_z,N_j);

%Output: Policy=zeros(2,N_a,N_z,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_j) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);
% 
% size(Policy)
% whos Policy n_d n_a n_z N_j
% size(Policy,1)
% [N_a,N_z,N_j]

Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);

%% Note: I could probably shave a few fractions of a second of runtime by actually copy-pasting code rather than calling 'KronPolicyIndexes_Case1' each time.
if n_d(1)==0
    if isa(Policy,'gpuArray')
        PolicyKron=zeros(N_a,N_z,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
        end
    else
        PolicyKron=zeros(N_a,N_z,N_j);
        for jj=1:N_j
            PolicyKron(:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
        end
    end
else
    if isa(Policy,'gpuArray')
        PolicyKron=zeros(2,N_a,N_z,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
        end
    else
        PolicyKron=zeros(2,N_a,N_z,N_j);
        for jj=1:N_j
            PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
        end
    end
end


end