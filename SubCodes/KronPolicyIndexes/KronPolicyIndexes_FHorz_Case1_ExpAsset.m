function PolicyKron=KronPolicyIndexes_FHorz_Case1_ExpAsset(Policy, n_d, n_a, n_z, N_j, n_e)
error('NOT USED FOR ANYTHING')
% n_e is an optional input
%
% Input: Policy (l_d+l_a-1,n_a,n_z,N_j);
%
% Output: 
%    if l_d==1: Policy=zeros(2,N_a,N_z,N_j);
%    if l_d>1:  Policy=zeros(3,N_a,N_z,N_j);

N_a=prod(n_a);
N_z=prod(n_z);

% When using n_e, is instead:
% Input: Policy (l_d+l_a-1,n_a,n_z,n_e,,N_j);
%
% Output:
%    if l_d==1: Policy=zeros(2,N_a,N_z,N_e,N_j);
%    if l_d>1:  Policy=zeros(3,N_a,N_z,N_e,N_j);

l_d=length(n_d);

if ~exist('n_e','var')
    %%
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    
    if l_d==1
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(2,N_a,N_z,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
            end
        else
            PolicyKron=zeros(2,N_a,N_z,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
            end
        end
    else % l_d>1, so there are d1 variables
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(3,N_a,N_z,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
            end
        else
            PolicyKron=zeros(3,N_a,N_z,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset(Policy(:,:,:,jj), n_d, n_a, n_z);%,options);
            end
        end
        
    end
    
else % exist('n_e','var')
    %%
    N_e=prod(n_e);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_e,N_j]);

    if l_d==1
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(2,N_a,N_z,N_e,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e);%,options);
            end
        else
            PolicyKron=zeros(2,N_a,N_z,N_e,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e);%,options);
            end
        end
    else % l_d>1
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(3,N_a,N_z,N_e,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e);%,options);
            end
        else
            PolicyKron=zeros(3,N_a,N_z,N_e,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_ExpAsset_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e);%,options);
            end
        end        
    end
end


end