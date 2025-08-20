function PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j, n_e)
% n_e is an optional input
%
% Input: Policy (l_d+l_a,n_a,n_z,N_j);
%
% Output: Policy=zeros(2,N_a,N_z,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_j) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_e,,N_j);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,N_j) if there is no d



if ~exist('n_e','var')
    %%
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    
    if n_d(1)==0
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(N_a,N_z,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
            end
        else
            PolicyKron=zeros(N_a,N_z,N_j);
            for jj=1:N_j
                PolicyKron(:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
            end
        end
    else
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(2,N_a,N_z,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
            end
        else
            PolicyKron=zeros(2,N_a,N_z,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
            end
        end
    end
    
else % exist('n_e','var')
    %%
    N_e=prod(n_e);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_e,N_j]);

    if n_d(1)==0
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(N_a,N_z,N_e,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
            end
        else
            PolicyKron=zeros(N_a,N_z,N_e,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
            end
        end
    else
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(2,N_a,N_z,N_e,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
            end
        else
            PolicyKron=zeros(2,N_a,N_z,N_e,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
            end
        end
    end
end


end