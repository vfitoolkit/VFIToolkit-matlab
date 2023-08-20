function PolicyKron=KronPolicyIndexes_FHorz_Case1_semiz(Policy, n_d1, n_d2, n_a, n_z, n_semiz, N_j, n_e)
% n_e is an optional input
%
% Input: Policy (l_d+l_a,n_a,n_z,n_semiz,N_j);
%
% Output: Policy=zeros(3,N_a,N_z,N_semiz,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                     (2,N_a,N_z,N_semiz,N_j) if there is no d1

N_d1=prod(n_d1);
N_a=prod(n_a);
N_z=prod(n_z);
N_semiz=prod(n_semiz);

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_semiz,n_e,,N_j);
%
% Output: Policy=zeros(3,N_a,N_z,N_semiz,N_e,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                     (2,N_a,N_z,N_semiz,N_e,N_j) if there is no d1

if N_z==0
    if exist('n_e','var')
        PolicyKron=KronPolicyIndexes_FHorz_Case1_noz_semiz(Policy, n_d1, n_d2, n_a, n_semiz, N_j, n_e); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
    else
        PolicyKron=KronPolicyIndexes_FHorz_Case1_noz_semiz(Policy, n_d1, n_d2, n_a, n_semiz, N_j); % Create it here as want it both here and inside SimPanelIndexes_FHorz_Case1 (which will recognise that it is already in this form)
    end
    return
end

if ~exist('n_e','var')
    %%
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_semiz,N_j]);
    
    if N_d1==0
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(2,N_a,N_z,N_semiz,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_semiz(Policy(:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz);%,options);
            end
        else
            PolicyKron=zeros(2,N_a,N_z,N_semiz,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1_semiz(Policy(:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz);%,options);
            end
        end
    else
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(3,N_a,N_z,N_semiz,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_semiz(Policy(:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz);%,options);
            end
        else
            PolicyKron=zeros(3,N_a,N_z,N_semiz,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_semiz(Policy(:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz);%,options);
            end
        end
    end
    
else % exist('n_e','var')
    %%
    N_e=prod(n_e);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_semiz,N_e,N_j]);

    if N_d1==0
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(2,N_a,N_z,N_semiz,N_e,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,:,:,jj)=KronPolicyIndexes_Case1_semiz_e(Policy(:,:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz, n_e);%,options);
            end
        else
            PolicyKron=zeros(2,N_a,N_z,N_semiz,N_e,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,:,:,jj)=KronPolicyIndexes_Case1_semiz_e(Policy(:,:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz, n_e);%,options);
            end
        end
    else
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(3,N_a,N_z,N_semiz,N_e,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,:,:,:,jj)=KronPolicyIndexes_Case1_semiz_e(Policy(:,:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz, n_e);%,options);
            end
        else
            PolicyKron=zeros(3,N_a,N_z,N_semiz,N_e,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,:,:,:,jj)=KronPolicyIndexes_Case1_semiz_e(Policy(:,:,:,:,:,jj), n_d1, n_d2, n_a, n_z, n_semiz, n_e);%,options);
            end
        end
    end
end


end