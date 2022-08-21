function PolicyPathKron=KronPolicyIndexes_TransPath_Case1(PolicyPath, n_d, n_a, n_z, T, n_e)
% n_e is an optional input
%
% Input: Policy (l_d+l_a,n_a,n_z,T);
%
% Output: Policy=zeros(2,N_a,N_z,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,T) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_e,,T);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,T) if there is no d



if ~exist('n_e','var')
    %%
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,T]);
    
    if n_d(1)==0
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(N_a,N_z,T,'gpuArray');
            for tt=1:T
                PolicyPathKron(:,:,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,tt), n_d, n_a, n_z);
            end
        else
            PolicyPathKron=zeros(N_a,N_z,T);
            for tt=1:T
                PolicyPathKron(:,:,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,tt), n_d, n_a, n_z);
            end
        end
    else
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(2,N_a,N_z,T,'gpuArray');
            for tt=1:T
                PolicyPathKron(:,:,:,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,tt), n_d, n_a, n_z);
            end
        else
            PolicyPathKron=zeros(2,N_a,N_z,T);
            for tt=1:T
                PolicyPathKron(:,:,:,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,tt), n_d, n_a, n_z);
            end
        end
    end
    
else % exist('n_e','var')
    %%
    N_e=prod(n_e);
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_e,T]);

    if n_d(1)==0
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(N_a,N_z,N_e,T,'gpuArray');
            for tt=1:T
                PolicyPathKron(:,:,:,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,tt), n_d, n_a, n_z, n_e);
            end
        else
            PolicyPathKron=zeros(N_a,N_z,N_e,T);
            for tt=1:T
                PolicyPathKron(:,:,:,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,tt), n_d, n_a, n_z, n_e);
            end
        end
    else
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(2,N_a,N_z,N_e,T,'gpuArray');
            for tt=1:T
                PolicyPathKron(:,:,:,:,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,tt), n_d, n_a, n_z, n_e);
            end
        else
            PolicyPathKron=zeros(2,N_a,N_z,N_e,T);
            for tt=1:T
                PolicyPathKron(:,:,:,:,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,tt), n_d, n_a, n_z, n_e);
            end
        end
    end
end


end