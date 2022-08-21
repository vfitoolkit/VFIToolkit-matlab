function PolicyPathKron=KronPolicyIndexes_TransPathFHorz_Case1(PolicyPath, n_d, n_a, n_z, N_j, T, n_e)
% n_e is an optional input
%
% Input: Policy (l_d+l_a,n_a,n_z,N_j,T);
%
% Output: Policy=zeros(2,N_a,N_z,N_j,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_j,T) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_e,,N_j,T);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,N_j,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,N_j,T) if there is no d



if ~exist('n_e','var')
    %%
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_j,T]);
    
    if n_d(1)==0
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(N_a,N_z,N_j,T,'gpuArray');
            for tt=1:T
                for jj=1:N_j
                    PolicyPathKron(:,:,jj,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,jj,tt), n_d, n_a, n_z);%,options);
                end
            end
        else
            PolicyPathKron=zeros(N_a,N_z,N_j,T);
            for tt=1:T
                for jj=1:N_j
                PolicyPathKron(:,:,jj,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,jj,tt), n_d, n_a, n_z);%,options);
                end
            end
        end
    else
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(2,N_a,N_z,N_j,T,'gpuArray');
            for tt=1:T
                for jj=1:N_j
                PolicyPathKron(:,:,:,jj,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,jj,tt), n_d, n_a, n_z);%,options);
                end
            end
        else
            PolicyPathKron=zeros(2,N_a,N_z,N_j,T);
            for tt=1:T
                for jj=1:N_j
                PolicyPathKron(:,:,:,jj,tt)=KronPolicyIndexes_Case1(PolicyPath(:,:,:,jj,tt), n_d, n_a, n_z);%,options);
                end
            end
        end
    end
    
else % exist('n_e','var')
    %%
    N_e=prod(n_e);
    PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_e,N_j,T]);

    if n_d(1)==0
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(N_a,N_z,N_e,N_j,T,'gpuArray');
            for tt=1:T
                for jj=1:N_j
                    PolicyPathKron(:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e);%,options);
                end
            end
        else
            PolicyPathKron=zeros(N_a,N_z,N_e,N_j,T);
            for tt=1:T
                for jj=1:N_j
                    PolicyPathKron(:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e);%,options);
                end
            end
        end
    else
        if isa(PolicyPath,'gpuArray')
            PolicyPathKron=zeros(2,N_a,N_z,N_e,N_j,T,'gpuArray');
            for tt=1:T
                for jj=1:N_j
                    PolicyPathKron(:,:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e);%,options);
                end
            end
        else
            PolicyPathKron=zeros(2,N_a,N_z,N_e,N_j,T);
            for tt=1:T
                for jj=1:N_j
                    PolicyPathKron(:,:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e);%,options);
                end
            end
        end
    end
end


end