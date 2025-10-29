function PolicyPathKron=KronPolicyIndexes_TransPathFHorz_Case1_e(PolicyPath, n_d, n_a, n_z, n_e, N_j, T, simoptions)

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_e,,N_j,T);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,N_j,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,N_j,T) if there is no d

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_z,N_e,N_j,T]);

if simoptions.gridinterplayer==0
    if n_d(1)==0
        PolicyPathKron=zeros(N_a,N_z,N_e,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e, simoptions);
            end
        end
    else
        PolicyPathKron=zeros(2,N_a,N_z,N_e,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e, simoptions);
            end
        end
    end
elseif simoptions.gridinterplayer==1
    if n_d(1)==0
        PolicyPathKron=zeros(2,N_a,N_z,N_e,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e, simoptions);
            end
        end
    else
        PolicyPathKron=zeros(3,N_a,N_z,N_e,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,:,:,jj,tt)=KronPolicyIndexes_Case1_e(PolicyPath(:,:,:,:,jj,tt), n_d, n_a, n_z, n_e, simoptions);
            end
        end
    end
end

end