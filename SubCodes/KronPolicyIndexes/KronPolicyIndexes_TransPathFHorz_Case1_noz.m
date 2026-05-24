function PolicyPathKron=KronPolicyIndexes_TransPathFHorz_Case1_noz(PolicyPath, n_d, n_a, N_j, T, simoptions)
%
% Input: Policy (l_d+l_a,n_a,N_j,T);
%
% Output: Policy=zeros(2,N_a,N_j,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a
%                       (1,N_a,N_j,T) if there is no d

N_a=prod(n_a);

PolicyPath=reshape(PolicyPath,[size(PolicyPath,1),N_a,N_j,T]);

% --- TEMPORARY (pilot): strip trailing PolicyL2flag channel if present ---
if size(PolicyPath,1) > (n_d(1)~=0)*length(n_d) + length(n_a) + 1
    tempsize=size(PolicyPath);
    PolicyPath=reshape(PolicyPath,[tempsize(1),prod(tempsize)/tempsize(1)]);
    PolicyPath=reshape(PolicyPath(1:end-1,:), [tempsize(1)-1, tempsize(2:end)]);
end

if simoptions.gridinterplayer==0
    if n_d(1)==0
        PolicyPathKron=zeros(1,N_a,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,jj,tt)=KronPolicyIndexes_Case1_noz(PolicyPath(:,:,jj,tt), n_d, n_a, simoptions);
            end
        end
    else
        PolicyPathKron=zeros(2,N_a,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,jj,tt)=KronPolicyIndexes_Case1_noz(PolicyPath(:,:,jj,tt), n_d, n_a, simoptions);
            end
        end
    end
elseif simoptions.gridinterplayer==1
    if n_d(1)==0
        PolicyPathKron=zeros(2,N_a,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,jj,tt)=KronPolicyIndexes_Case1_noz(PolicyPath(:,:,jj,tt), n_d, n_a, simoptions);
            end
        end
    else
        PolicyPathKron=zeros(3,N_a,N_j,T,'gpuArray');
        for tt=1:T
            for jj=1:N_j
                PolicyPathKron(:,:,jj,tt)=KronPolicyIndexes_Case1_noz(PolicyPath(:,:,jj,tt), n_d, n_a, simoptions);
            end
        end
    end
end

end