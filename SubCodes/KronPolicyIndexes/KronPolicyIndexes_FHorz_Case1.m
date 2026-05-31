function PolicyKron=KronPolicyIndexes_FHorz_Case1(Policy, n_d, n_a, n_z, N_j, simoptions)
% Input: Policy (l_d+l_a,n_a,n_z,N_j);
%
% Output: Policy=zeros(2,N_a,N_z,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%                       (1,N_a,N_z,N_j) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);
l_a=length(n_a);

%%
Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);

% Strip trailing PolicyL2flag channel (always present under gridinterplayer==1)
if simoptions.gridinterplayer==1
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1),prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:end-1,:), [tempsize(1)-1, tempsize(2:end)]);
end

if simoptions.gridinterplayer==0
    if n_d(1)==0
        if isa(Policy,'gpuArray')
            PolicyKron=zeros(1,N_a,N_z,N_j,'gpuArray');
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
            end
        else
            PolicyKron=zeros(1,N_a,N_z,N_j);
            for jj=1:N_j
                PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
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
elseif simoptions.gridinterplayer==1
    % Under GI the inner KronPolicyIndexes_Case1 returns rows:
    %   l_a==1: [a1, L2] (2 rows) nod or [d, a1, L2] (3 rows) with-d
    %   l_a>=2: [a1, a2, L2] (3 rows) nod or [d, a1, a2, L2] (4 rows) with-d
    %   (l_a>2 collapses a2..al_a into a single row; output count caps at 3/4.)
    n_rows_a=1+min(l_a,2); % 2 if l_a==1, 3 if l_a>=2
    if n_d(1)==0
        PolicyKron=zeros(n_rows_a,N_a,N_z,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
        end
    else
        PolicyKron=zeros(n_rows_a+1,N_a,N_z,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case1(Policy(:,:,:,jj), n_d, n_a, n_z, simoptions);
        end
    end
end

end