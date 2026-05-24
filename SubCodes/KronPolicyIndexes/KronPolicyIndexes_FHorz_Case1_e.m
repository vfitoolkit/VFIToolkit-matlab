function PolicyKron=KronPolicyIndexes_FHorz_Case1_e(Policy, n_d, n_a, n_z, n_e, N_j, simoptions)
% Input: Policy (l_d+l_a,n_a,n_z,n_e,N_j);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%                       (1,N_a,N_z,N_e,N_j) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);
l_a=length(n_a);

% When using n_e, is instead:
% Input: Policy (l_d+l_a,n_a,n_z,n_e,N_j);
%
% Output: Policy=zeros(2,N_a,N_z,N_e,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%                       (1,N_a,N_z,N_e,N_j) if there is no d


%%
N_e=prod(n_e);
Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_e,N_j]);

% --- TEMPORARY (pilot): strip trailing PolicyL2flag channel if present ---
if size(Policy,1) > (n_d(1)~=0)*length(n_d) + length(n_a) + 1
    tempsize=size(Policy);
    Policy=reshape(Policy,[tempsize(1),prod(tempsize)/tempsize(1)]);
    Policy=reshape(Policy(1:end-1,:), [tempsize(1)-1, tempsize(2:end)]);
end

% gpu only
if simoptions.gridinterplayer==0
    if n_d(1)==0
        PolicyKron=zeros(1,N_a,N_z,N_e,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
        end
    else
        PolicyKron=zeros(2,N_a,N_z,N_e,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
        end
    end
elseif simoptions.gridinterplayer==1
    % Under GI the inner KronPolicyIndexes_Case1_e returns rows:
    %   l_a==1: [a1, L2] (2 rows) nod or [d, a1, L2] (3 rows) with-d
    %   l_a>=2: [a1, a2, L2] (3 rows) nod or [d, a1, a2, L2] (4 rows) with-d
    %   (l_a>2 collapses a2..al_a into a single row; output count caps at 3/4.)
    n_rows_a=1+min(l_a,2); % 2 if l_a==1, 3 if l_a>=2
    if n_d(1)==0
        PolicyKron=zeros(n_rows_a,N_a,N_z,N_e,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
        end
    else
        PolicyKron=zeros(n_rows_a+1,N_a,N_z,N_e,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,:,jj)=KronPolicyIndexes_Case1_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e, simoptions);
        end
    end
end

end