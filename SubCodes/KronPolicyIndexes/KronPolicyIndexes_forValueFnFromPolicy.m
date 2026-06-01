function PolicyKron=KronPolicyIndexes_forValueFnFromPolicy(Policy, n_d, n_a, n_z, N_j, vfoptions)

% Input: Policy (l_d+l_a, n_a, n_z[, N_j])
%
% Output: PolicyKron=zeros(2, N_a, N_z[, N_j]); % first dim indexes optimal d and aprime
%                       (1, N_a, N_z[, N_j]) if there is no d
% Note: vfoptions.gridinterplayer=1 adds an L2 (second-layer) row to the output. Input is
%       expected to carry a trailing L2flag row as well; that row is dropped here (callers
%       of this function don't use L2flag). For GI2A (l_a==2, no d) the a-indices are NOT
%       Kron'd into a single index — they are returned separately as rows (a1, a2, L2).
% Note: n_semiz and n_e can be passed as part of n_z, e.g. pass [n_semiz,n_z,n_e] as the n_z input
% N_j==0: infinite horizon
% N_j>0:  finite horizon — N_j is folded into the trailing z-dim, body runs once,
%         then PolicyKron is reshaped back to (..., N_z, N_j) at the end.

if N_j==0
    N_a=prod(n_a);
    N_z=prod(n_z);

    l_a=length(n_a);

    if vfoptions.gridinterplayer==0
        % Reshape Policy
        Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);

        if n_d(1)==0
            if l_a==1
                PolicyKron=reshape(Policy,[1,N_a,N_z]);
            else %l_a>1
                temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
                temp2=gpuArray(cumprod(n_a')); % column vector
                PolicyTemp=(reshape(Policy,[l_a,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
                PolicyKron=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
            end
        else
            l_d=length(n_d);

            PolicyKron=zeros(2,N_a,N_z,'gpuArray');

            if l_d==1
                PolicyKron(1,:,:)=Policy(1,:,:);
            else
                temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
                temp2=gpuArray(cumprod(n_d')); % column vector
                PolicyTemp=(reshape(Policy(1:l_d,:,:),[l_d,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
                PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
            end
            % Then, a
            if l_a==1
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
            else
                temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
                temp2=gpuArray(cumprod(n_a')); % column vector
                PolicyTemp=(reshape(Policy(l_d+1:l_d+l_a,:,:),[l_a,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
                PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
            end
        end

    elseif vfoptions.gridinterplayer==1
        % Input Policy has a trailing L2flag row; output PolicyKron drops it (callers don't use L2flag).
        % Reshape Policy
        Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);

        if n_d(1)==0
            if l_a==1 || l_a==2
                PolicyKron=Policy(1:end-1,:,:); % a1, possibly a2, L2 (drops trailing L2flag)
            else %l_a>2
                PolicyKron=zeros(3,N_a,N_z,'gpuArray');
                PolicyKron(1,:,:)=Policy(1,:,:); % a1
                temp=[0; ones(l_a-2,1,'gpuArray')];
                temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
                PolicyTemp=(reshape(Policy(2:l_a,:,:),[l_a-1,N_a*N_z])-temp).*[1;temp2(1:end-1)];
                PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
                PolicyKron(3,:,:)=Policy(l_a+1,:,:); % L2 index
            end
        else
            l_d=length(n_d);
            if l_a==1
                PolicyKron=zeros(3,N_a,N_z,'gpuArray');
            else
                PolicyKron=zeros(4,N_a,N_z,'gpuArray');
            end

            if l_d==1
                PolicyKron(1,:,:)=Policy(1,:,:);
            else
                temp=[0; ones(l_d-1,1,'gpuArray')];
                temp2=gpuArray(cumprod(n_d')); % column vector
                PolicyTemp=(reshape(Policy(1:l_d,:,:),[l_d,N_a*N_z])-temp).*[1;temp2(1:end-1)];
                PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
            end
            % Then, a
            if l_a==1
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
                PolicyKron(3,:,:)=Policy(l_a+l_d+1,:,:); % L2 index
            elseif l_a==2
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
                PolicyKron(3,:,:)=Policy(l_d+2,:,:);
                PolicyKron(4,:,:)=Policy(l_a+l_d+1,:,:); % L2 index
            else
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
                temp=[0; ones(l_a-2,1,'gpuArray')];
                temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
                PolicyTemp=(reshape(Policy(l_d+2:l_d+l_a,:,:),[l_a-1,N_a*N_z])-temp).*[1;temp2(1:end-1)];
                PolicyKron(3,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
                PolicyKron(4,:,:)=Policy(l_a+l_d+1,:,:); % L2 index
            end

        end

    end

else % N_j>0: fold N_j into the trailing z-dim, run body, split back
    N_a=prod(n_a);
    N_z=prod(n_z)*N_j;

    l_a=length(n_a);

    if vfoptions.gridinterplayer==0
        % Reshape Policy
        Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);

        if n_d(1)==0
            if l_a==1
                PolicyKron=reshape(Policy,[1,N_a,N_z]);
            else %l_a>1
                temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
                temp2=gpuArray(cumprod(n_a')); % column vector
                PolicyTemp=(reshape(Policy,[l_a,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
                PolicyKron=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
            end
        else
            l_d=length(n_d);

            PolicyKron=zeros(2,N_a,N_z,'gpuArray');

            if l_d==1
                PolicyKron(1,:,:)=Policy(1,:,:);
            else
                temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
                temp2=gpuArray(cumprod(n_d')); % column vector
                PolicyTemp=(reshape(Policy(1:l_d,:,:),[l_d,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
                PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
            end
            % Then, a
            if l_a==1
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
            else
                temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
                temp2=gpuArray(cumprod(n_a')); % column vector
                PolicyTemp=(reshape(Policy(l_d+1:l_d+l_a,:,:),[l_a,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
                PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
            end
        end

    elseif vfoptions.gridinterplayer==1
        % Input Policy has a trailing L2flag row; output PolicyKron drops it (callers don't use L2flag).
        % Reshape Policy
        Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);

        if n_d(1)==0
            if l_a==1 || l_a==2
                PolicyKron=Policy(1:end-1,:,:); % a1, possibly a2, L2 (drops trailing L2flag)
            else %l_a>2
                PolicyKron=zeros(3,N_a,N_z,'gpuArray');
                PolicyKron(1,:,:)=Policy(1,:,:); % a1
                temp=[0; ones(l_a-2,1,'gpuArray')];
                temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
                PolicyTemp=(reshape(Policy(2:l_a,:,:),[l_a-1,N_a*N_z])-temp).*[1;temp2(1:end-1)];
                PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
                PolicyKron(3,:,:)=Policy(l_a+1,:,:); % L2 index
            end
        else
            l_d=length(n_d);
            if l_a==1
                PolicyKron=zeros(3,N_a,N_z,'gpuArray');
            else
                PolicyKron=zeros(4,N_a,N_z,'gpuArray');
            end

            if l_d==1
                PolicyKron(1,:,:)=Policy(1,:,:);
            else
                temp=[0; ones(l_d-1,1,'gpuArray')];
                temp2=gpuArray(cumprod(n_d')); % column vector
                PolicyTemp=(reshape(Policy(1:l_d,:,:),[l_d,N_a*N_z])-temp).*[1;temp2(1:end-1)];
                PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
            end
            % Then, a
            if l_a==1
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
                PolicyKron(3,:,:)=Policy(l_a+l_d+1,:,:); % L2 index
            elseif l_a==2
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
                PolicyKron(3,:,:)=Policy(l_d+2,:,:);
                PolicyKron(4,:,:)=Policy(l_a+l_d+1,:,:); % L2 index
            else
                PolicyKron(2,:,:)=Policy(l_d+1,:,:);
                temp=[0; ones(l_a-2,1,'gpuArray')];
                temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
                PolicyTemp=(reshape(Policy(l_d+2:l_d+l_a,:,:),[l_a-1,N_a*N_z])-temp).*[1;temp2(1:end-1)];
                PolicyKron(3,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
                PolicyKron(4,:,:)=Policy(l_a+l_d+1,:,:); % L2 index
            end

        end

    end

    % Split trailing dim back into (N_z, N_j)
    PolicyKron=reshape(PolicyKron,[size(PolicyKron,1),N_a,prod(n_z),N_j]);
end

end
