function PolicyKron=KronPolicyIndexes_Case1_noz(Policy, n_d, n_a, simoptions)

%Input: Policy (l_d+l_a,n_a);

%Output: Policy=zeros(2,N_a); %first dim indexes the optimal choice for d and aprime rest of dimensions a
%                       (N_a) if there is no d

N_a=prod(n_a);

l_a=length(n_a);

Policy=reshape(Policy,[size(Policy,1),N_a]);


if simoptions.gridinterplayer==0
    % Reshape Policy
    % Policy=reshape(Policy,[size(Policy,1),N_a]);

    if n_d(1)==0
        if l_a==1
            PolicyKron=reshape(Policy,[N_a,1]);
        else %l_a>1
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy,[l_a,N_a])-temp*ones(1,N_a,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a,'gpuArray'));
            PolicyKron=reshape(sum(PolicyTemp,1),[N_a,1]);
        end

    else
        l_d=length(n_d);

        % Should test whether code runs faster with this predeclaration of PolicyKron commented or uncommented
        % PolicyKron=zeros(2,N_a,'gpuArray');

        if l_d==1
            PolicyKron(1,:)=Policy(1,:);
        else
            temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
            temp2=gpuArray(cumprod(n_d')); % column vector
            PolicyTemp=(reshape(Policy(1:l_d,:),[l_d,N_a])-temp*ones(1,N_a,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a,'gpuArray'));
            PolicyKron(1,:)=reshape(sum(PolicyTemp,1),[1,N_a]);
        end
        % Then, a
        if l_a==1
            PolicyKron(2,:)=Policy(l_d+1,:);
        else
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy(l_d+1:l_d+l_a,:),[l_a,N_a])-temp*ones(1,N_a,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a,'gpuArray'));
            PolicyKron(2,:)=reshape(sum(PolicyTemp,1),[1,N_a]);
        end
    end    

elseif simoptions.gridinterplayer==1
    % Reshape Policy
    % Policy=reshape(Policy,[size(Policy,1),N_a]);
    
    if n_d(1)==0
        if l_a==1 || l_a==2
            PolicyKron=Policy; % a1, possibly a2, L2
        else %l_a>2
            PolicyKron=zeros(3,N_a,'gpuArray');
            PolicyKron(1,:)=Policy(1,:); % a1
            temp=[0; ones(l_a-2,1,'gpuArray')];
            temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
            PolicyTemp=(reshape(Policy(2:l_a,:),[l_a-1,N_a])-temp).*[1;temp2(1:end-1)];
            PolicyKron(2,:)=reshape(sum(PolicyTemp,1),[1,N_a]);
            PolicyKron(3,:)=Policy(l_a+1,:); % L2 index
        end
    else
        l_d=length(n_d);
        if l_a==1
            PolicyKron=zeros(3,N_a,'gpuArray');
        else
            PolicyKron=zeros(4,N_a,'gpuArray');
        end

        if l_d==1
            PolicyKron(1,:)=Policy(1,:);
        else
            temp=[0; ones(l_d-1,1,'gpuArray')];
            temp2=gpuArray(cumprod(n_d')); % column vector
            PolicyTemp=(reshape(Policy(1:l_d,:),[l_d,N_a])-temp).*[1;temp2(1:end-1)];
            PolicyKron(1,:)=reshape(sum(PolicyTemp,1),[1,N_a]);
        end
        % Then, a
        if l_a==1
            PolicyKron(2,:)=Policy(l_d+1,:);
            PolicyKron(3,:)=Policy(l_a+l_d+1,:); % L2 index
        elseif l_a==2
            PolicyKron(2,:)=Policy(l_d+1,:);
            PolicyKron(3,:)=Policy(l_d+2,:);            
            PolicyKron(4,:)=Policy(l_a+l_d+1,:); % L2 index
        else
            PolicyKron(2,:)=Policy(l_d+1,:,:);
            temp=[0; ones(l_a-2,1,'gpuArray')];
            temp2=gpuArray(cumprod(n_a(2:end)')); % column vector
            PolicyTemp=(reshape(Policy(l_d+2:l_d+l_a,:,:),[l_a-1,N_a])-temp).*[1;temp2(1:end-1)];
            PolicyKron(3,:)=reshape(sum(PolicyTemp,1),[1,N_a]);
            PolicyKron(4,:)=Policy(l_a+l_d+1,:,:); % L2 index
        end

    end

end






end