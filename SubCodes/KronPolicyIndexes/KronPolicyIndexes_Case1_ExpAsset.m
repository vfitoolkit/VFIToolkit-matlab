function PolicyKron=KronPolicyIndexes_Case1_ExpAsset(Policy, n_d, n_a, n_z) %,options)
error('NOT USED FOR ANYTHING')

%
% Input: Policy (l_d+l_a-1,n_a,n_z);
%
% Output:
%    if l_d==1: Policy=zeros(2,N_a,N_z); 
%    if l_d>1:  Policy=zeros(3,N_a,N_z); 

n_aprime=n_a(1:end-1);
% n_expasset=n_a(end);

N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_aprime);

Policy=reshape(Policy,[size(Policy,1),N_a,N_z]);

% Impossible to have no decision variables when using experience asset
l_d=length(n_d);
if l_d==1 % no d1
    if isa(Policy,'gpuArray')
        PolicyKron=zeros(2,N_a,N_z,'gpuArray');       
        PolicyKron(1,:,:)=Policy(1,:,:);
        % Then, a
        if l_aprime==1        
            PolicyKron(2,:,:)=Policy(l_d+1,:,:);
        else
            temp=ones(l_aprime,1,'gpuArray')-eye(l_aprime,1,'gpuArray');
            temp2=gpuArray(cumprod(n_aprime')); % column vector
            PolicyTemp=(reshape(Policy(l_d+1:l_d+l_aprime,:,:),[l_aprime,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
            PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
        end
    else
        PolicyTemp=zeros(2,N_a,N_z);
        for i=1:N_a
            for j=1:N_z
                optdsub=Policy(1:l_d,i,j);
                optasub=Policy((l_d+1):(l_d+l_aprime),i,j);
                optD=sub2ind_homemade(n_d',optdsub);
                optA=sub2ind_homemade(n_aprime',optasub);
                PolicyTemp(:,i,j)=[optD;optA];
            end
        end
        PolicyKron=PolicyTemp; %Overwrite
    end
else % l_d>1, so there are d1
    n_d1=n_d(1:end-1);
    l_d1=length(n_d1);
    % Deal with d2, the decision which influence the experience asset first as this is trivial
    PolicyKron=zeros(3,N_a,N_z,'gpuArray'); 
    PolicyKron(2,:,:)=Policy(l_d,:,:);

    if isa(Policy,'gpuArray')
        % Deal with d1
        temp=ones(l_d1,1,'gpuArray')-eye(l_d1,1,'gpuArray');
        temp2=gpuArray(cumprod(n_d1')); % column vector
        PolicyTemp=(reshape(Policy(1:l_d1,:,:),[l_d1,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
        PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);

        % Then, a
        if l_aprime==1        
            PolicyKron(3,:,:)=Policy(l_d+1,:,:);
        else
            temp=ones(l_aprime,1,'gpuArray')-eye(l_aprime,1,'gpuArray');
            temp2=gpuArray(cumprod(n_aprime')); % column vector
            PolicyTemp=(reshape(Policy(l_d+1:l_d+l_aprime,:,:),[l_aprime,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
            PolicyKron(3,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z]);
        end
    else
        PolicyTemp=zeros(3,N_a,N_z);
        for i=1:N_a
            for j=1:N_z
                optd1sub=Policy(1:l_d1,i,j);
                optasub=Policy((l_d+1):(l_d+l_aprime),i,j);
                optD1=sub2ind_homemade(n_d1',optd1sub);
                optA=sub2ind_homemade(n_aprime',optasub);
                PolicyTemp(1,i,j)=optD1;
                PolicyTemp(3,i,j)=optA;
            end
        end
        PolicyKron=PolicyTemp; %Overwrite
    end
end


end