function PolicyKron=KronPolicyIndexes_Case1(Policy, n_d, n_a, n_z,options)

%Input: Policy (l_d+l_a,n_a,n_z);

%Output: Policy=zeros(2,N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a);

if n_d(1)==0
    
    if options.parallel~=2
        PolicyTemp=zeros(N_a,N_z);
        for i=1:N_a
            for j=1:N_z
                optasub=Policy(1,i,j);
                optA=sub2ind_homemade([n_a'],optasub);
                PolicyTemp(i,j)=optA;
            end
        end
        PolicyKron=PolicyTemp; %Overwrite
    else
        
        if l_a==1        
            PolicyKron=shiftdim(Policy,1);
        else %l_a>1
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy,[l_a,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
            PolicyKron=reshape(sum(PolicyTemp,1),[N_a,N_z]);
%             temp=ones(l_a,1)-eye(l_a,1);
%             temp2=cumprod(n_a);
%             PolicyTemp=shiftdim((Policy-temp).*([0,temp2(1:end-1)]*ones(1,N_a,N_z)),1);           
        end            

    end
    
else
    l_d=length(n_d);
    
    if options.parallel~=2
        PolicyTemp=zeros(2,N_a,N_z);
        for i=1:N_a
            for j=1:N_z
                optdsub=Policy(1,i,j);
                optasub=Policy(2,i,j);
                optD=sub2ind_homemade(n_d',optdsub);
                optA=sub2ind_homemade(n_a',optasub);
                PolicyTemp(:,i,j)=[optD;optA];
            end
        end
        PolicyKron=PolicyTemp; %Overwrite
    else
%         PolicyTemp=zeros(2,N_a,N_z,'gpuArray');
        
        % First, d
        if l_d==1        
            PolicyKron(1,:,:)=Policy(1,:,:);
        else
            temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
            temp2=gpuArray(cumprod(n_d')); % column vector
            PolicyTemp=(reshape(Policy(1:l_d,:,:),[l_d,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
            PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
%             temp=ones(l_d,1)-eye(l_d,1);
%             temp2=cumprod(n_d);
%             PolicyTemp(1,:,:)=shiftdim((Policy(1:l_d,:,:)-temp).*([0,temp2(1:end-1)]*ones(1,N_a,N_z)),1);
        end
        % Then, a
        if l_a==1        
            PolicyKron(2,:,:)=Policy(l_d+1,:,:);
        else
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy(l_d+1:l_d+l_a,:,:),[l_a,N_a*N_z])-temp*ones(1,N_a*N_z,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z,'gpuArray'));
            PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z]);
%             temp=ones(l_a,1)-eye(l_a,1);
%             temp2=cumprod(n_a);
%             PolicyTemp(2,:,:)=shiftdim((Policy(l_d+1:l_d+l_a,:,:)-temp).*([0,temp2(1:end-1)]*ones(1,N_a,N_z)),1);
        end
        
%         Policy=PolicyTemp;
    end
    
end


end