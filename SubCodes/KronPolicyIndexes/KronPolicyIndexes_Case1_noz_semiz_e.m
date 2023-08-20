function PolicyKron=KronPolicyIndexes_Case1_noz_semiz_e(Policy, n_d1, n_d2, n_a, n_semiz, n_e) %,options)

%Input: Policy (l_d+l_a,n_a,n_semiz,n_e);

%Output: Policy=zeros(3,N_a,N_semiz,N_e); %first dim indexes the optimal choice for d1, d2 and aprime rest of dimensions a,z 
%                    (2,N_a,N_semiz,N_e) if there is no d1

N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_e=prod(n_e);

l_d2=length(n_d2);
l_a=length(n_a);

Policy=reshape(Policy,[size(Policy,1),N_a,N_semiz*N_e]);

if n_d1(1)==0
  
    if isa(Policy,'gpuArray')
        % Should test whether code runs faster with this predeclaration of PolicyKron commented or uncommented
        % PolicyKron=zeros(2,N_a,N_z,N_semiz,'gpuArray'); 
        
        if l_d2==1        
            PolicyKron(1,:,:,:)=reshape(Policy(1,:,:),[1,N_a,N_semiz,N_e]);
        else
            temp=ones(l_d2,1,'gpuArray')-eye(l_d2,1,'gpuArray');
            temp2=gpuArray(cumprod(n_d2')); % column vector
            PolicyTemp=(reshape(Policy(1:l_d2,:,:),[l_d2,N_a*N_semiz*N_e])-temp*ones(1,N_a*N_semiz*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_semiz*N_e,'gpuArray'));
            PolicyKron(1,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_semiz,N_e]);
        end
        % Then, a
        if l_a==1        
            PolicyKron(2,:,:,:)=reshape(Policy(l_d2+1,:,:),[1,N_a,N_semiz,N_e]);
        else
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy(l_d2+1:l_d+l_a,:,:),[l_a,N_a*N_semiz*N_e])-temp*ones(1,N_a*N_semiz*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_semiz*N_e,'gpuArray'));
            PolicyKron(2,:,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_semiz,N_e]);
        end
    else
        PolicyKron=zeros(2,N_a,N_semiz*N_e);
        for i=1:N_a
            for j=1:N_semiz*N_e
                optd2sub=Policy(1:l_d2,i,j);
                optasub=Policy((l_d2+1):(l_d2+l_a),i,j);
                optD=sub2ind_homemade(n_d',optd2sub);
                optA=sub2ind_homemade(n_a',optasub);
                PolicyKron(:,i,j)=[optD;optA];
            end
        end
        PolicyKron=reshape(PolicyKron,[2,N_a,N_semiz,N_e]); %Overwrite
    end
    
else
    l_d1=length(n_d1);
  
    if isa(Policy,'gpuArray')
        % Should test whether code runs faster with this predeclaration of PolicyKron commented or uncommented
        % PolicyKron=zeros(3,N_a,N_z,'gpuArray'); 
        
        if l_d1==1        
            PolicyKron(1,:,:,:)=reshape(Policy(1,:,:),[1,N_a,N_semiz,N_e]);
        else
            temp=ones(l_d1,1,'gpuArray')-eye(l_d1,1,'gpuArray');
            temp2=gpuArray(cumprod(n_d1')); % column vector
            PolicyTemp=(reshape(Policy(1:l_d1,:,:),[l_d1,N_a*N_semiz*N_e])-temp*ones(1,N_a*N_semiz*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_semiz*N_e,'gpuArray'));
            PolicyKron(1,:,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_semiz,N_e]);
        end
        % Then, d2
        if l_d2==1        
            PolicyKron(2,:,:,:)=reshape(Policy(l_d1+1,:,:),[1,N_a,N_semiz,N_e]);
        else
            temp=ones(l_d2,1,'gpuArray')-eye(l_d2,1,'gpuArray');
            temp2=gpuArray(cumprod(n_d2')); % column vector
            PolicyTemp=(reshape(Policy(l_d1+1:l_d1+l_d2,:,:),[l_d2,N_a*N_semiz*N_e])-temp*ones(1,N_a*N_semiz*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_semiz*N_e,'gpuArray'));
            PolicyKron(2,:,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_semiz,N_e]);
        end
        % Then, a
        if l_a==1        
            PolicyKron(3,:,:,:)=reshape(Policy(l_d1+l_d2+1,:,:),[1,N_a,N_semiz,N_e]);
        else
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy(l_d1+l_d2+1:l_d1+l_d2+l_a,:,:),[l_a,N_a*N_semiz*N_e])-temp*ones(1,N_a*N_semiz*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_semiz*N_e,'gpuArray'));
            PolicyKron(3,:,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_semiz,N_e]);
        end

    else
        PolicyKron=zeros(3,N_a,N_semiz*N_e);
        for i=1:N_a
            for j=1:N_semiz*N_e
                optd1sub=Policy(1:l_d1,i,j);
                optd2sub=Policy((l_d1+1):(l_d1+l_d2),i,j);
                optasub=Policy((l_d1+l_d2+1):(l_d1+l_d2+l_a),i,j);
                optD1=sub2ind_homemade(n_d1',optd1sub);
                optD2=sub2ind_homemade(n_d2',optd2sub);
                optA=sub2ind_homemade(n_a',optasub);
                PolicyKron(:,i,j)=[optD1;optD2;optA];
            end
        end
        PolicyKron=reshape(PolicyKron,[3,N_a,N_semiz,N_e]); %Overwrite
    end
end


end