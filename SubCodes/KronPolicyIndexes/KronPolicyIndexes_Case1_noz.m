function PolicyKron=KronPolicyIndexes_Case1_noz(Policy, n_d, n_a) %,options)

%Input: Policy (l_d+l_a,n_a);

%Output: Policy=zeros(2,N_a); %first dim indexes the optimal choice for d and aprime rest of dimensions a
%                       (N_a) if there is no d

N_a=prod(n_a);

l_a=length(n_a);

Policy=reshape(Policy,[size(Policy,1),N_a]);

if n_d(1)==0
    
    if isa(Policy,'gpuArray') %options.parallel==2
        if l_a==1        
            PolicyKron=shiftdim(Policy,1);
        else %l_a>1
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy,[l_a,N_a])-temp*ones(1,N_a,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a,'gpuArray'));
            PolicyKron=reshape(sum(PolicyTemp,1),[N_a,1]);
        end            
    else
        PolicyTemp=zeros(N_a);
        for i=1:N_a
            optasub=Policy(1:l_a,i);
            optA=sub2ind_homemade([n_a'],optasub);
            PolicyTemp(i)=optA;
        end
        PolicyKron=PolicyTemp; %Overwrite
    end
    
else
    l_d=length(n_d);
  
    if isa(Policy,'gpuArray')
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
    else
        PolicyTemp=zeros(2,N_a);
        for i=1:N_a
            optdsub=Policy(1:l_d,i);
            optasub=Policy((l_d+1):(l_d+l_a),i);
            optD=sub2ind_homemade(n_d',optdsub);
            optA=sub2ind_homemade(n_a',optasub);
            PolicyTemp(:,i)=[optD;optA];
        end
        PolicyKron=PolicyTemp; %Overwrite
    end
end


end