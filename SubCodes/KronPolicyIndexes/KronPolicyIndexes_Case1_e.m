function PolicyKron=KronPolicyIndexes_Case1_e(Policy, n_d, n_a, n_z, n_e) %,options)

%Input: Policy (l_d+l_a,n_a,n_z,n_e);

%Output: Policy=zeros(2,N_a,N_z,N_e); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e) if there is no d

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_a=length(n_a);

Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_e]);

if n_d(1)==0
    
    if isa(Policy,'gpuArray') %options.parallel==2
        if l_a==1        
            PolicyKron=shiftdim(Policy,1);
        else %l_a>1
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy,[l_a,N_a*N_z*N_e])-temp*ones(1,N_a*N_z*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z*N_e,'gpuArray'));
            PolicyKron=reshape(sum(PolicyTemp,1),[N_a,N_z,N_e]);
        end            
    else
        PolicyTemp=zeros(N_a,N_z,N_e);
        for i=1:N_a
            for j=1:N_z
                for k=1:N_e
                    optasub=Policy(1:l_a,i,j,k);
                    optA=sub2ind_homemade([n_a'],optasub);
                    PolicyTemp(i,j,k)=optA;
                end
            end
        end
        PolicyKron=PolicyTemp; %Overwrite
    end
    
else
    l_d=length(n_d);
  
    if isa(Policy,'gpuArray')
        % Should test whether code runs faster with this predeclaration of PolicyKron commented or uncommented
        % PolicyKron=zeros(2,N_a,N_z,'gpuArray'); 
        
        if l_d==1        
            PolicyKron(1,:,:,:)=Policy(1,:,:,:);
        else
            temp=ones(l_d,1,'gpuArray')-eye(l_d,1,'gpuArray');
            temp2=gpuArray(cumprod(n_d')); % column vector
            PolicyTemp=(reshape(Policy(1:l_d,:,:),[l_d,N_a*N_z*N_e])-temp*ones(1,N_a*N_z*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z*N_e,'gpuArray'));
            PolicyKron(1,:,:)=reshape(sum(PolicyTemp,1),[N_a,N_z,N_e]);
        end
        % Then, a
        if l_a==1        
            PolicyKron(2,:,:,:)=Policy(l_d+1,:,:,:);
        else
            temp=ones(l_a,1,'gpuArray')-eye(l_a,1,'gpuArray');
            temp2=gpuArray(cumprod(n_a')); % column vector
            PolicyTemp=(reshape(Policy(l_d+1:l_d+l_a,:,:),[l_a,N_a*N_z*N_e])-temp*ones(1,N_a*N_z*N_e,'gpuArray')).*([1;temp2(1:end-1)]*ones(1,N_a*N_z*N_e,'gpuArray'));
            PolicyKron(2,:,:)=reshape(sum(PolicyTemp,1),[1,N_a,N_z,N_e]);
        end
    else
        PolicyTemp=zeros(2,N_a,N_z,N_e);
        for i=1:N_a
            for j=1:N_z
                for k=1:N_e
                    optdsub=Policy(1:l_d,i,j,k);
                    optasub=Policy((l_d+1):(l_d+l_a),i,j,k);
                    optD=sub2ind_homemade(n_d',optdsub);
                    optA=sub2ind_homemade(n_a',optasub);
                    PolicyTemp(:,i,j,k)=[optD;optA];
                end
            end
        end
        PolicyKron=PolicyTemp; %Overwrite
    end
end


end