function Policy=UnKronPolicyIndexes_Case1(PolicyKron, n_d, n_a, n_z,vfoptions)

% Input: PolicyKron=zeros(2,N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z) if there is no d
% Output: Policy (l_d+l_a,n_a,n_z);
% Note: simoptions.gridinterplayer=1 means there will be an additional index for the second layer in both the input and output versions

N_a=prod(n_a);
N_z=prod(n_z);

l_a=length(n_a);

if vfoptions.gridinterplayer==0

    if vfoptions.parallel==2
        if n_d(1)==0
            Policy=zeros(l_a,N_a,N_z,'gpuArray');

            Policy(1,:,:)=shiftdim(rem(PolicyKron-1,n_a(1))+1,-1);
            if l_a>1
                if l_a>2
                    for ii=2:l_a-1
                        Policy(ii,:,:)=shiftdim(rem(ceil(PolicyKron/prod(n_a(1:ii-1)))-1,n_a(ii))+1,-1);
                    end
                end
                Policy(l_a,:,:)=shiftdim(ceil(PolicyKron/prod(n_a(1:l_a-1))),-1);
            end

            Policy=reshape(Policy,[l_a,n_a,n_z]);
        else
            l_d=length(n_d);
            l_da=length(n_d)+length(n_a);
            n_da=[n_d,n_a];
            Policy=zeros(l_da,N_a,N_z,'gpuArray');

            Policy(1,:,:)=rem(PolicyKron(1,:,:)-1,n_da(1))+1;
            if l_d>1
                if l_d>2
                    for ii=2:l_d-1
                        Policy(ii,:,:)=rem(ceil(PolicyKron(1,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
                    end
                end
                Policy(l_d,:,:)=ceil(PolicyKron(1,:,:)/prod(n_d(1:l_d-1)));
            end

            Policy(l_d+1,:,:)=rem(PolicyKron(2,:,:)-1,n_a(1))+1;
            if l_a>1
                if l_a>2
                    for ii=2:l_a-1
                        Policy(l_d+ii,:,:)=rem(ceil(PolicyKron(2,:,:)/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
                    end
                end
                Policy(l_da,:,:)=ceil(PolicyKron(2,:,:)/prod(n_a(1:l_a-1)));
            end

            Policy=reshape(Policy,[l_da,n_a,n_z]);
        end

    else 
        % On CPU
        if n_d(1)==0
            Policy=zeros(l_a,N_a,N_z);
            for i=1:N_a
                for j=1:N_z
                    optaindexKron=PolicyKron(i,j);
                    optA=ind2sub_homemade(n_a',optaindexKron);
                    Policy(:,i,j)=optA';
                end
            end
            Policy=reshape(Policy,[l_a,n_a,n_z]);
        else
            l_d=length(n_d);
            Policy=zeros(l_d+l_a,N_a,N_z);
            for i=1:N_a
                for j=1:N_z
                    optdindexKron=PolicyKron(1,i,j);
                    optaindexKron=PolicyKron(2,i,j);
                    optD=ind2sub_homemade(n_d',optdindexKron);
                    optA=ind2sub_homemade(n_a',optaindexKron);
                    Policy(:,i,j)=[optD';optA'];
                end
            end
            Policy=reshape(Policy,[l_d+l_a,n_a,n_z]);
        end
    end

elseif vfoptions.gridinterplayer==1

    if n_d(1)==0
        Policy=zeros(l_a+1,N_a,N_z,'gpuArray');

        Policy(1,:,:)=shiftdim(rem(PolicyKron(1,:,:)-1,n_a(1))+1,-1);
        if l_a>1
            if l_a>2
                for ii=2:l_a-1
                    Policy(ii,:,:)=shiftdim(rem(ceil(PolicyKron(1,:,:)/prod(n_a(1:ii-1)))-1,n_a(ii))+1,-1);
                end
            end
            Policy(l_a,:,:)=shiftdim(ceil(PolicyKron(1,:,:)/prod(n_a(1:l_a-1))),-1);
        end
        Policy(l_a+1,:,:)=PolicyKron(2,:,:); % L2 index

        Policy=reshape(Policy,[l_a+1,n_a,n_z]);
    else
        l_d=length(n_d);
        l_da=length(n_d)+length(n_a);
        n_da=[n_d,n_a];
        Policy=zeros(l_da+1,N_a,N_z,'gpuArray');

        Policy(1,:,:)=rem(PolicyKron(1,:,:)-1,n_da(1))+1;
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    Policy(ii,:,:)=rem(ceil(PolicyKron(1,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
                end
            end
            Policy(l_d,:,:)=ceil(PolicyKron(1,:,:)/prod(n_d(1:l_d-1)));
        end

        Policy(l_d+1,:,:)=rem(PolicyKron(2,:,:)-1,n_a(1))+1;
        if l_a>1
            if l_a>2
                for ii=2:l_a-1
                    Policy(l_d+ii,:,:)=rem(ceil(PolicyKron(2,:,:)/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
                end
            end
            Policy(l_da,:,:)=ceil(PolicyKron(2,:,:)/prod(n_a(1:l_a-1)));
        end
        Policy(l_da+1,:,:)=PolicyKron(3,:,:); % L2 index

        Policy=reshape(Policy,[l_da+1,n_a,n_z]);
    end

end


end