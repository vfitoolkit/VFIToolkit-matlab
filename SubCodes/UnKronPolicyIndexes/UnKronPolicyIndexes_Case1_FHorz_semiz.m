function Policy=UnKronPolicyIndexes_Case1_FHorz_semiz(PolicyKron, n_d1, n_d2, n_a, n_semiz,n_z,N_j,vfoptions)
% Can use vfoptions OR simoptions
% Input: PolicyKron is (3,N_a,N_semiz,N_z,N_j); % first dim indexes the optimal choice for (d1,d2,aprime)
%     without d1, the first dim is 2 and indexes the optimal choice for (d2,aprime)
%    vfoptions.gridinterplayer=1 will mean the first dimension has one extra value (so 4 if d1, 3 without)
% Output: Policy (l_d+l_a,n_a,n_semiz,n_z,N_j);

N_d1=prod(n_d1);
% N_d2=prod(n_d2);
N_a=prod(n_a);
N_semiz=prod(n_semiz);
N_z=prod(n_z);

l_d2=length(n_d2);

l_aprime=length(n_a);
n_aprime=n_a;
extra=(vfoptions.gridinterplayer==1);

% Only for GPU
if N_d1==0
    Policy=zeros(l_d2+l_aprime+extra,N_a,N_semiz,N_z,N_j,'gpuArray');

    Policy(1,:,:,:,:)=rem(PolicyKron(1,:,:,:,:)-1,n_d2(1))+1;
    if l_d2>1
        if l_d2>2
            for ii=1:l_d2-1
                Policy(ii,:,:,:,:)=rem(ceil(PolicyKron(1,:,:,:,:)/prod(n_d2(1:ii-1)))-1,n_d2(ii))+1;
            end
        end
        Policy(l_d2,:,:,:,:)=ceil(PolicyKron(1,:,:,:,:)/prod(n_d2(1:l_d2-1)));
    end

    Policy(l_d2+1,:,:,:,:)=rem(PolicyKron(2,:,:,:,:)-1,n_aprime(1))+1;
    if l_aprime>1
        if l_aprime>2
            for ii=1:l_aprime-1
                Policy(l_d2+ii,:,:,:,:)=rem(ceil(PolicyKron(2,:,:,:,:)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
            end
        end
        Policy(l_d2+l_aprime,:,:,:,:)=ceil(PolicyKron(2,:,:,:,:)/prod(n_aprime(1:l_aprime-1)));
    end

    if vfoptions.gridinterplayer==1
        Policy(l_d2+l_aprime+1,:,:,:,:)=PolicyKron(3,:,:,:,:);
    end

    Policy=reshape(Policy,[l_d2+l_aprime+extra,n_a,n_semiz,n_z,N_j]);

else % N_d1>0
    l_d1=length(n_d1); % Note: this is anyway only used is N_d1~=0

    Policy=zeros(l_d1+l_d2+l_aprime+extra,N_a,N_semiz,N_z,N_j,'gpuArray');

    Policy(1,:,:,:,:)=rem(PolicyKron(1,:,:,:,:)-1,n_d1(1))+1;
    if l_d1>1
        if l_d1>2
            for ii=1:l_d1-1
                Policy(ii,:,:,:,:)=rem(ceil(PolicyKron(1,:,:,:,:)/prod(n_d1(1:ii-1)))-1,n_d1(ii))+1;
            end
        end
        Policy(l_d1,:,:,:,:)=ceil(PolicyKron(1,:,:,:,:)/prod(n_d1(1:l_d1-1)));
    end
    
    Policy(l_d1+1,:,:,:,:)=rem(PolicyKron(2,:,:,:,:)-1,n_d2(1))+1;
    if l_d2>1
        if l_d2>2
            for ii=1:l_d2-1
                Policy(l_d1+ii,:,:,:,:)=rem(ceil(PolicyKron(2,:,:,:,:)/prod(n_d2(1:ii-1)))-1,n_d2(ii))+1;
            end
        end
        Policy(l_d1+l_d2,:,:,:,:)=ceil(PolicyKron(2,:,:,:,:)/prod(n_d2(1:l_d2-1)));
    end

    Policy(l_d1+l_d2+1,:,:,:,:)=rem(PolicyKron(3,:,:,:,:)-1,n_aprime(1))+1;
    if l_aprime>1
        if l_aprime>2
            for ii=1:l_aprime-1
                Policy(l_d1+l_d2+ii,:,:,:,:)=rem(ceil(PolicyKron(3,:,:,:,:)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
            end
        end
        Policy(l_d1+l_d2+l_aprime,:,:,:,:)=ceil(PolicyKron(3,:,:,:,:)/prod(n_aprime(1:l_aprime-1)));
    end

    if vfoptions.gridinterplayer==1
        Policy(l_d1+l_d2+l_aprime+1,:,:,:,:)=PolicyKron(4,:,:,:,:);
    end

    Policy=reshape(Policy,[l_d1+l_d2+l_aprime+extra,n_a,n_semiz,n_z,N_j]);
end


end
