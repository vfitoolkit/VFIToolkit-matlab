function Policy=UnKronPolicyIndexes_Case1_FHorz_e(PolicyKron, n_d,n_a, n_z,n_e,N_j, vfoptions)
% Can use vfoptions OR simoptions
% Input: PolicyKron is (2,N_a,N_z,N_e,N_j) first dim indexes the optimal choice for d and aprime
%                      (1,N_a,N_z,N_e,N_j) if there is no d
%    vfoptions.gridinterplayer=1 will mean the first dimension has one extra value (so 3 if d, 2 without)
% Output: Policy is (l_d+l_a,n_a,n_z,n_e,N_j);

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_aprime=length(n_a);
n_aprime=n_a;
extra=(vfoptions.gridinterplayer==1);

if n_d(1)==0
    Policy=zeros(l_aprime+extra,N_a,N_z,N_e,N_j,'gpuArray');
    Policy(1,:,:,:,:)=rem(PolicyKron(1,:,:,:,:)-1,n_aprime(1))+1;
    if l_aprime>1
        if l_aprime>2
            for ii=2:l_aprime-1
                Policy(ii,:,:,:,:)=rem(ceil(PolicyKron(1,:,:,:,:)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
            end
        end
        Policy(l_aprime,:,:,:,:)=ceil(PolicyKron(1,:,:,:,:)/prod(n_aprime(1:l_aprime-1)));
    end

    if vfoptions.gridinterplayer==1
        Policy(l_aprime+1,:,:,:,:)=PolicyKron(2,:,:,:,:);
    end

    Policy=reshape(Policy,[l_aprime+extra,n_a,n_z,n_e,N_j]);
else
    l_d=length(n_d);
    Policy=zeros(l_d+l_aprime+extra,N_a,N_z,N_e,N_j,'gpuArray');

    Policy(1,:,:,:,:)=rem(PolicyKron(1,:,:,:,:)-1,n_d(1))+1;
    if l_d>1
        if l_d>2
            for ii=2:l_d-1
                Policy(ii,:,:,:,:)=rem(ceil(PolicyKron(1,:,:,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
            end
        end
        Policy(l_d,:,:,:,:)=ceil(PolicyKron(1,:,:,:,:)/prod(n_d(1:l_d-1)));
    end

    Policy(l_d+1,:,:,:,:)=rem(PolicyKron(2,:,:,:,:)-1,n_aprime(1))+1;
    if l_aprime>1
        if l_aprime>2
            for ii=2:l_aprime-1
                Policy(l_d+ii,:,:,:,:)=rem(ceil(PolicyKron(2,:,:,:,:)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
            end
        end
        Policy(l_d+l_aprime,:,:,:,:)=ceil(PolicyKron(2,:,:,:,:)/prod(n_aprime(1:l_aprime-1)));
    end

    if vfoptions.gridinterplayer==1
        Policy(l_d+l_aprime+1,:,:,:,:)=PolicyKron(3,:,:,:,:);
    end

    Policy=reshape(Policy,[l_d+l_aprime+extra,n_a,n_z,n_e,N_j]);
end


end