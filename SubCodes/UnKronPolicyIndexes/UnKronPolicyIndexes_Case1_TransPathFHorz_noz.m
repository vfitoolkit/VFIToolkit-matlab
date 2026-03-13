function PolicyPath=UnKronPolicyIndexes_Case1_TransPathFHorz_noz(PolicyPathKron, n_d, n_a,N_j,T,vfoptions)
% Can use vfoptions OR simoptions
% Input: PolicyKron is (2,N_a,N_j,T) first dim indexes the optimal choice for d and aprime
%                      (1,N_a,N_j,T) if there is no d
%    vfoptions.gridinterplayer=1 will mean the first dimension has one extra value (so 3 if d, 2 without)
% Output: Policy is (l_d+l_a,n_a,N_j,T);

% N_d=prod(n_d);
N_a=prod(n_a);

l_aprime=length(n_a);
n_aprime=n_a;
extra=(vfoptions.gridinterplayer==1);

if n_d(1)==0
    PolicyPath=zeros(l_aprime+extra,N_a,N_j,T,'gpuArray');
    PolicyPath(1,:,:,:)=rem(PolicyPathKron(1,:,:,:)-1,n_aprime(1))+1;
    if l_aprime>1
        if l_aprime>2
            for ii=2:l_aprime-1
                PolicyPath(ii,:,:,:)=rem(ceil(PolicyPathKron(1,:,:,:)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
            end
        end
        PolicyPath(l_aprime,:,:,:)=ceil(PolicyPathKron(1,:,:,:)/prod(n_aprime(1:l_aprime-1)));
    end

    if vfoptions.gridinterplayer==1
        PolicyPath(l_aprime+1,:,:,:)=PolicyPathKron(2,:,:,:);
    end

    PolicyPath=reshape(PolicyPath,[l_aprime+extra,n_a,N_j,T]);
else
    l_d=length(n_d);
    PolicyPath=zeros(l_d+l_aprime+extra,N_a,N_j,T,'gpuArray');

    PolicyPath(1,:,:,:)=rem(PolicyPathKron(1,:,:,:)-1,n_d(1))+1;
    if l_d>1
        if l_d>2
            for ii=2:l_d-1
                PolicyPath(ii,:,:,:)=rem(ceil(PolicyPathKron(1,:,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
            end
        end
        PolicyPath(l_d,:,:,:)=ceil(PolicyPathKron(1,:,:,:)/prod(n_d(1:l_d-1)));
    end

    PolicyPath(l_d+1,:,:,:)=rem(PolicyPathKron(2,:,:,:)-1,n_a(1))+1;
    if l_aprime>1
        if l_aprime>2
            for ii=2:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=rem(ceil(PolicyPathKron(2,:,:,:)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
            end
        end
        PolicyPath(l_d+l_aprime,:,:,:)=ceil(PolicyPathKron(2,:,:,:)/prod(n_aprime(1:l_aprime-1)));
    end

    if vfoptions.gridinterplayer==1
        PolicyPath(l_d+l_aprime+1,:,:,:)=PolicyPathKron(3,:,:,:);
    end

    PolicyPath=reshape(PolicyPath,[l_d+l_aprime+extra,n_a,N_j,T]);
end


end