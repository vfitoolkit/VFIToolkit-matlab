function PolicyPath=UnKronPolicyIndexes_InfHorz_TransPath_ExpAsset(PolicyPathKron, n_d, n_a1, n_a, n_z,T,vfoptions,justfirstdim)

% Input: PolicyPathKron=zeros(2,N_a,N_z,T); % if no d1: this is d2,a1prime
%        PolicyPathKron=zeros(3,N_a,N_z,T); % d1,d2,a1prime
% Output: PolicyPath (l_d+l_a,n_a,n_z,T);
% Note: This can look slightly different based on vfoptions

N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_a1);

l_d=length(n_d); % expasset so has to be a least 1 due to d2
if l_d==1
    l_d1=0;
else % l_d>1
    n_d1=n_d(1:end-1);
    l_d1=length(n_d1);
end

if vfoptions.gridinterplayer==0
    PolicyPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');

    if l_d1==0
        % d2
        PolicyPath(1,:,:,:)=PolicyPathKron(1,:,:,:);

        % a1prime
        PolicyPath(l_d+1,:,:,:)=shiftdim(rem(PolicyPathKron(2,:,:,:)-1,n_a1(1))+1,-1);
        if l_aprime>1
            for ii=2:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(2,:,:,:)/prod(n_a1(1:ii-1)))-1,n_a1(ii))+1,-1);
            end
            PolicyPath(l_d+l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(2,:,:,:)/prod(n_a1(1:l_aprime-1))),-1);
        end

    else
        % d1
        PolicyPath(1,:,:,:)=shiftdim(rem(PolicyPathKron(1,:,:,:)-1,n_d1(1))+1,-1);
        if l_d1>1
            if l_d1>2
                for ii=2:l_d1-1
                    PolicyPath(ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(1,:,:,:)/prod(n_d1(1:ii-1)))-1,n_d1(ii))+1,-1);
                end
            end
            PolicyPath(l_d1,:,:,:)=shiftdim(ceil(PolicyPathKron(1,:,:,:)/prod(n_d1(1:l_d1-1))),-1);
        end

        % d2
        PolicyPath(l_d1+1,:,:,:)=PolicyPathKron(2,:,:,:);

        % a1prime
        PolicyPath(l_d+1,:,:,:)=shiftdim(rem(PolicyPathKron(3,:,:,:)-1,n_a1(1))+1,-1);
        if l_aprime>1
            for ii=2:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(3,:,:,:)/prod(n_a1(1:ii-1)))-1,n_a1(ii))+1,-1);
            end
            PolicyPath(l_d+l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(3,:,:,:)/prod(n_a1(1:l_aprime-1))),-1);
        end
    end
    
    if justfirstdim==0
        PolicyPath=reshape(PolicyPath,[l_d+l_aprime,n_a,n_z,T]);
    end

elseif vfoptions.gridinterplayer==1
    PolicyPath=zeros(l_d+l_aprime+1,N_a,N_z,T,'gpuArray');

    if l_d1==0
        % d2
        PolicyPath(1,:,:,:)=PolicyPathKron(1,:,:,:);

        % a1prime
        PolicyPath(l_d+1,:,:,:)=PolicyPathKron(2,:,:,:);
        if l_aprime==2
            PolicyPath(l_d+2,:,:,:)=PolicyPathKron(3,:,:,:);
        elseif l_aprime>2
            PolicyPath(l_d+2,:,:,:)=shiftdim(rem(PolicyPathKron(3,:,:,:)-1,n_a1(2))+1,-1);
            for ii=3:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(3,:,:,:)/prod(n_a1(2:ii-1)))-1,n_a1(ii))+1,-1);
            end
            PolicyPath(l_d+l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(3,:,:,:)/prod(n_a1(2:l_aprime-1))),-1);
        end
    else
        % d1
        PolicyPath(1,:,:,:)=shiftdim(rem(PolicyPathKron(1,:,:,:)-1,n_d1(1))+1,-1);
        if l_d1>1
            if l_d1>2
                for ii=2:l_d1-1
                    PolicyPath(ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(1,:,:,:)/prod(n_d1(1:ii-1)))-1,n_d1(ii))+1,-1);
                end
            end
            PolicyPath(l_d1,:,:,:)=shiftdim(ceil(PolicyPathKron(1,:,:,:)/prod(n_d1(1:l_d1-1))),-1);
        end

        PolicyPath(l_d1+1,:,:,:)=PolicyPathKron(2,:,:,:); % d2

        % a1prime
        PolicyPath(l_d+1,:,:,:)=PolicyPathKron(3,:,:,:);
        if l_aprime==2
            PolicyPath(l_d+2,:,:,:)=PolicyPathKron(4,:,:,:);
        elseif l_aprime>2
            PolicyPath(l_d+2,:,:,:)=shiftdim(rem(PolicyPathKron(4,:,:,:)-1,n_a1(2))+1,-1);
            for ii=3:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(4,:,:,:)/prod(n_a1(2:ii-1)))-1,n_a1(ii))+1,-1);
            end
            PolicyPath(l_d+l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(4,:,:,:)/prod(n_a1(2:l_aprime-1))),-1);
        end
    end

    % L2 index
    PolicyPath(l_d+l_aprime+1,:,:,:)=PolicyPathKron(end,:,:,:);

    if justfirstdim==0
        PolicyPath=reshape(PolicyPath,[l_d+l_aprime+1,n_a,n_z,T]);
    end
end


end