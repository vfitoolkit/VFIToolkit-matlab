function PolicyPath=UnKronPolicyIndexes_InfHorz_TransPath(PolicyPathKron, n_d, n_a, n_z,T,vfoptions)

% Input: PolicyPathKron=zeros(2,N_a,N_z,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,T) if there is no d
% Output: PolicyPath (l_d+l_a,n_a,n_z,T);
% Note: This can look slightly different based on vfoptions

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

l_aprime=length(n_a);

if N_d==0
    if vfoptions.gridinterplayer==0
        PolicyPath=zeros(l_aprime,N_a,N_z,T,'gpuArray');

        PolicyPath(1,:,:,:)=shiftdim(rem(PolicyPathKron(:,:,:)-1,n_a(1))+1,-1);
        if l_aprime>1
            if l_aprime>2
                for ii=2:l_aprime-1
                    PolicyPath(ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(:,:,:)/prod(n_a(1:ii-1)))-1,n_a(ii))+1,-1);
                end
            end
            PolicyPath(l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(:,:,:)/prod(n_a(1:l_aprime-1))),-1);
        end

        PolicyPath=reshape(PolicyPath,[l_aprime,n_a,n_z,T]);

    elseif vfoptions.gridinterplayer==1
        PolicyPath=zeros(l_aprime+1,N_a,N_z,T,'gpuArray');

        PolicyPath(1,:,:,:)=PolicyPathKron(1,:,:,:);
        if l_aprime==2
            PolicyPath(2,:,:,:)=PolicyPathKron(2,:,:,:);
        elseif l_aprime>2
            PolicyPath(2,:,:,:)=shiftdim(rem(PolicyPathKron(2,:,:,:)-1,n_a(2))+1,-1);
            for ii=3:l_aprime-1
                PolicyPath(ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(2,:,:,:)/prod(n_a(2:ii-1)))-1,n_a(ii))+1,-1);
            end
            PolicyPath(l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(2,:,:,:)/prod(n_a(2:l_aprime-1))),-1);
        end

        if l_aprime==1
            PolicyPath(l_aprime+1,:,:,:)=PolicyPathKron(2,:,:,:);
        else
            PolicyPath(l_aprime+1,:,:,:)=PolicyPathKron(3,:,:,:);
        end
        PolicyPath=reshape(PolicyPath,[l_aprime+1,n_a,n_z,T]);
    end

elseif N_d>0 
    l_d=length(n_d);
    if vfoptions.gridinterplayer==0
        PolicyPath=zeros(l_d+l_aprime,N_a,N_z,T,'gpuArray');

        PolicyPath(1,:,:,:)=shiftdim(rem(PolicyPathKron(1,:,:,:)-1,n_d(1))+1,-1);
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    PolicyPath(ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(1,:,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1,-1);
                end
            end
            PolicyPath(l_d,:,:,:)=shiftdim(ceil(PolicyPathKron(1,:,:,:)/prod(n_d(1:l_d-1))),-1);
        end

        PolicyPath(l_d+1,:,:,:)=shiftdim(rem(PolicyPathKron(2,:,:,:)-1,n_a(1))+1,-1);
        if l_aprime>1
            for ii=2:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(2,:,:,:)/prod(n_a(1:ii-1)))-1,n_a(ii))+1,-1);
            end
            PolicyPath(l_d+l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(2,:,:,:)/prod(n_a(1:l_aprime-1))),-1);
        end

        PolicyPath=reshape(PolicyPath,[l_d+l_aprime,n_a,n_z,T]);

    elseif vfoptions.gridinterplayer==1
        PolicyPath=zeros(l_d+l_aprime+1,N_a,N_z,T,'gpuArray');

        PolicyPath(1,:,:,:)=shiftdim(rem(PolicyPathKron(1,:,:,:)-1,n_d(1))+1,-1);
        if l_d>1
            if l_d>2
                for ii=2:l_d-1
                    PolicyPath(ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(1,:,:,:)/prod(n_d(1:ii-1)))-1,n_d(ii))+1,-1);
                end
            end
            PolicyPath(l_d,:,:,:)=shiftdim(ceil(PolicyPathKron(1,:,:,:)/prod(n_d(1:l_d-1))),-1);
        end

        PolicyPath(l_d+1,:,:,:)=PolicyPathKron(2,:,:,:);
        if l_aprime==2
            PolicyPath(l_d+2,:,:,:)=PolicyPathKron(3,:,:,:);
        elseif l_aprime>2
            PolicyPath(l_d+2,:,:,:)=shiftdim(rem(PolicyPathKron(3,:,:,:)-1,n_a(2))+1,-1);
            for ii=3:l_aprime-1
                PolicyPath(l_d+ii,:,:,:)=shiftdim(rem(ceil(PolicyPathKron(3,:,:,:)/prod(n_a(2:ii-1)))-1,n_a(ii))+1,-1);
            end
            PolicyPath(l_d+l_aprime,:,:,:)=shiftdim(ceil(PolicyPathKron(3,:,:,:)/prod(n_a(2:l_aprime-1))),-1);
        end

        if l_aprime==1
            PolicyPath(l_d+l_aprime+1,:,:,:)=PolicyPathKron(3,:,:,:);
        else
            PolicyPath(l_d+l_aprime+1,:,:,:)=PolicyPathKron(4,:,:,:);
        end
        PolicyPath=reshape(PolicyPath,[l_d+l_aprime+1,n_a,n_z,T]);
    end

end


end