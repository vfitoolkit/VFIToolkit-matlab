function PolicyPath=UnKronPolicyIndexes_InfHorz_TransPath(PolicyPathKron, n_d, n_a, n_z,T,vfoptions)

% Input: PolicyPath=zeros(2,N_a,N_z,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,T) if there is no d
% Output: PolicyPath (l_d+l_a,n_a,n_z,T);
% Note: This can look slightly different based on vfoptions

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);

if N_d==0 && vfoptions.gridinterplayer==0
    l_aprime=length(n_a);

    PolicyPath=zeros(l_aprime,N_a,N_z,T-1,'gpuArray');
    for tt=1:T
        PolicyPath(1,:,:,tt)=shiftdim(rem(PolicyPathKron(:,:,tt)-1,n_a(1))+1,-1);
        if l_aprime>1
            if l_aprime>2
                for ii=2:l_aprime-1
                    PolicyPath(ii,:,:,tt)=shiftdim(rem(ceil(PolicyPathKron(:,:,tt)/prod(n_a(1:ii-1)))-1,n_a(ii))+1,-1);
                end
            end
            PolicyPath(l_aprime,:,:,tt)=shiftdim(ceil(PolicyPathKron(:,:,tt)/prod(n_a(1:l_aprime-1))),-1);
        end
    end

    PolicyPath=reshape(PolicyPath,[l_aprime,n_a,n_z,T]);
        
else
    l_daprime=size(PolicyPathKron,1);
    if vfoptions.gridinterplayer==0
        n_daprime=[n_d,n_a];
    elseif vfoptions.gridinterplayer==1
        n_daprime=[n_d,n_a,vfoptions.ngridinterp];
    end
    
    PolicyPath=zeros(l_daprime,N_a,N_z,T-1,'gpuArray');
    for tt=1:T
        PolicyPath(1,:,:,tt)=rem(PolicyPathKron(1,:,:,tt)-1,n_daprime(1))+1;
        if l_daprime>1
            if l_daprime>2
                for ii=2:l_daprime-1
                    PolicyPath(ii,:,:,tt)=rem(ceil(PolicyPathKron(1,:,:,tt)/prod(n_daprime(1:ii-1)))-1,n_daprime(ii))+1;
                end
            end
            PolicyPath(l_daprime,:,:,tt)=ceil(PolicyPathKron(1,:,:,tt)/prod(n_daprime(1:l_daprime-1)));
        end
    end

    PolicyPath=reshape(PolicyPath,[l_daprime,n_a,n_z,T]);
end


end