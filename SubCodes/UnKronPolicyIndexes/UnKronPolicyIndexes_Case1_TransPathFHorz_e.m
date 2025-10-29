function Policy=UnKronPolicyIndexes_Case1_TransPathFHorz_e(Policy, n_d, n_a, n_z,n_e,N_j,T,vfoptions)
% can input vfoptions OR simoptions
%Input: Policy=zeros(2,N_a,N_z,N_e,N_j,T); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,N_j,T) if there is no d
%Output: Policy (l_d+l_a,n_a,n_z,n_e,N_j,T);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_a=length(n_a);
if vfoptions.gridinterplayer==0
    l_aprime=l_a;
    n_aprime=n_a;
elseif vfoptions.gridinterplayer==1
    l_aprime=l_a+1;
    n_aprime=[n_a,vfoptions.ngridinterp+2];
end

if N_d==0

    PolicyTemp=zeros(l_aprime,N_a,N_z,N_e,N_j,T,'gpuArray');

    for jj=1:N_j
        for tt=1:T
            PolicyTemp(1,:,:,:,jj,tt)=shiftdim(rem(Policy(:,:,:,jj,tt)-1,n_aprime(1))+1,-1);
            if l_aprime>1
                if l_aprime>2
                    for ii=2:l_aprime-1
                        PolicyTemp(ii,:,:,:,jj,tt)=shiftdim(rem(ceil(Policy(:,:,:,jj,tt)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1,-1);
                    end
                end
                PolicyTemp(l_aprime,:,:,:,jj,tt)=shiftdim(ceil(Policy(:,:,:,jj,tt)/prod(n_aprime(1:l_aprime-1))),-1);
            end
        end
    end

    Policy=reshape(PolicyTemp,[l_aprime,n_a,n_z,n_e,N_j,T]);
        
else
    l_d=length(n_d);
    l_daprime=length(n_d)+l_aprime;
    PolicyTemp=zeros(l_daprime,N_a,N_z,N_e,N_j,T,'gpuArray');

    for jj=1:N_j
        for tt=1:T
            PolicyTemp(1,:,:,:,jj,tt)=rem(Policy(1,:,:,:,jj,tt)-1,n_d(1))+1;
            if l_d>1
                if l_d>2
                    for ii=2:l_d-1
                        PolicyTemp(ii,:,:,:,jj,tt)=rem(ceil(Policy(1,:,:,:,jj,tt)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
                    end
                end
                PolicyTemp(l_d,:,:,:,jj,tt)=ceil(Policy(1,:,:,:,jj,tt)/prod(n_d(1:l_d-1)));
            end

            PolicyTemp(l_d+1,:,:,:,jj,tt)=rem(Policy(2,:,:,:,jj,tt)-1,n_aprime(1))+1;
            if l_aprime>1
                if l_aprime>2
                    for ii=2:l_aprime-1
                        PolicyTemp(l_d+ii,:,:,:,jj,tt)=rem(ceil(Policy(2,:,:,:,jj,tt)/prod(n_aprime(1:ii-1)))-1,n_aprime(ii))+1;
                    end
                end
                PolicyTemp(l_daprime,:,:,:,jj,tt)=ceil(Policy(2,:,:,:,jj,tt)/prod(n_aprime(1:l_aprime-1)));
            end
        end
    end

    Policy=reshape(PolicyTemp,[l_daprime,n_a,n_z,n_e,N_j,T]);
end

% % Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% % that Policy is not integer valued. The following corrects this by converting to int64 and then
% % makes the output back into double as Matlab otherwise cannot use it in
% % any arithmetical expressions.
% if vfoptions.policy_forceintegertype==1
%     Policy=uint64(Policy);
%     Policy=double(Policy);
% end


end