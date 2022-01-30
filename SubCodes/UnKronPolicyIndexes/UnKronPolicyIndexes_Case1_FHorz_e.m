function Policy=UnKronPolicyIndexes_Case1_FHorz_e(Policy, n_d, n_a, n_z,n_e,N_j,vfoptions)

%Input: Policy=zeros(2,N_a,N_z,N_e,N_j); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z 
%                       (N_a,N_z,N_e,N_j) if there is no d
%Output: Policy (l_d+l_a,n_a,n_z,n_e,N_j);

N_d=prod(n_d);
N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_a=length(n_a);


% Sometimes numerical rounding errors (of the order of 10^(-16) can mean
% that Policy is not integer valued. The following corrects this by converting to int64 and then
% makes the output back into double as Matlab otherwise cannot use it in
% any arithmetical expressions.
if vfoptions.policy_forceintegertype==1
    Policy=round(Policy);
end

if N_d==0
    if vfoptions.parallel~=2
        PolicyTemp=zeros(l_a,N_a,N_z,N_e,N_j);
        for a_c=1:N_a
            for z_c=1:N_z
                for e_c=1:N_e
                    for jj=1:N_j
                        optaindexKron=Policy(a_c,z_c,e_c,jj);
                        optA=ind2sub_homemade([n_a'],optaindexKron);
                        PolicyTemp(:,a_c,z_c,e_c,jj)=[optA'];
                    end
                end
            end
        end
        Policy=reshape(PolicyTemp,[l_a,n_a,n_z,n_e,N_j]);
    else
        PolicyTemp=zeros(l_a,N_a,N_z,N_e,N_j,'gpuArray');
        
        for jj=1:N_j
            PolicyTemp(1,:,:,:,jj)=shiftdim(rem(Policy(:,:,:,jj)-1,n_a(1))+1,-1);
            if l_a>1
                if l_a>2
                    for ii=2:l_a-1
                        PolicyTemp(ii,:,:,:,jj)=shiftdim(rem(ceil(Policy(:,:,:,jj)/prod(n_a(1:ii-1)))-1,n_a(ii))+1,-1);
                    end
                end
                PolicyTemp(l_a,:,:,:,jj)=shiftdim(ceil(Policy(:,:,:,jj)/prod(n_a(1:l_a-1))),-1);
            end
        end
        
        Policy=reshape(PolicyTemp,[l_a,n_a,n_z,n_e,N_j]);
    end
        
else
    l_d=length(n_d);
    
    if vfoptions.parallel~=2
        PolicyTemp=zeros(l_d+l_a,N_a,N_z,N_e,N_j);
        for a_c=1:N_a
            for z_c=1:N_z
                for e_c=1:N_e
                    for jj=1:N_j
                        optdindexKron=Policy(1,a_c,z_c,e_c,jj);
                        optaindexKron=Policy(2,a_c,z_c,e_c,jj);
                        optD=ind2sub_homemade(n_d',optdindexKron);
                        optA=ind2sub_homemade(n_a',optaindexKron);
                        PolicyTemp(:,a_c,z_c,e_c,jj)=[optD';optA'];
                    end
                end
            end
        end
        Policy=reshape(PolicyTemp,[l_d+l_a,n_a,n_z,n_e,N_j]);
    else
        l_da=length(n_d)+length(n_a);
        n_da=[n_d,n_a];
        PolicyTemp=zeros(l_da,N_a,N_z,N_e,N_j,'gpuArray');
        
        for jj=1:N_j
            PolicyTemp(1,:,:,:,jj)=rem(Policy(1,:,:,:,jj)-1,n_da(1))+1;
            if l_d>1
                if l_d>2
                    for ii=2:l_d-1
                        PolicyTemp(ii,:,:,:,jj)=rem(ceil(Policy(1,:,:,:,jj)/prod(n_d(1:ii-1)))-1,n_d(ii))+1;
                    end
                end
                PolicyTemp(l_d,:,:,:,jj)=ceil(Policy(1,:,:,:,jj)/prod(n_d(1:l_d-1)));
            end
            
            PolicyTemp(l_d+1,:,:,:,jj)=rem(Policy(2,:,:,:,jj)-1,n_a(1))+1;
            if l_a>1
                if l_a>2
                    for ii=2:l_a-1
                        PolicyTemp(l_d+ii,:,:,:,jj)=rem(ceil(Policy(2,:,:,:,jj)/prod(n_a(1:ii-1)))-1,n_a(ii))+1;
                    end
                end
                PolicyTemp(l_da,:,:,:,jj)=ceil(Policy(2,:,:,:,jj)/prod(n_a(1:l_a-1)));
            end
        end
        
        Policy=reshape(PolicyTemp,[l_da,n_a,n_z,n_e,N_j]);
    end
end

if vfoptions.policy_forceintegertype==1
    Policy=round(Policy);
end

end