function Policy=UnKronPolicyIndexes_Case2(Policy, n_d, n_a, n_z,vfoptions)

%PolicyIndexesKron=zeros(N_a,N_z); %first dim indexes the optimal choice for d and aprime rest of dimensions a,z
%PolicyIndexesIntermediate (l_d,N_a,N_z)
%PolicyIndexes (l_d,n_a,n_z);

N_a=prod(n_a);
N_z=prod(n_z);

l_d=length(n_d);

if vfoptions.parallel~=2
    PolicyTemp=zeros(l_d,N_a,N_z);
    for i=1:N_a
        for j=1:N_z
            optdindexKron=Policy(i,j);
            optD=ind2sub_homemade(n_d',optdindexKron);
            PolicyTemp(:,i,j)=[optD'];
        end
    end
    Policy=reshape(PolicyTemp,[l_d,n_a,n_z]);
else
    PolicyTemp=zeros(l_d,N_a,N_z,'gpuArray');
    
    PolicyTemp(1,:,:)=shiftdim(rem(Policy-1,n_d(1))+1,-1);
    if l_d>1
        if l_d>2
            for ii=2:l_d-1
                PolicyTemp(ii,:,:)=shiftdim(rem(ceil(Policy/prod(n_d(1:ii-1)))-1,n_d(ii))+1,-1);
            end
        end
        PolicyTemp(l_d,:,:)=shiftdim(ceil(Policy/prod(n_d(1:l_d-1))),-1);
    end
    
    Policy=reshape(PolicyTemp,[l_d,n_a,n_z]);
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