function Policy=UnKronPolicyIndexes_Case2_FHorz_noz(Policy, n_d, n_a,N_j, vfoptions)

%Input: Policy (N_a,N_j); 
%PolicyIndexesIntermediate (l_d,N_a,N_j)
%Output: Policy (l_d,n_a,N_j);

N_a=prod(n_a);

l_d=length(n_d);

if vfoptions.parallel~=2
    PolicyTemp=zeros(l_d,N_a,N_j);
    for i=1:N_a
        for k=1:N_j
            optdindexKron=Policy(i,k);
            optD=ind2sub_homemade(n_d',optdindexKron);
            PolicyTemp(:,i,k)=[optD'];
        end
    end
    Policy=reshape(PolicyTemp,[l_d,n_a,N_j]);
else
    PolicyTemp=zeros(l_d,N_a,N_j,'gpuArray');
    
    PolicyTemp(1,:,:)=shiftdim(rem(Policy-1,n_d(1))+1,-1);
    if l_d>1
        if l_d>2
            for ii=2:l_d-1
                PolicyTemp(ii,:,:)=shiftdim(rem(ceil(Policy/prod(n_d(1:ii-1)))-1,n_d(ii))+1,-1);
            end
        end
        PolicyTemp(l_d,:,:)=shiftdim(ceil(Policy/prod(n_d(1:l_d-1))),-1);
    end
    
    Policy=reshape(PolicyTemp,[l_d,n_a,N_j]);
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