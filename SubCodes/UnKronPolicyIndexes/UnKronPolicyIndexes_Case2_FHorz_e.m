function Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_d, n_a, n_z,n_e,N_j, vfoptions)
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(N_a,N_z,N_j); % indexes the optimal choice for d
% Output: Policy is (l_d,n_a,n_z,N_j);

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_d=length(n_d);

Policy=zeros(l_d,N_a,N_z,N_e,N_j,'gpuArray');

Policy(1,:,:,:,:)=shiftdim(rem(PolicyKron-1,n_d(1))+1,-1);
if l_d>1
    if l_d>2
        for ii=2:l_d-1
            Policy(ii,:,:,:,:)=shiftdim(rem(ceil(PolicyKron/prod(n_d(1:ii-1)))-1,n_d(ii))+1,-1);
        end
    end
    Policy(l_d,:,:,:,:)=shiftdim(ceil(PolicyKron/prod(n_d(1:l_d-1))),-1);
end

Policy=reshape(Policy,[l_d,n_a,n_z,n_e,N_j]);

end