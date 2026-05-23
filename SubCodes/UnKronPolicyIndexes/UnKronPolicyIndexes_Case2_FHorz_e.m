function Policy=UnKronPolicyIndexes_Case2_FHorz_e(PolicyKron, n_daprime, n_a, n_z,n_e,N_j, vfoptions)
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(N_a,N_z,N_e,N_j); % indexes the optimal choice for d
% Output: Policy is (l_d,n_a,n_z,n_e,N_j);
%
% Note: if you input (Policy,n_d,N_a,N_z,N_e,N_j,vfoptions) then the output only unpacks the first dimension and is (l_d,N_a,N_z,N_e,N_j)

N_a=prod(n_a);
N_z=prod(n_z);
N_e=prod(n_e);

l_daprime=length(n_daprime);

Policy=zeros(l_daprime,N_a,N_z,N_e,N_j,'gpuArray');

Policy(1,:,:,:,:)=shiftdim(rem(PolicyKron-1,n_daprime(1))+1,-1);
if l_daprime>1
    if l_daprime>2
        for ii=2:l_daprime-1
            Policy(ii,:,:,:,:)=shiftdim(rem(ceil(PolicyKron/prod(n_daprime(1:ii-1)))-1,n_daprime(ii))+1,-1);
        end
    end
    Policy(l_daprime,:,:,:,:)=shiftdim(ceil(PolicyKron/prod(n_daprime(1:l_daprime-1))),-1);
end

Policy=reshape(Policy,[l_daprime,n_a,n_z,n_e,N_j]);

end