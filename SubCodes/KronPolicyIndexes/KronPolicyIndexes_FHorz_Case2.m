function PolicyKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z,N_j,options)

%Input: Policy (l_d,n_a,n_z,N_j);

%Output: Policy=zeros(N_a,N_z,N_j); % contains indexes for the optimal choice for d 

N_a=prod(n_a);
N_z=prod(n_z);

Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);

%% Note: I could probably shave a few fractions of a second of runtime by actually copy-pasting code rather than calling 'KronPolicyIndexes_Case1' each time.
if options.parallel~=2
    PolicyKron=zeros(N_a,N_z,N_j);
    for jj=1:N_j
        PolicyKron(:,:,jj)=KronPolicyIndexes_Case2(Policy(:,:,:,jj), n_d, n_a, n_z,options);
    end
else
    PolicyKron=zeros(N_a,N_z,N_j,'gpuArray');
    for jj=1:N_j
        PolicyKron(:,:,jj)=KronPolicyIndexes_Case2(Policy(:,:,:,jj), n_d, n_a, n_z,options);
    end
end


end