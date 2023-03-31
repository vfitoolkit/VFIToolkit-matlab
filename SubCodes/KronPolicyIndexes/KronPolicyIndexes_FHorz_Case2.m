function PolicyKron=KronPolicyIndexes_FHorz_Case2(Policy, n_d, n_a, n_z,N_j, n_e)
% n_e is an optional input
%
% Input: Policy (l_d,n_a,n_z,N_j);
% 
% Output: Policy=zeros(N_a,N_z,N_j); % contains indexes for the optimal choice for d 

N_a=prod(n_a);
N_z=prod(n_z);

% When using n_e, is instead:
% Input: Policy (l_d,n_a,n_z,n_e,,N_j);
%
% Output: Policy=zeros(N_a,N_z,N_e,N_j);

if ~exist('n_e','var')
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_j]);
    
    %% Note: I could probably shave a few fractions of a second of runtime by actually copy-pasting code rather than calling 'KronPolicyIndexes_Case1' each time.
    if isa(Policy,'gpuArray') %options.parallel~=2
        PolicyKron=zeros(N_a,N_z,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,jj)=KronPolicyIndexes_Case2(Policy(:,:,:,jj), n_d, n_a, n_z); %,options);
        end
    else
        PolicyKron=zeros(N_a,N_z,N_j);
        for jj=1:N_j
            PolicyKron(:,:,jj)=KronPolicyIndexes_Case2(Policy(:,:,:,jj), n_d, n_a, n_z); %,options);
        end
    end
else
    N_e=prod(n_e);
    Policy=reshape(Policy,[size(Policy,1),N_a,N_z,N_e,N_j]);
    
    %% Note: I could probably shave a few fractions of a second of runtime by actually copy-pasting code rather than calling 'KronPolicyIndexes_Case1' each time.
    if isa(Policy,'gpuArray') %options.parallel~=2
        PolicyKron=zeros(N_a,N_z,N_e,N_j,'gpuArray');
        for jj=1:N_j
            PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case2_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e); %,options);
        end
    else
        PolicyKron=zeros(N_a,N_z,N_e,N_j);
        for jj=1:N_j
            PolicyKron(:,:,:,jj)=KronPolicyIndexes_Case2_e(Policy(:,:,:,:,jj), n_d, n_a, n_z, n_e); %,options);
        end
    end
end

end