function Policy=UnKronPolicyIndexes1_FHorz_z_e(PolicyKron, n_daprime1, n_a, n_z, n_e, N_j, vfoptions)
% For models with z and e, single daprime dimension, finite horizon.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(1,N_a,N_z,N_e,N_j); % indexes the optimal choice for daprime1
%        If vfoptions.gridinterplayer==1, PolicyKron is (3,N_a,N_z,N_e,N_j): row 1 is the
%        Kron index, rows 2 and 3 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1,n_a,n_z,n_e,N_j);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+2,n_a,n_z,n_e,N_j).
% Handy trick: You can pass N_a in place of n_a, N_z in place of n_z, or N_e in place of n_e, to skip unpacking that dimension.

l_daprime1=length(n_daprime1);

divisors=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]

if vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors),n_daprime1(:))+1;
            PolicyKron(2,:);
            PolicyKron(3,:)];
    Policy=reshape(Policy,[l_daprime1+2,n_a,n_z,n_e,N_j]);
else
    Policy=mod(floor((PolicyKron-1)./divisors),n_daprime1(:))+1;
    Policy=reshape(Policy,[l_daprime1,n_a,n_z,n_e,N_j]);
end

end
