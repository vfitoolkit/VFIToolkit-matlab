function Policy=UnKronPolicyIndexes1_z(PolicyKron, n_daprime1, n_a, n_z, vfoptions)
% For models with z, single daprime dimension.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(1,N_a,N_z); % indexes the optimal choice for daprime1
%        If vfoptions.gridinterplayer==1, PolicyKron is (3,N_a,N_z): row 1 is the
%        Kron index, rows 2 and 3 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1,n_a,n_z);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+2,n_a,n_z).
% Handy trick: You can pass N_a in place of n_a, or N_z in place of n_z, to skip unpacking that dimension.

l_daprime1=length(n_daprime1);

divisors=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]

if vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors),n_daprime1(:))+1;
            PolicyKron(2,:);
            PolicyKron(3,:)];
    Policy=reshape(Policy,[l_daprime1+2,n_a,n_z]);
else
    Policy=mod(floor((PolicyKron-1)./divisors),n_daprime1(:))+1;
    Policy=reshape(Policy,[l_daprime1,n_a,n_z]);
end

end
