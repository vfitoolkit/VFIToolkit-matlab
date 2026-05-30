function Policy=UnKronPolicyIndexes2_FHorz_noz(PolicyKron, n_daprime1, n_daprime2, n_a, N_j, vfoptions)
% For models with no z (and no e, no semiz), two daprime dimensions, finite horizon.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(2,N_a,N_j);
%        PolicyKron(1,:,:) indexes the optimal choice for daprime1
%        PolicyKron(2,:,:) indexes the optimal choice for daprime2
%        If vfoptions.gridinterplayer==1, PolicyKron is (4,N_a,N_j): rows 1-2 are the
%        Kron indices, rows 3 and 4 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1+l_daprime2,n_a,N_j);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+l_daprime2+2,n_a,N_j).
% Handy trick: You can pass N_a in place of n_a to skip unpacking that dimension.

l_daprime1=length(n_daprime1);
l_daprime2=length(n_daprime2);

divisors1=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]
divisors2=cumprod([1,n_daprime2(1:end-1)])';   % [l_daprime2,1]

if vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            PolicyKron(3,:);
            PolicyKron(4,:)];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+2,n_a,N_j]);
else
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1];
    Policy=reshape(Policy,[l_daprime1+l_daprime2,n_a,N_j]);
end

end
