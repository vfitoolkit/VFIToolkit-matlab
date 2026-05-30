function Policy=UnKronPolicyIndexes3_FHorz_z_e(PolicyKron, n_daprime1, n_daprime2, n_daprime3, n_a, n_z, n_e, N_j, vfoptions)
% For models with z and e, three daprime dimensions, finite horizon.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(3,N_a,N_z,N_e,N_j);
%        PolicyKron(k,:,:,:,:) indexes the optimal choice for daprime k (k=1,2,3)
%        If vfoptions.gridinterplayer==1, PolicyKron is (5,N_a,N_z,N_e,N_j): rows 1-3 are the
%        Kron indices, rows 4 and 5 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1+l_daprime2+l_daprime3,n_a,n_z,n_e,N_j);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+l_daprime2+l_daprime3+2,n_a,n_z,n_e,N_j).
% Handy trick: You can pass N_a in place of n_a, N_z in place of n_z, or N_e in place of n_e, to skip unpacking that dimension.

l_daprime1=length(n_daprime1);
l_daprime2=length(n_daprime2);
l_daprime3=length(n_daprime3);

divisors1=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]
divisors2=cumprod([1,n_daprime2(1:end-1)])';   % [l_daprime2,1]
divisors3=cumprod([1,n_daprime3(1:end-1)])';   % [l_daprime3,1]

if vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            mod(floor((PolicyKron(3,:)-1)./divisors3),n_daprime3(:))+1;
            PolicyKron(4,:);
            PolicyKron(5,:)];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+l_daprime3+2,n_a,n_z,n_e,N_j]);
else
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            mod(floor((PolicyKron(3,:)-1)./divisors3),n_daprime3(:))+1];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+l_daprime3,n_a,n_z,n_e,N_j]);
end

end
