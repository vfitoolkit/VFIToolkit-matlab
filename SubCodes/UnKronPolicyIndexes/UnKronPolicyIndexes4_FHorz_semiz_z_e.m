function Policy=UnKronPolicyIndexes4_FHorz_semiz_z_e(PolicyKron, n_daprime1, n_daprime2, n_daprime3, n_daprime4, n_a, n_semiz, n_z, n_e, N_j, vfoptions)
% For models with semiz, z and e, four daprime dimensions, finite horizon.
% Can input vfoptions OR simoptions
% Input: PolicyKron=zeros(4,N_a,N_semiz,N_z,N_e,N_j);
%        PolicyKron(k,:,:,:,:,:) indexes the optimal choice for daprime k (k=1,2,3,4)
%        If vfoptions.gridinterplayer==1, PolicyKron is (6,N_a,N_semiz,N_z,N_e,N_j): rows 1-4 are the
%        Kron indices, rows 5 and 6 are L2 and L2flag (passed through unchanged).
% Output: Policy is (l_daprime1+l_daprime2+l_daprime3+l_daprime4,n_a,n_semiz,n_z,n_e,N_j);
%         If vfoptions.gridinterplayer==1, Policy is (l_daprime1+l_daprime2+l_daprime3+l_daprime4+2,n_a,n_semiz,n_z,n_e,N_j).
% Handy trick: You can pass N_a in place of n_a, N_semiz in place of n_semiz, N_z in place of n_z, or N_e in place of n_e, to skip unpacking that dimension.

l_daprime1=length(n_daprime1);
l_daprime2=length(n_daprime2);
l_daprime3=length(n_daprime3);
l_daprime4=length(n_daprime4);

divisors1=cumprod([1,n_daprime1(1:end-1)])';   % [l_daprime1,1]
divisors2=cumprod([1,n_daprime2(1:end-1)])';   % [l_daprime2,1]
divisors3=cumprod([1,n_daprime3(1:end-1)])';   % [l_daprime3,1]
divisors4=cumprod([1,n_daprime4(1:end-1)])';   % [l_daprime4,1]

if vfoptions.gridinterplayer==1
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            mod(floor((PolicyKron(3,:)-1)./divisors3),n_daprime3(:))+1;
            mod(floor((PolicyKron(4,:)-1)./divisors4),n_daprime4(:))+1;
            PolicyKron(5,:);
            PolicyKron(6,:)];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+l_daprime3+l_daprime4+2,n_a,n_semiz,n_z,n_e,N_j]);
else
    Policy=[mod(floor((PolicyKron(1,:)-1)./divisors1),n_daprime1(:))+1;
            mod(floor((PolicyKron(2,:)-1)./divisors2),n_daprime2(:))+1;
            mod(floor((PolicyKron(3,:)-1)./divisors3),n_daprime3(:))+1;
            mod(floor((PolicyKron(4,:)-1)./divisors4),n_daprime4(:))+1];
    Policy=reshape(Policy,[l_daprime1+l_daprime2+l_daprime3+l_daprime4,n_a,n_semiz,n_z,n_e,N_j]);
end

end
